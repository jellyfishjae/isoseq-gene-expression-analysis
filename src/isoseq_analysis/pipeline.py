"""Run the analysis and save figures, tables, and a summary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_dataset
from .plots import (
    SYNTHETIC_OUTPUT,
    plot_confounding,
    plot_group_signal_heatmap,
    plot_metadata_associations,
    plot_pca,
    plot_qc_overview,
    plot_sample_correlation,
    plot_scree,
    plot_sensitivity,
    plot_top_loadings,
)
from .preprocess import calculate_qc, fit_pca, prepare_expression
from .stats import (
    centroid_permutation_test,
    confounding_diagnostic,
    exploratory_group_signals,
    pc_metadata_associations,
    preprocessing_sensitivity,
)


@dataclass(frozen=True)
class AnalysisConfig:
    expression_path: Path
    clinical_path: Path
    output_root: Path
    min_tpm: float = 1.0
    min_sample_fraction: float = 0.20
    top_variable_genes: int = 5_000
    pca_components: int = 10
    permutations: int = 999
    random_state: int = 42
    synthetic: bool = False


def _top_loadings(loadings: pd.DataFrame, genes: pd.DataFrame, per_direction: int = 10) -> pd.DataFrame:
    rows = []
    for component in loadings.columns[:3]:
        series = loadings[component]
        selected = pd.concat([series.nsmallest(per_direction), series.nlargest(per_direction)])
        frame = selected.rename("Loading").reset_index()
        frame.insert(0, "Component", component)
        frame["Gene_Symbol"] = frame["Gene_ID"].map(genes["Gene_Symbol"])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _group_qc_summary(qc: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in ["Read_Depth", "Detected_Genes", "Zero_Fraction", "Mitochondrial_Read_Fraction", "RIN", "PMI", "Age"]
        if column in qc
    ]
    return qc.groupby("Group", observed=True)[numeric].agg(["count", "mean", "std", "min", "max"])


def _result_markdown(
    config: AnalysisConfig,
    sample_count: int,
    gene_count: int,
    selected_count: int,
    group_counts: pd.Series,
    explained: pd.DataFrame,
    associations: pd.DataFrame,
    permanova: dict[str, float | int],
    confounding: dict[str, float],
    signals: pd.DataFrame,
) -> str:
    variance = explained.set_index("Component")["Explained_Variance_Ratio"]

    def association(variable: str) -> str:
        rows = associations.loc[
            associations["Component"].eq("PC1") & associations["Variable"].eq(variable)
        ]
        if rows.empty:
            return "not available"
        row = rows.iloc[0]
        effect_label = "rho" if row["Variable_Type"] == "continuous" else "eta-squared"
        return (f"{effect_label} = {row['Effect']:.2f}, p = {row['P_Value']:.3g}, "
                f"BH q = {row['Q_Value']:.3g}")

    counts = ", ".join(f"{group}: {count}" for group, count in group_counts.items())
    significant = int((signals["Q_Value"] < 0.05).sum())
    label = "SYNTHETIC DATA - demonstration only." if config.synthetic else "Exploratory research analysis."
    batch = "Candidate-batch association could not be calculated."
    if confounding:
        batch = (f"Candidate-batch Cramer's V = {confounding['cramers_v']:.2f}; "
                 f"approximate chi-squared p = {confounding['p_value']:.3g}.")
        if confounding["small_expected_counts"]:
            batch += " Expected cell counts below five make this p-value unreliable."
    return f"""# Results

{label}

## Data

{sample_count} samples ({counts}); {gene_count:,} input genes.
PCA uses {selected_count:,} high-MAD genes after filtering and log2(TPM + 1).
Samples are relabeled; this does not establish anonymity.

## PCA and metadata

PC1 explains {variance['PC1']:.1%} of the selected-gene variance.

| PC1 comparison | Result |
| --- | --- |
| RIN | {association("RIN")} |
| Mitochondrial read fraction | {association("Mitochondrial_Read_Fraction")} |
| Pathology group | {association("Group")} |

These are associations, not evidence of what caused the component.

## Group comparisons

The unadjusted permutation test gives pseudo-F = {permanova['pseudo_f']:.2f},
R-squared = {permanova['r_squared']:.2%}, and p = {permanova['p_value']:.3g}
({permanova['permutations']} permutations).
Confounding and differences in within-group dispersion can affect this test.
A nonsignificant result does not rule out disease effects.

Of {len(signals):,} genes tested, {significant} have BH q < 0.05.
Correction and counts use all tested genes; the CSV contains the top 50.
These are exploratory rankings, without covariate adjustment.

## Candidate batches

{batch}

Labels are parsed from report identifiers in the Iso-seq field. Confirm that these
represent sequencing batches before interpreting the table as batch confounding.
No batch correction is applied.

## Run settings

Input filenames and parameters are recorded in ../results/run_summary.json.
See the repository README for matching demo and research commands.
"""


def run_analysis(config: AnalysisConfig) -> dict[str, object]:
    if config.permutations < 1 or config.pca_components < 3:
        raise ValueError("Use at least one permutation and three PCA components")
    token = SYNTHETIC_OUTPUT.set(config.synthetic)
    try:
        return _run_analysis(config)
    finally:
        SYNTHETIC_OUTPUT.reset(token)


def _run_analysis(config: AnalysisConfig) -> dict[str, object]:
    output_root = Path(config.output_root)
    figures = output_root / "figures"
    results = output_root / "results"
    docs = output_root / "docs"
    for directory in (figures, results, docs):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.expression_path, config.clinical_path)
    qc = calculate_qc(dataset, detection_tpm=config.min_tpm)
    matrix, _ = prepare_expression(
        dataset.tpm,
        min_tpm=config.min_tpm,
        min_sample_fraction=config.min_sample_fraction,
        top_variable_genes=config.top_variable_genes,
    )
    pca = fit_pca(matrix, n_components=config.pca_components)
    associations = pc_metadata_associations(pca.scores, dataset.metadata, qc)
    loadings = _top_loadings(pca.loadings, dataset.genes)

    group_labels = dataset.metadata["Group"]
    permanova = centroid_permutation_test(
        matrix,
        group_labels,
        permutations=config.permutations,
        random_state=config.random_state,
    )
    confounding_table, confounding = confounding_diagnostic(dataset.metadata)
    signals = exploratory_group_signals(matrix, group_labels, dataset.genes)

    minimum_samples = max(2, int(np.ceil(dataset.tpm.shape[1] * config.min_sample_fraction)))
    expressed = (dataset.tpm >= config.min_tpm).sum(axis=1) >= minimum_samples
    log_filtered = np.log2(dataset.tpm.loc[expressed].T + 1.0)
    log_filtered = log_filtered.loc[:, log_filtered.var() > 0]
    sensitivity = preprocessing_sensitivity(
        log_filtered,
        reference_genes=config.top_variable_genes,
        gene_counts=(2_000, 5_000, 10_000),
    )

    plot_qc_overview(qc, figures / "01_qc_overview.png")
    plot_scree(pca.explained_variance, figures / "02_pca_scree.png")
    plot_pca(pca.scores, pca.explained_variance, dataset.metadata, figures / "03_pca_by_group.png")
    plot_sample_correlation(matrix, dataset.metadata, figures / "04_sample_correlation.png")
    plot_metadata_associations(associations, figures / "05_pc_metadata_associations.png")
    plot_top_loadings(loadings, figures / "06_top_pc_loadings.png")
    plot_group_signal_heatmap(matrix, signals, dataset.metadata, figures / "07_group_signal_heatmap.png")
    plot_confounding(confounding_table, figures / "08_group_batch_confounding.png")
    plot_sensitivity(sensitivity, figures / "09_preprocessing_sensitivity.png")

    pca.explained_variance.to_csv(results / "pca_explained_variance.csv", index=False)
    associations.to_csv(results / "pc_metadata_associations.csv", index=False)
    loadings.to_csv(results / "top_pc_loadings.csv", index=False)
    signals.head(50).to_csv(results / "exploratory_group_signals.csv", index=False)
    sensitivity.to_csv(results / "preprocessing_sensitivity.csv", index=False)
    _group_qc_summary(qc).to_csv(results / "qc_summary_by_group.csv")
    confounding_table.to_csv(results / "group_by_batch_counts.csv")

    summary: dict[str, object] = {
        "inputs": {
            "expression_file": Path(config.expression_path).name,
            "clinical_file": Path(config.clinical_path).name,
        },
        "parameters": {
            key: value
            for key, value in asdict(config).items()
            if key not in {"expression_path", "clinical_path", "output_root"}
        },
        "dataset": {
            "samples": dataset.tpm.shape[1],
            "input_genes": dataset.tpm.shape[0],
            "expression_filtered_genes": int(expressed.sum()),
            "pca_genes": matrix.shape[1],
            "group_counts": group_labels.value_counts().to_dict(),
        },
        "gene_tests": {"tested": len(signals), "bh_q_below_0_05": int((signals["Q_Value"] < 0.05).sum())},
        "data_kind": "synthetic" if config.synthetic else "research",
        "pathology_group_permutation_test": permanova,
        "group_candidate_batch_diagnostic": confounding,
        "privacy": "Source sample identifiers and row-level clinical data were not exported.",
    }
    with (results / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report = _result_markdown(
        config=config,
        sample_count=dataset.tpm.shape[1],
        gene_count=dataset.tpm.shape[0],
        selected_count=matrix.shape[1],
        group_counts=group_labels.value_counts(),
        explained=pca.explained_variance,
        associations=associations,
        permanova=permanova,
        confounding=confounding,
        signals=signals,
    )
    (docs / "RESULTS.md").write_text(report, encoding="utf-8")
    return summary
