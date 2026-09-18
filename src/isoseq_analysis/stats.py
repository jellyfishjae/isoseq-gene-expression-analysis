"""Small-sample statistical diagnostics used by the analysis pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from .preprocess import median_absolute_deviation


def benjamini_hochberg(p_values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    valid = values[finite]
    order = np.argsort(valid)
    ranked = valid[order]
    correction = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    correction = np.minimum.accumulate(correction[::-1])[::-1]
    restored = np.empty_like(correction)
    restored[order] = np.clip(correction, 0.0, 1.0)
    adjusted[finite] = restored
    return adjusted


def categorical_effect(values: np.ndarray, labels: pd.Series) -> float:
    """Return eta-squared: fraction of score variance explained by labels."""

    frame = pd.DataFrame({"value": values, "label": labels}).dropna()
    if frame.empty or frame["label"].nunique() < 2:
        return np.nan
    grand = frame["value"].mean()
    total = np.square(frame["value"] - grand).sum()
    if total == 0:
        return 0.0
    between = sum(
        len(group) * np.square(group["value"].mean() - grand)
        for _, group in frame.groupby("label", observed=True)
    )
    return float(between / total)


def pc_metadata_associations(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    qc: pd.DataFrame,
    max_components: int = 5,
) -> pd.DataFrame:
    """Test exploratory associations between leading PCs and sample attributes."""

    combined = metadata.join(qc.drop(columns=metadata.columns, errors="ignore"))
    continuous_candidates = [
        "Age",
        "PMI",
        "RIN",
        "Read_Depth",
        "Detected_Genes",
        "Mitochondrial_Read_Fraction",
    ]
    categorical_candidates = ["Group", "Sex", "APOE", "Candidate_Batch"]
    rows: list[dict[str, object]] = []

    for component in scores.columns[:max_components]:
        for variable in continuous_candidates:
            if variable not in combined:
                continue
            frame = pd.DataFrame({"score": scores[component], "value": combined[variable]}).dropna()
            if len(frame) < 4 or frame["value"].nunique() < 2:
                continue
            rho, p_value = spearmanr(frame["score"], frame["value"])
            rows.append(
                {
                    "Component": component,
                    "Variable": variable,
                    "Variable_Type": "continuous",
                    "Effect": float(rho),
                    "Effect_Magnitude": abs(float(rho)),
                    "Test": "Spearman correlation",
                    "P_Value": float(p_value),
                    "N": len(frame),
                }
            )

        for variable in categorical_candidates:
            if variable not in combined:
                continue
            frame = pd.DataFrame({"score": scores[component], "value": combined[variable]}).dropna()
            groups = [part["score"].to_numpy() for _, part in frame.groupby("value", observed=True)]
            if len(groups) < 2 or min(map(len, groups)) < 2:
                continue
            statistic, p_value = kruskal(*groups)
            effect = categorical_effect(frame["score"].to_numpy(), frame["value"])
            rows.append(
                {
                    "Component": component,
                    "Variable": variable,
                    "Variable_Type": "categorical",
                    "Effect": effect,
                    "Effect_Magnitude": effect,
                    "Test": "Kruskal-Wallis",
                    "P_Value": float(p_value),
                    "N": len(frame),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["Q_Value"] = benjamini_hochberg(result["P_Value"])
    return result


def centroid_permutation_test(
    matrix: pd.DataFrame,
    labels: pd.Series,
    permutations: int = 999,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Euclidean PERMANOVA-style test of group centroid separation."""

    frame_labels = labels.reindex(matrix.index)
    valid = frame_labels.notna()
    x = matrix.loc[valid].to_numpy(dtype=float)
    y = frame_labels.loc[valid].to_numpy()
    groups = np.unique(y)
    n, group_count = len(y), len(groups)
    if group_count < 2 or n <= group_count:
        raise ValueError("Permutation test requires at least two groups and residual degrees of freedom")

    def pseudo_f(current_labels: np.ndarray) -> tuple[float, float]:
        grand = x.mean(axis=0)
        total = np.square(x - grand).sum()
        within = sum(
            np.square(x[current_labels == group] - x[current_labels == group].mean(axis=0)).sum()
            for group in np.unique(current_labels)
        )
        between = total - within
        statistic = (between / (group_count - 1)) / (within / (n - group_count))
        return float(statistic), float(between / total)

    observed, r_squared = pseudo_f(y)
    rng = np.random.default_rng(random_state)
    permuted = np.array([pseudo_f(rng.permutation(y))[0] for _ in range(permutations)])
    p_value = (1 + np.count_nonzero(permuted >= observed)) / (permutations + 1)
    return {
        "pseudo_f": observed,
        "r_squared": r_squared,
        "p_value": float(p_value),
        "permutations": permutations,
        "n_samples": n,
        "n_groups": group_count,
    }


def confounding_diagnostic(metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if "Group" not in metadata or "Candidate_Batch" not in metadata:
        return pd.DataFrame(), {}
    table = pd.crosstab(metadata["Candidate_Batch"], metadata["Group"])
    if min(table.shape, default=0) < 2:
        return table, {}
    chi2, p_value, _, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = np.sqrt((chi2 / n) / denominator) if denominator > 0 else np.nan
    return table, {"chi2": float(chi2), "p_value": float(p_value), "cramers_v": float(cramers_v), "small_expected_counts": bool((expected < 5).any())}


def exploratory_group_signals(
    matrix: pd.DataFrame,
    groups: pd.Series,
    gene_annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Rank genes by unadjusted multi-group signal; results are hypothesis-generating."""

    labels = groups.reindex(matrix.index)
    unique_groups = labels.dropna().unique()
    rows: list[dict[str, object]] = []
    for gene in matrix.columns:
        values = matrix[gene]
        arrays = [values[labels == group].to_numpy() for group in unique_groups]
        if np.allclose(values, values.iloc[0]):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = kruskal(*arrays)
        row: dict[str, object] = {
            "Gene_ID": gene,
            "Gene_Symbol": gene_annotations.at[gene, "Gene_Symbol"],
            "Kruskal_H": float(statistic),
            "P_Value": float(p_value),
            "Eta_Squared": categorical_effect(values.to_numpy(), labels),
            "Max_Mean_Difference": float(
                max(array.mean() for array in arrays) - min(array.mean() for array in arrays)
            ),
        }
        for group, array in zip(unique_groups, arrays):
            row[f"Mean_{group}"] = float(array.mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    result["Q_Value"] = benjamini_hochberg(result["P_Value"])
    return result.sort_values(["Q_Value", "P_Value", "Eta_Squared"], ascending=[True, True, False])


def preprocessing_sensitivity(
    log_expression: pd.DataFrame,
    reference_genes: int = 5_000,
    gene_counts: tuple[int, ...] = (2_000, 5_000, 10_000),
) -> pd.DataFrame:
    """Compare sample geometry under several high-variability gene cutoffs."""

    variability = median_absolute_deviation(log_expression).sort_values(ascending=False)

    def distances(gene_count: int) -> np.ndarray:
        selected = variability.head(min(gene_count, len(variability))).index
        component_count = min(5, log_expression.shape[0] - 1, len(selected))
        scores = PCA(n_components=component_count, svd_solver="full").fit_transform(
            log_expression[selected]
        )
        return pairwise_distances(scores)[np.triu_indices(len(scores), k=1)]

    reference_genes = min(reference_genes, len(variability))
    reference = distances(reference_genes)
    rows = []
    for gene_count in sorted({min(count, len(variability)) for count in gene_counts + (reference_genes,)}):
        candidate = distances(gene_count)
        rho, _ = spearmanr(reference, candidate)
        rows.append({"Top_MAD_Genes": gene_count, "Reference_Genes": reference_genes, "Distance_Spearman": float(rho)})
    return pd.DataFrame(rows)
