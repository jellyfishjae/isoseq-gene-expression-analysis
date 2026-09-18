"""QC and expression plots."""

from __future__ import annotations

from pathlib import Path

from contextvars import ContextVar

SYNTHETIC_OUTPUT = ContextVar("synthetic_output", default=False)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = {
    "ADNC": "#D55E00",
    "ADNC+LBD": "#CC79A7",
    "LBD": "#0072B2",
    "Control (PART)": "#009E73",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180, "font.family": "DejaVu Sans"})


def _save(figure: plt.Figure, destination: Path) -> None:
    if SYNTHETIC_OUTPUT.get():
        figure.text(0.5, 1.09, "SYNTHETIC DATA - demonstration only",
                    ha="center", fontsize=11, weight="bold", color="#8B4513")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_qc_overview(qc: pd.DataFrame, destination: Path) -> None:
    _style()
    metrics = [
        ("Read_Depth", "Read depth", 1e6, "millions of reads"),
        ("Detected_Genes", "Detected genes", 1.0, "genes with TPM ≥ 1"),
        ("Mitochondrial_Read_Fraction", "Mitochondrial fraction", 0.01, "% of reads"),
    ]
    metrics = [metric for metric in metrics if metric[0] in qc]
    figure, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8))
    if len(metrics) == 1:
        axes = [axes]
    colors = qc.get("Group", pd.Series("Samples", index=qc.index)).map(PALETTE).fillna("#666666")
    for axis, (column, title, divisor, ylabel) in zip(axes, metrics):
        values = qc[column] / divisor
        axis.bar(qc.index, values, color=colors, edgecolor="white", linewidth=0.5)
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=90, labelsize=7)
    figure.suptitle("Per-sample quality control", fontsize=15, fontweight="bold", y=1.03)
    figure.tight_layout()
    _save(figure, destination)


def plot_scree(explained: pd.DataFrame, destination: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(8, 4.8))
    percent = explained["Explained_Variance_Ratio"] * 100
    axis.bar(explained["Component"], percent, color="#4C78A8")
    axis.plot(explained["Component"], percent, color="#1F3552", marker="o", linewidth=1.5)
    axis.set_ylabel("Explained variance (%)")
    axis.set_xlabel("Principal component")
    axis.set_title("PCA scree plot", fontweight="bold")
    _save(figure, destination)


def plot_pca(scores: pd.DataFrame, explained: pd.DataFrame, metadata: pd.DataFrame, destination: Path) -> None:
    _style()
    joined = scores.join(metadata)
    pairs = [("PC1", "PC2"), ("PC1", "PC3")]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    variance = explained.set_index("Component")["Explained_Variance_Ratio"] * 100
    for axis, (x_component, y_component) in zip(axes, pairs):
        for group, subset in joined.groupby("Group", observed=True):
            axis.scatter(
                subset[x_component],
                subset[y_component],
                s=70,
                alpha=0.88,
                color=PALETTE.get(group, "#666666"),
                edgecolor="white",
                linewidth=0.7,
                label=group,
            )
            for sample, row in subset.iterrows():
                axis.annotate(sample, (row[x_component], row[y_component]), xytext=(4, 3),
                              textcoords="offset points", fontsize=7, alpha=0.8)
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.axvline(0, color="#BBBBBB", linewidth=0.7)
        axis.set_xlabel(f"{x_component} ({variance[x_component]:.1f}%)")
        axis.set_ylabel(f"{y_component} ({variance[y_component]:.1f}%)")
        axis.set_title(f"{x_component} vs {y_component}", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle("Gene-expression PCA by pathology group", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    _save(figure, destination)


def plot_sample_correlation(matrix: pd.DataFrame, metadata: pd.DataFrame, destination: Path) -> None:
    _style()
    order = metadata.sort_values(["Group"], kind="stable").index
    correlation = matrix.loc[order].T.corr(method="spearman")
    figure, axis = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0.85,
        vmin=0.55,
        vmax=1.0,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.8},
        ax=axis,
    )
    axis.set_title("Sample-to-sample expression correlation", fontweight="bold", pad=12)
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    axis.tick_params(axis="y", rotation=0, labelsize=7)
    _save(figure, destination)


def plot_metadata_associations(associations: pd.DataFrame, destination: Path) -> None:
    _style()
    data = associations.copy()
    data["Signed_Effect"] = np.where(
        data["Variable_Type"].eq("continuous"), data["Effect"], data["Effect_Magnitude"]
    )
    matrix = data.pivot(index="Variable", columns="Component", values="Signed_Effect")
    component_order = [component for component in ["PC1", "PC2", "PC3", "PC4", "PC5"] if component in matrix]
    matrix = matrix[component_order]
    q_values = data.pivot(index="Variable", columns="Component", values="Q_Value").reindex_like(matrix)
    annotations = matrix.copy().astype(object)
    for row in matrix.index:
        for column in matrix.columns:
            value = matrix.at[row, column]
            q_value = q_values.at[row, column]
            star = "*" if pd.notna(q_value) and q_value < 0.05 else ""
            annotations.at[row, column] = "" if pd.isna(value) else f"{value:.2f}{star}"
    figure, axis = plt.subplots(figsize=(8.2, max(4.5, 0.52 * len(matrix))))
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annotations,
        fmt="",
        linewidths=0.6,
        cbar_kws={"label": "ρ (continuous) or η² (categorical)"},
        ax=axis,
    )
    axis.set_title("Principal-component associations with sample attributes", fontweight="bold", pad=12)
    axis.set_xlabel("")
    axis.set_ylabel("")
    figure.text(0.01, 0.01, "* Benjamini–Hochberg q < 0.05", fontsize=8, color="#555555")
    _save(figure, destination)


def plot_top_loadings(loadings: pd.DataFrame, destination: Path) -> None:
    _style()
    components = [component for component in ["PC1", "PC2", "PC3"] if component in loadings["Component"].unique()]
    figure, axes = plt.subplots(1, len(components), figsize=(5.2 * len(components), 7.8))
    if len(components) == 1:
        axes = [axes]
    for axis, component in zip(axes, components):
        subset = loadings.loc[loadings["Component"].eq(component)].copy()
        subset = subset.sort_values("Loading")
        labels = subset["Gene_Symbol"].fillna(subset["Gene_ID"])
        colors = np.where(subset["Loading"] >= 0, "#4C78A8", "#E45756")
        axis.barh(labels, subset["Loading"], color=colors)
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set_title(component, fontweight="bold")
        axis.set_xlabel("Loading")
        axis.tick_params(axis="y", labelsize=8)
    figure.suptitle("Genes with the strongest PCA loadings", fontsize=15, fontweight="bold", y=1.01)
    figure.tight_layout()
    _save(figure, destination)


def plot_group_signal_heatmap(
    matrix: pd.DataFrame,
    signals: pd.DataFrame,
    metadata: pd.DataFrame,
    destination: Path,
    gene_count: int = 20,
) -> None:
    _style()
    genes = signals.head(gene_count)["Gene_ID"]
    labels = signals.set_index("Gene_ID")["Gene_Symbol"].reindex(genes).fillna(pd.Series(genes, index=genes))
    order = metadata.sort_values("Group", kind="stable").index
    values = matrix.loc[order, genes].T
    standardized = values.sub(values.mean(axis=1), axis=0).div(values.std(axis=1).replace(0, 1), axis=0)
    standardized.index = labels
    figure, axis = plt.subplots(figsize=(11, 7.2))
    sns.heatmap(
        standardized,
        cmap="vlag",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Gene-wise z-score"},
        ax=axis,
    )
    axis.set_title("Top exploratory pathology-group signals", fontweight="bold", pad=12)
    axis.set_xlabel("De-identified samples (ordered by pathology group)")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    axis.tick_params(axis="y", rotation=0, labelsize=8)
    _save(figure, destination)


def plot_confounding(table: pd.DataFrame, destination: Path) -> None:
    if table.empty:
        return
    _style()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", linewidths=0.8, cbar=False, ax=axis)
    axis.set_title("Pathology group × candidate batch", fontweight="bold", pad=12)
    axis.set_xlabel("Pathology group")
    axis.set_ylabel("Candidate batch (unconfirmed)")
    _save(figure, destination)


def plot_sensitivity(sensitivity: pd.DataFrame, destination: Path) -> None:
    _style()
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.bar(
        sensitivity["Top_MAD_Genes"].astype(str),
        sensitivity["Distance_Spearman"],
        color="#59A14F",
    )
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Number of high-MAD genes")
    reference = int(sensitivity["Reference_Genes"].iloc[0])
    axis.set_ylabel(f"Distance correlation vs {reference:,}-gene analysis")
    axis.set_title("PCA preprocessing sensitivity", fontweight="bold")
    for index, value in enumerate(sensitivity["Distance_Spearman"]):
        axis.text(index, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    _save(figure, destination)
