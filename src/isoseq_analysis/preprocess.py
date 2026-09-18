"""Quality control, filtering, transformation, and dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .io import ExpressionDataset


@dataclass(frozen=True)
class PCAResult:
    matrix: pd.DataFrame
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    model: PCA


def calculate_qc(dataset: ExpressionDataset, detection_tpm: float = 1.0) -> pd.DataFrame:
    """Calculate per-sample sequencing and expression quality metrics."""

    qc = pd.DataFrame(index=dataset.tpm.columns)
    qc.index.name = "Sample"
    qc["Detected_Genes"] = (dataset.tpm >= detection_tpm).sum(axis=0)
    qc["Zero_Fraction"] = (dataset.tpm == 0).mean(axis=0)
    qc["TPM_Sum"] = dataset.tpm.sum(axis=0)

    if not dataset.read_counts.empty and not dataset.read_counts.isna().all().all():
        qc["Read_Depth"] = dataset.read_counts.sum(axis=0)
        mitochondrial = dataset.genes["Position"].astype(str).str.startswith(("chrM:", "MT:"))
        mitochondrial_reads = dataset.read_counts.loc[mitochondrial].sum(axis=0)
        qc["Mitochondrial_Read_Fraction"] = mitochondrial_reads / qc["Read_Depth"]

    return qc.join(dataset.metadata)


def median_absolute_deviation(frame: pd.DataFrame) -> pd.Series:
    median = frame.median(axis=0)
    return frame.subtract(median, axis=1).abs().median(axis=0)


def prepare_expression(
    tpm: pd.DataFrame,
    min_tpm: float = 1.0,
    min_sample_fraction: float = 0.20,
    top_variable_genes: int = 5_000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter low-expression genes, log transform, and select high-MAD genes.

    Samples are rows and genes are columns in the returned matrix. PCA centers
    each gene but intentionally does not unit-scale it; unit scaling can amplify
    noise from low-abundance genes.
    """

    if not 0 < min_sample_fraction <= 1 or min_tpm < 0 or top_variable_genes < 3:
        raise ValueError("Use a fraction in (0, 1], nonnegative TPM threshold, and at least three genes")
    minimum_samples = max(2, int(np.ceil(tpm.shape[1] * min_sample_fraction)))
    expressed = (tpm >= min_tpm).sum(axis=1) >= minimum_samples
    log_expression = np.log2(tpm.loc[expressed].T + 1.0)
    log_expression = log_expression.loc[:, log_expression.var() > 0]
    if log_expression.shape[1] < 3:
        raise ValueError("At least three variable genes must pass the expression filter")
    variability = median_absolute_deviation(log_expression).sort_values(ascending=False)
    selected = variability.head(min(top_variable_genes, len(variability))).index
    return log_expression.loc[:, selected], variability


def fit_pca(matrix: pd.DataFrame, n_components: int = 10) -> PCAResult:
    component_count = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
    model = PCA(n_components=component_count, svd_solver="full")
    transformed = model.fit_transform(matrix)
    component_names = [f"PC{index}" for index in range(1, component_count + 1)]
    scores = pd.DataFrame(transformed, index=matrix.index, columns=component_names)
    scores.index.name = "Sample"
    loadings = pd.DataFrame(model.components_.T, index=matrix.columns, columns=component_names)
    loadings.index.name = "Gene_ID"
    explained = pd.DataFrame(
        {
            "Component": component_names,
            "Explained_Variance_Ratio": model.explained_variance_ratio_,
            "Cumulative_Explained_Variance": np.cumsum(model.explained_variance_ratio_),
        }
    )
    return PCAResult(matrix=matrix, scores=scores, loadings=loadings, explained_variance=explained, model=model)
