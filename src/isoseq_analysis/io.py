"""Input validation and privacy-preserving sample metadata handling."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpressionDataset:
    """Validated expression matrices and de-identified sample metadata."""

    tpm: pd.DataFrame
    read_counts: pd.DataFrame
    genes: pd.DataFrame
    metadata: pd.DataFrame


def _sample_columns(frame: pd.DataFrame, suffix: str) -> list[str]:
    return [column for column in frame.columns if column.endswith(suffix)]


def _extract_batch(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    matches = re.findall(r"TBD\d+_\d+", str(value))
    return matches[-1] if matches else np.nan


def _private_to_public_ids(sample_names: list[str]) -> dict[str, str]:
    return {name: f"S{index:02d}" for index, name in enumerate(sample_names, start=1)}


def load_dataset(expression_path: str | Path, clinical_path: str | Path) -> ExpressionDataset:
    """Load the two source tables, validate their schema, and anonymize samples.

    The source identifiers are used only for the in-memory join. They are replaced
    immediately afterward and are never written by the analysis pipeline.
    """

    expression = pd.read_csv(expression_path, sep="\t")
    clinical = pd.read_csv(clinical_path)

    required_expression = {"Gene_ID", "Gene_Symbol", "Position"}
    missing_expression = required_expression.difference(expression.columns)
    if missing_expression:
        raise ValueError(f"Expression table is missing columns: {sorted(missing_expression)}")
    if "B-No" not in clinical.columns:
        raise ValueError("Clinical table must contain a 'B-No' sample identifier column")
    for frame, column in [(expression, "Gene_ID"), (clinical, "B-No")]:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must contain nonempty identifiers")
    if "Group" not in clinical:
        raise ValueError("Clinical table must contain Group")
    if expression["Gene_ID"].duplicated().any():
        raise ValueError("Gene_ID must be unique")
    if clinical["B-No"].duplicated().any():
        raise ValueError("Clinical sample identifiers must be unique")

    tpm_columns = _sample_columns(expression, ":TPM")
    read_columns = _sample_columns(expression, ":Readcount")
    if not tpm_columns:
        raise ValueError("No columns ending in ':TPM' were found")

    tpm_samples = [column.removesuffix(":TPM") for column in tpm_columns]
    read_samples = [column.removesuffix(":Readcount") for column in read_columns]
    if read_samples and read_samples != tpm_samples:
        raise ValueError("TPM and read-count samples must appear in the same order")
    clinical_samples = set(clinical["B-No"].astype(str))
    unmatched = set(tpm_samples).difference(clinical_samples)
    if unmatched:
        raise ValueError(f"Expression samples absent from clinical table: {sorted(unmatched)}")

    numeric_columns = tpm_columns + read_columns
    try:
        expression[numeric_columns] = expression[numeric_columns].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Expression values must be numeric") from error
    if not np.isfinite(expression[numeric_columns].to_numpy()).all():
        raise ValueError("Expression values must be finite and nonmissing")
    if expression[numeric_columns].isna().any().any():
        raise ValueError("Expression matrices contain missing values")
    if (expression[numeric_columns] < 0).any().any():
        raise ValueError("Expression matrices contain negative values")

    id_map = _private_to_public_ids(tpm_samples)
    gene_ids = expression["Gene_ID"].astype(str)
    genes = expression[["Gene_ID", "Gene_Symbol", "Position"]].copy().set_index("Gene_ID")

    tpm = expression[tpm_columns].copy()
    tpm.columns = [id_map[sample] for sample in tpm_samples]
    tpm.index = gene_ids
    tpm.index.name = "Gene_ID"

    if read_columns:
        read_counts = expression[read_columns].copy()
        read_counts.columns = [id_map[sample] for sample in read_samples]
        read_counts.index = gene_ids
        read_counts.index.name = "Gene_ID"
    else:
        read_counts = pd.DataFrame(index=tpm.index, columns=tpm.columns, dtype=float)

    metadata_columns = [
        column
        for column in ["Group", "Sex", "Age", "PMI", "RIN", "APOE"]
        if column in clinical.columns
    ]
    clinical_indexed = clinical.set_index(clinical["B-No"].astype(str))
    metadata = clinical_indexed.loc[tpm_samples, metadata_columns].copy()
    metadata.index = [id_map[sample] for sample in tpm_samples]
    metadata.index.name = "Sample"
    if len(metadata) < 4:
        raise ValueError("At least four samples are required")
    if metadata["Group"].isna().any() or metadata["Group"].astype(str).str.strip().eq("").any():
        raise ValueError("Group must be nonmissing for every sample")
    counts = metadata["Group"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("At least two groups with two samples each are required")
    for column in ["Age", "PMI", "RIN"]:
        if column in metadata:
            metadata[column] = pd.to_numeric(metadata[column], errors="raise")
            if np.isinf(metadata[column].to_numpy(dtype=float)).any():
                raise ValueError(f"{column} must be finite or missing")

    if "Iso-seq" in clinical.columns:
        raw_batch = clinical_indexed.loc[tpm_samples, "Iso-seq"].map(_extract_batch)
        unique_batches = sorted(raw_batch.dropna().unique())
        batch_map = {batch: f"Candidate {index}" for index, batch in enumerate(unique_batches, start=1)}
        metadata["Candidate_Batch"] = raw_batch.map(batch_map).to_numpy()

    return ExpressionDataset(tpm=tpm, read_counts=read_counts, genes=genes, metadata=metadata)
