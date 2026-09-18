#!/usr/bin/env python3
"""Generate a small, explicitly synthetic dataset for the public demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(2026)
    output = Path(__file__).resolve().parents[1] / "data" / "demo"
    output.mkdir(parents=True, exist_ok=True)

    groups = ["ADNC", "ADNC+LBD", "LBD", "Control (PART)"]
    samples = [f"DEMO-{index:02d}" for index in range(1, 17)]
    group_labels = np.repeat(groups, 4)
    genes = 600

    baseline = rng.normal(4.2, 1.1, size=(genes, 1))
    log_expression = baseline + rng.normal(0, 0.45, size=(genes, len(samples)))
    for group_index in range(4):
        columns = np.where(group_labels == groups[group_index])[0]
        start = group_index * 15
        log_expression[start : start + 15, columns] += 1.1
    quality = np.linspace(-0.8, 0.8, len(samples))
    log_expression[60:100] += quality
    abundance = np.maximum(np.exp2(log_expression) - 1, 0.0)
    tpm = abundance / abundance.sum(axis=0) * 1_000_000
    totals = rng.integers(1_500_000, 2_500_000, size=len(samples))
    read_counts = np.column_stack(
        [rng.multinomial(int(total), tpm[:, index] / 1_000_000) for index, total in enumerate(totals)]
    )

    expression = pd.DataFrame(
        {
            "Gene_ID": [f"ENSGSYN{index:06d}.1" for index in range(1, genes + 1)],
            "Gene_Symbol": [f"SYN{index:04d}" for index in range(1, genes + 1)],
            "Position": [
                f"{'chrM' if index < 20 else 'chr' + str(index % 22 + 1)}:{1000 * index}-{1000 * index + 500}"
                for index in range(genes)
            ],
        }
    )
    for index, sample in enumerate(samples):
        expression[f"{sample}:TPM"] = tpm[:, index]
    for index, sample in enumerate(samples):
        expression[f"{sample}:Readcount"] = read_counts[:, index]

    clinical = pd.DataFrame(
        {
            "B-No": samples,
            "Group": group_labels,
            "Sex": ["Female", "Male"] * 8,
            "Age": rng.integers(65, 91, size=len(samples)),
            "PMI": rng.uniform(4, 20, size=len(samples)).round(1),
            "RIN": (7.0 + quality + rng.normal(0, 0.15, len(samples))).round(1),
            "APOE": ["E3/3", "E3/4", "E3/3", "E2/3"] * 4,
            "Iso-seq": [f"SYNTHETIC\nTBD000001_{10000 + (index % 4)}" for index in range(len(samples))],
        }
    )

    expression.to_csv(output / "Gene_expression.tsv", sep="\t", index=False)
    clinical.to_csv(output / "clinical_info.csv", index=False)
    print(f"Wrote synthetic demo data to {output}")


if __name__ == "__main__":
    main()
