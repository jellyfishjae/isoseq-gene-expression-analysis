#!/usr/bin/env python3
"""Command-line entry point for the complete analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from isoseq_analysis import AnalysisConfig, run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path, required=True, help="Gene_expression.tsv path")
    parser.add_argument("--clinical", type=Path, required=True, help="clinical_info.csv path")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--top-genes", type=int, default=5_000)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--synthetic", action="store_true", help="Label outputs as synthetic demo data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        expression_path=args.expression,
        clinical_path=args.clinical,
        output_root=args.output,
        top_variable_genes=args.top_genes,
        permutations=args.permutations,
        random_state=args.seed,
        synthetic=args.synthetic,
    )
    summary = run_analysis(config)
    print(
        "Analysis complete: "
        f"{summary['dataset']['samples']} samples, "
        f"{summary['dataset']['pca_genes']} genes used for PCA."
    )


if __name__ == "__main__":
    main()
