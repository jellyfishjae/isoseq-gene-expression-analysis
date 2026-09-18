from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from isoseq_analysis.io import load_dataset
from isoseq_analysis.preprocess import fit_pca, prepare_expression
from isoseq_analysis.stats import benjamini_hochberg, centroid_permutation_test
from isoseq_analysis.stats import exploratory_group_signals, preprocessing_sensitivity, confounding_diagnostic
from isoseq_analysis.pipeline import AnalysisConfig, _result_markdown
from isoseq_analysis.plots import SYNTHETIC_OUTPUT, _save
import matplotlib.pyplot as plt


def _write_inputs(tmp_path: Path):
    samples = [f"RAW-{index}" for index in range(8)]
    rng = np.random.default_rng(7)
    values = rng.gamma(2, 3, size=(80, 8))
    values[:20, 4:] += 60
    values = values / values.sum(axis=0) * 1_000_000
    expression = pd.DataFrame(
        {
            "Gene_ID": [f"G{index}" for index in range(80)],
            "Gene_Symbol": [f"SYM{index}" for index in range(80)],
            "Position": [f"chr{index % 3 + 1}:{index}-{index + 1}" for index in range(80)],
        }
    )
    for index, sample in enumerate(samples):
        expression[f"{sample}:TPM"] = values[:, index]
    for index, sample in enumerate(samples):
        expression[f"{sample}:Readcount"] = np.rint(values[:, index] * 2).astype(int)
    clinical = pd.DataFrame(
        {
            "B-No": samples,
            "Group": ["A"] * 4 + ["B"] * 4,
            "Sex": ["Female", "Male"] * 4,
            "Age": np.arange(70, 78),
            "PMI": np.arange(5, 13),
            "RIN": np.linspace(6.2, 7.6, 8),
            "Iso-seq": [f"run\nTBD000001_{index % 2}" for index in range(8)],
        }
    )
    expression_path = tmp_path / "expression.tsv"
    clinical_path = tmp_path / "clinical.csv"
    expression.to_csv(expression_path, sep="\t", index=False)
    clinical.to_csv(clinical_path, index=False)
    return expression_path, clinical_path


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)
        self.expression_path, self.clinical_path = _write_inputs(self.tmp_path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loader_anonymizes_source_ids(self):
        dataset = load_dataset(self.expression_path, self.clinical_path)
        self.assertEqual(list(dataset.tpm.columns), [f"S{index:02d}" for index in range(1, 9)])
        self.assertFalse(any("RAW" in sample for sample in dataset.metadata.index))
        self.assertEqual(set(dataset.metadata["Candidate_Batch"]), {"Candidate 1", "Candidate 2"})

    def test_preprocessing_and_pca_dimensions(self):
        dataset = load_dataset(self.expression_path, self.clinical_path)
        matrix, _ = prepare_expression(dataset.tpm, min_sample_fraction=0.2, top_variable_genes=30)
        result = fit_pca(matrix, n_components=5)
        self.assertEqual(matrix.shape, (8, 30))
        self.assertEqual(result.scores.shape, (8, 5))
        self.assertTrue(
            np.isclose(
                result.explained_variance["Explained_Variance_Ratio"].sum(),
                result.model.explained_variance_ratio_.sum(),
            )
        )

    def test_bh_adjustment_is_bounded_and_monotone(self):
        p_values = np.array([0.04, 0.001, 0.02, 0.8])
        adjusted = benjamini_hochberg(p_values)
        self.assertTrue(np.all((0 <= adjusted) & (adjusted <= 1)))
        order = np.argsort(p_values)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= 0))

    def test_permutation_test_detects_large_group_shift(self):
        dataset = load_dataset(self.expression_path, self.clinical_path)
        matrix, _ = prepare_expression(dataset.tpm, min_sample_fraction=0.2, top_variable_genes=50)
        result = centroid_permutation_test(
            matrix, dataset.metadata["Group"], permutations=199, random_state=1
        )
        self.assertGreater(result["r_squared"], 0.35)
        self.assertLessEqual(result["p_value"], 0.05)


    def test_invalid_inputs(self):
        for bad in [np.inf, -1, np.nan, "not a number"]:
            with self.subTest(bad=bad):
                expression, clinical = _write_inputs(self.tmp_path)
                frame = pd.read_csv(expression, sep="\t")
                frame["RAW-0:TPM"] = frame["RAW-0:TPM"].astype(object)
                frame.loc[0, "RAW-0:TPM"] = bad
                frame.to_csv(expression, sep="\t", index=False)
                with self.assertRaises(ValueError):
                    load_dataset(expression, clinical)

    def test_unmatched_sample_and_missing_group(self):
        frame = pd.read_csv(self.clinical_path)
        frame.iloc[1:].to_csv(self.clinical_path, index=False)
        with self.assertRaisesRegex(ValueError, "absent"):
            load_dataset(self.expression_path, self.clinical_path)
        frame.drop(columns="Group").to_csv(self.clinical_path, index=False)
        with self.assertRaisesRegex(ValueError, "Group"):
            load_dataset(self.expression_path, self.clinical_path)

    def test_gene_report_counts_more_than_fifty_hits(self):
        # Sixty completely separated genes expose truncation before counting.
        index = [f"S{i}" for i in range(20)]
        genes = [f"G{i}" for i in range(60)]
        values = np.arange(20, dtype=float)
        matrix = pd.DataFrame(np.tile(values[:, None], (1, 60)), index=index, columns=genes)
        groups = pd.Series(["A"] * 10 + ["B"] * 10, index=index)
        annotations = pd.DataFrame({"Gene_Symbol": genes}, index=genes)
        signals = exploratory_group_signals(matrix, groups, annotations)
        self.assertEqual(len(signals), 60)
        self.assertEqual(int((signals.Q_Value < 0.05).sum()), 60)
        report = _result_markdown(
            AnalysisConfig(self.expression_path, self.clinical_path, self.tmp_path, synthetic=True),
            20, 60, 60, groups.value_counts(),
            pd.DataFrame({"Component": ["PC1"], "Explained_Variance_Ratio": [1.0]}),
            pd.DataFrame(columns=["Component", "Variable"]),
            {"pseudo_f": 3, "r_squared": 0.5, "p_value": 0.1, "permutations": 19},
            {}, signals,
        )
        self.assertIn("Of 60 genes tested, 60", report)
        self.assertIn("SYNTHETIC DATA", report)
        self.assertNotIn("data/raw/", report)

    def test_sensitivity_uses_actual_cutoffs(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(rng.normal(size=(8, 12)))
        result = preprocessing_sensitivity(frame, reference_genes=6, gene_counts=(3, 50, 100))
        self.assertEqual(result.Top_MAD_Genes.tolist(), [3, 6, 12])
        self.assertTrue((result.Reference_Genes == 6).all())
        self.assertAlmostEqual(result.loc[result.Top_MAD_Genes.eq(6), "Distance_Spearman"].iloc[0], 1)

    def test_missing_candidate_batch_is_skipped(self):
        metadata = pd.DataFrame({"Group": ["A", "B"], "Candidate_Batch": [np.nan, np.nan]})
        _, diagnostic = confounding_diagnostic(metadata)
        self.assertEqual(diagnostic, {})

    def test_synthetic_label_is_added_to_figure(self):
        token = SYNTHETIC_OUTPUT.set(True)
        figure, _ = plt.subplots()
        try:
            with patch.object(figure, "savefig"):
                _save(figure, self.tmp_path / "example.png")
            self.assertTrue(any("SYNTHETIC DATA" in text.get_text() for text in figure.texts))
        finally:
            SYNTHETIC_OUTPUT.reset(token)


if __name__ == "__main__":
    unittest.main()
