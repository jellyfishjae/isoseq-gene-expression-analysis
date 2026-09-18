# Results

SYNTHETIC DATA - demonstration only.

## Data

16 samples (ADNC: 4, ADNC+LBD: 4, LBD: 4, Control (PART): 4); 600 input genes.
PCA uses 300 high-MAD genes after filtering and log2(TPM + 1).
Samples are relabeled; this does not establish anonymity.

## PCA and metadata

PC1 explains 18.8% of the selected-gene variance.

| PC1 comparison | Result |
| --- | --- |
| RIN | rho = 0.92, p = 2.99e-07, BH q = 1.49e-05 |
| Mitochondrial read fraction | rho = -0.85, p = 2.68e-05, BH q = 0.00067 |
| Pathology group | eta-squared = 0.95, p = 0.00321, BH q = 0.0535 |

These are associations, not evidence of what caused the component.

## Group comparisons

The unadjusted permutation test gives pseudo-F = 2.32,
R-squared = 36.71%, and p = 0.005
(199 permutations).
Confounding and differences in within-group dispersion can affect this test.
A nonsignificant result does not rule out disease effects.

Of 300 genes tested, 0 have BH q < 0.05.
Correction and counts use all tested genes; the CSV contains the top 50.
These are exploratory rankings, without covariate adjustment.

## Candidate batches

Candidate-batch Cramer's V = 0.00; approximate chi-squared p = 1. Expected cell counts below five make this p-value unreliable.

Labels are parsed from report identifiers in the Iso-seq field. Confirm that these
represent sequencing batches before interpreting the table as batch confounding.
No batch correction is applied.

## Run settings

Input filenames and parameters are recorded in ../results/run_summary.json.
See the repository README for matching demo and research commands.
