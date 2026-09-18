# Methods

## Input and QC

The expression table contains gene-level TPM and optional read counts. Samples are matched to clinical metadata by identifier, then relabeled S01, S02, etc. These are pseudonyms, not a guarantee of anonymity.

QC includes read totals, genes with TPM >= 1, zero-expression fraction, and the fraction of reads mapped to chrM or MT. RIN and postmortem interval come from the clinical table. No sample is automatically excluded.

## PCA

Genes must reach TPM >= 1 in at least 20% of samples, rounded up, with a minimum of two samples. Values are transformed with log2(TPM + 1). The default analysis selects the 5,000 genes with the largest median absolute deviation (MAD).

PCA centers genes without scaling them to unit variance. This preserves differences in log-expression variance; it is one exploratory choice, not a universally optimal normalization. The pipeline reports explained variance and the strongest positive and negative loadings on PC1-PC3. PCA signs are arbitrary.

Sensitivity analysis compares pairwise distances in the first five PCs across feature cutoffs. Requested cutoffs are capped at the number of available genes and reported as actual counts. High distance correlation indicates similar sample geometry, not proof of biological validity.

## Metadata and group comparisons

For the first five PCs, continuous metadata are tested with Spearman correlation and categorical metadata with Kruskal-Wallis tests. Benjamini-Hochberg correction covers the full family of PC-metadata tests. Categorical effect size is eta-squared on scores, not the Kruskal-Wallis statistic. These small-sample p-values are approximate.

Overall group structure is tested using a Euclidean PERMANOVA-style statistic on selected log-expression values, with 999 label permutations by default. R-squared measures the between-group share of total squared variation. Unrestricted permutation assumes samples are exchangeable under the null; batch imbalance, covariates, and unequal within-group dispersion can complicate interpretation. This is not a covariate-adjusted disease test.

Genes are ranked with Kruskal-Wallis tests on the same selected matrix. BH correction covers all tested genes before exporting the top 50. The summary counts significant genes across the full tested set. Rankings are exploratory because feature selection and testing use the same data, covariates are not adjusted, and groups may be confounded. Mean differences are on the log2(TPM + 1) scale, not conventional log fold changes.

## Candidate batches

The last identifier matching `TBD<number>_<number>` in each `Iso-seq` field is used as a candidate batch label. It might identify a report or delivery rather than an actual sequencing run. Confirm its meaning with the data owner.

The group-by-candidate-batch table and Cramer's V describe imbalance. The chi-squared p-value is asymptotic and may be unreliable when expected cell counts are small; the output flags that condition. This diagnostic does not establish a technical batch effect. No automatic batch correction is performed.

## Limits

The pipeline is for exploration, not diagnosis or confirmed biomarker discovery. RNA-quality associations can have technical and biological explanations. A nonsignificant group comparison does not demonstrate equivalence or absence of disease effects. Confirmatory work needs an appropriate count-based model, a defensible covariate design, and sufficient replication.

The table is gene-level, so transcript usage, isoform switching, and splicing are outside its scope. The included synthetic dataset has deliberately planted patterns; its results demonstrate software behavior only.
