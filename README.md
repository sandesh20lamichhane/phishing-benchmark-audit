# Structural Artefacts in Phishing URL Benchmarks

Code and result tables for:

> S. Lamichhane, U. Aijaz N, and M. Sadath P, "Structural Artefacts in Phishing
> URL Benchmarks: An Audit of Five Public Datasets and the Failure of
> Cross-Dataset Transfer." Manuscript prepared for AICCoNS 2027.

Machine learning models for phishing URL detection routinely report accuracies
above 99% on public benchmarks. We audit five independent public datasets with a
common structural diagnostic and find that this figure is not evidence of
detection capability. Three of the five exhibit **degenerate class properties** — a
structural attribute of the URL fixed by construction across an entire class —
and in two of them the property separates the classes. In
PhiUSIIL, all 134,850 legitimate URLs use HTTPS, have an empty path, and carry no
query string, while the phishing class consists of complete URLs.

Under cross-dataset transfer, mean ROC-AUC falls from 0.9715 to 0.6694
(gap 0.3021; exact permutation test over the 25 dataset-pair cells, *p* = 7.0×10⁻⁴;
Cliff's δ = 0.86). More tellingly,
**six of twenty transfer results fall below 0.5**, reaching 0.2020: these models
are not uninformative but systematically *inverted*, because the datasets encode
contradictory structural conventions. An ablation removing every scheme-, path-,
and query-derived feature does not restore transfer, and logistic regression
reproduces the pattern — the artefact extends to the hostname distributions and
cannot be repaired by feature deletion.

We also document two provenance defects in circulation:

1. A Kaggle release and the Hannousse & Yahiouche corpus are the **same data** —
   normalised URL sets identical (Jaccard = 1.000), one containing 8,002
   duplicated rows. They have been cited as independent datasets.
2. **PhishStorm** is contained *wholesale* inside a widely used Kaggle aggregate
   (containment = 1.000) and is distributed with the scheme stripped from every
   URL in both classes — which explains that aggregate's anomalous 0.5% HTTPS
   rate in its legitimate class.

## Repository layout

    notebooks/          the runs behind every number in the paper
    src/                feature extraction, statistics, checkpointing
                        (each notebook also embeds and rewrites these sources,
                        so the notebooks are self-contained)
    reports/tables/     all result CSVs and run metadata
    reports/figures/    the three figures as published
    checksums.txt       SHA-256 of every raw dataset file used

## Data (not included — fetch, then verify)

Raw datasets are third-party and are **not** redistributed here. The notebooks
download them; sources:

| Key | Source |
|---|---|
| phiusiil | UCI ML Repository, id 967 |
| kaggle_malicious | Kaggle, `sid321axn/malicious-urls-dataset` |
| mendeley_phishurl | Mendeley Data, `vfszbj9b36` |
| hannousse | Mendeley Data, `c2gw7fy2j4` (V3) |
| ebbu2017 | GitHub, `ebubekirbbr/pdd` (authors' repository for Sahingoz et al., ESWA 2019) |
| phishstorm | Public mirror of Marchal et al. (2014); excluded from the independent set, retained as a probe |

After download, verify with `sha256sum -c checksums.txt`.

## Reproducing the paper

Run in order on Google Colab (Drive-mounted; paths are set inside the notebooks):

| Notebook | Produces |
|---|---|
| `06_extended_corpora_and_statistics_v2` | acquisition, structural audit, duplication analysis, first transfer matrix |
| `07_dedup_fix_and_figures` | deduplicated dataset set, corrected 4×4 transfer matrix, statistics (Table 3) |
| `08_ablation_and_model_robustness_v2` | causal ablation and logistic-regression robustness (Table 5) |
| `09_fifth_dataset_and_provenance_v2` | Ebbu2017 acquisition, PhishStorm containment, 5×5 matrix, all figures |

`scripts/cell_level_stats.py` recomputes the aggregate within- vs cross-dataset
gap from `reports/tables/transfer_matrix_5way.csv`, treating each dataset-pair
cell (mean over five seeds) as one observation, and writes
`reports/tables/cell_level_stats.csv`. Seeds of one cell share their data, so
the cell is the correct unit for the permutation test and Cliff's δ.

Notebooks 02 and 03 are included for completeness (initial acquisition and
feature/split development). Every long-running cell checkpoints to Drive and
resumes, so a disconnect costs nothing.

## Reproducibility note

The protocol was executed independently in two environments differing in
hardware and scikit-learn version (1.6.1 and 1.8.0). Individual XGBoost transfer
cells shift by up to ~0.02–0.03 across sessions on this near-degenerate data,
while **the number of inverted cells, their identity, and every aggregate gap
were identical in all runs**. Treat cell identities and aggregates as the
findings and third-decimal values as environment-dependent. Notebook 08 contains
a reproduction gate and a cell-by-cell comparison against the published values;
run metadata (library versions, seeds, feature lists) is written to
`reports/tables/`.

Reported values come from runs on the authors' own Colab environment
(xgboost 3.3.0, scikit-learn 1.6.1, Python 3.12).

## License

Code: MIT (see `LICENSE`). The datasets remain under their original providers'
terms and are not covered by this license.
