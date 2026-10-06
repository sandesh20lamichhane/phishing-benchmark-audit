# Structural Artefacts in Phishing URL Benchmarks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23184584.svg)](https://doi.org/10.5281/zenodo.23184584)

Code and result tables for:

> S. Lamichhane, U. Aijaz N, and M. Sadath P, "Structural Artefacts in Phishing
> URL Benchmarks: An Audit of Five Public Datasets and the Failure of
> Cross-Dataset Transfer."

Machine learning models for phishing URL detection routinely report accuracies
above 99% on public benchmarks. Auditing five widely used public URL benchmarks,
we show that they encode collection conventions that inflate within-dataset
results and can cause systematic cross-dataset prediction inversion, and that a
simple pre-model audit exposes this.

Three of the five datasets exhibit **degenerate class properties** — a
structural attribute of the URL fixed across an entire class. In PhiUSIIL, all
134,850 legitimate URLs are `https://www.` domain roots with no path and no
query string, while the phishing class consists of complete URLs.

With thirty-two syntactic features, mean within-dataset ROC-AUC is 0.9728;
under cross-dataset transfer it falls to 0.6654 (exact permutation test over
the 25 dataset-pair cells, *p* = 6.4×10⁻⁴; Cliff's δ = 0.88). **Six of twenty
transfer results fall below 0.5**, reaching 0.1923: the models are not
uninformative but systematically *inverted*. Character-level CNN and fine-tuned
DistilBERT models on the raw URL string show the same gap.

Full-file provenance checks explain part of the picture:

1. Mendeley PhishURL's legitimate class is largely Kaggle Malicious's benign
   class with `https://www.` prepended (339,802 shared URLs), so its
   conventions were manufactured by string concatenation.
2. Kaggle Malicious contains the **PhishStorm** benchmark wholesale
   (containment 1.000) with **95.9% of its labels inverted**. With those labels
   corrected, four of the six inverted cells remain, each pairing datasets whose
   scheme and `www.` conventions run in opposite directions.
3. A Kaggle release and the Hannousse & Yahiouche corpus are the **same data**
   (normalised URL sets identical, one containing 8,002 duplicated rows), and
   two further Kaggle aggregates repeat existing benchmarks.

Models trained on a procedure-aligned benchmark, built from Phishing.Database
and Common Crawl with four URL-level conventions matched across classes,
transfer to all five public benchmarks without inversion. The benchmark is not
an unbiased reference: its legitimate domains come from the Tranco popularity
ranking.

## Repository layout

    notebooks/                 Colab runs of the original study (06-09) and of
                               the journal revision (10)
    scripts/                   the revision experiments, provenance checks,
                               aligned-benchmark builder, headline statistics
                               and the transfer heat map
    src/                       feature extraction, statistics, checkpointing
                               (notebooks 06-09 also embed these sources, so
                               they are self-contained)
    reports/tables/            result CSVs and run metadata of the original study
    reports/tables/revision/   revision results from the primary environment
    reports/tables/revision_colab/
                               revision results from Colab (GPU models and the
                               aligned benchmark); see its README.md
    reports/figures/           the three figures of the paper
    datasets/aligned/          the two procedure-aligned sets used in the paper
    checksums.txt              SHA-256 of every raw dataset file used

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
| Tranco top-1M | `https://tranco-list.eu/top-1m.csv.zip`, downloaded 2026-07-24 (used for Tranco membership and the aligned benchmark) |

After download, verify with `sha256sum -c checksums.txt` from `data/raw`
(the Tranco list is expected at `data/external/tranco/`). Notebook 09 converts the Ebbu2017 and PhishStorm sources to
`ebbu2017/ebbu2017.csv` and `phishstorm/phishstorm.csv`; the scripts read those
files, and their checksums are listed too.

## Reproducing the paper

### Original study

Run in order on Google Colab (Drive-mounted; paths are set inside the notebooks):

| Notebook | Produces |
|---|---|
| `06_extended_corpora_and_statistics_v2` | acquisition, structural audit, duplication analysis, first transfer matrix |
| `07_dedup_fix_and_figures` | deduplicated dataset set, corrected 4×4 transfer matrix, statistics |
| `08_ablation_and_model_robustness_v2` | ablation and logistic-regression robustness |
| `09_fifth_dataset_and_provenance_v2` | Ebbu2017 acquisition, PhishStorm containment, 5×5 matrix, Figs. 1 and 2 |

Notebooks 02 and 03 are included for completeness (initial acquisition and
feature/split development). Every long-running cell checkpoints to Drive and
resumes, so a disconnect costs nothing.

### Journal revision

The scripts read the raw files from `RAW_DIR` (default `data/raw`, laid out as
in `checksums.txt`) and the Tranco list from `TRANCO`, and write to
`reports/tables/revision/`. Every step appends and resumes.

| Command | Produces | Paper |
|---|---|---|
| `python scripts/revision_experiments.py` | transfer matrices for XGBoost, logistic regression and character n-grams; overlap-controlled transfer; operating points; TreeSHAP; `www.` removal; structural audit; Tranco membership | Tables 3, 5, 7; Sections 6.5, 6.6, 6.8 |
| `python scripts/provenance_checks.py --write-variants data/variants` | full-file containment with label agreement, the Mendeley prefix and PhishStorm checks, two corrected Kaggle Malicious files | Table 2; Section 4 |
| `EXTRA_DIR=data/variants python scripts/kaggle_relabel_check.py` | transfer with Kaggle Malicious corrected | Table 6 |
| `python scripts/headline_stats.py --baselines` | transfer matrix, permutation test, asymmetry, DeLong comparisons, PhiUSIIL → PhishStorm probe | Table 4; Sections 2.3, 6.1-6.4 |
| `python scripts/make_heatmap.py` | `reports/figures/fig_heatmap.pdf` from `headline_transfer_matrix.csv` | Fig. 3 |
| `notebooks/10_q1_revision.ipynb` (Colab, GPU) | character CNN (`scripts/char_cnn.py`), DistilBERT (`scripts/transformer_url.py`), the procedure-aligned benchmark (`scripts/build_aligned_benchmark.py`) | Tables 5 and 8; Section 7 |

`scripts/cell_level_stats.py` recomputes the original study's aggregate gap
from `reports/tables/transfer_matrix_5way.csv` with the dataset-pair cell as
the unit of analysis.

## Reproducibility note

Every headline number in the paper comes from one environment: Python 3.11,
xgboost 3.2.0, scikit-learn 1.9.1 (`reports/tables/revision/`). The original
submission ran the same full-feature matrix on Colab (xgboost 3.3.0,
scikit-learn 1.6.1) and reported within 0.9715 and cross 0.6694; individual
cells differ by at most 0.035 and the same six cells are inverted. The GPU
models and the aligned benchmark ran on Colab (xgboost 3.4.1, scikit-learn
1.6.1), where the five-dataset matrix reproduces the primary one to within
0.013 per cell, again with the same six inverted cells. Treat cell identities
and aggregates as the findings and third-decimal values as
environment-dependent.

The procedure-aligned benchmark is built from a live phishing feed, so a
rebuild on a later date gives a different set. The two sets used in the paper
are in `datasets/aligned/`; copy them into `data/raw/extra/` to rerun Table 8
on exactly those URLs.

## Archive

Each release is archived on Zenodo. Release v2.0.0 (the journal revision) is
[10.5281/zenodo.23184584](https://doi.org/10.5281/zenodo.23184584); release
v1.0.0 (the original submission) is
[10.5281/zenodo.21756633](https://doi.org/10.5281/zenodo.21756633).

## License

Code: MIT (see `LICENSE`). The datasets remain under their original providers'
terms and are not covered by this license.
