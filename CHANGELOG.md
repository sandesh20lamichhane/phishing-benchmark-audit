# Changelog

## v2.0.0 — journal revision

Everything behind the revised paper. The headline numbers now come from one
environment (Python 3.11, xgboost 3.2.0, scikit-learn 1.9.1): mean
within-dataset ROC-AUC 0.9728, cross-dataset 0.6654, six of twenty transfer
cells inverted (lowest 0.1923), exact permutation p = 6.4×10⁻⁴, Cliff's
δ = 0.88. v1.0.0 reported 0.9715 and 0.6694 from an earlier Colab environment;
cells differ by at most 0.035 and the same six are inverted.

Added:
- `scripts/cell_level_stats.py`: the within- vs cross-dataset gap tested with
  the dataset-pair cell as the unit (exact permutation over 53,130
  assignments), replacing the seed-level test.
- `scripts/revision_experiments.py`: targeted and host-only ablations,
  logistic regression, character n-gram TF-IDF, overlap-controlled transfer,
  PR-AUC and TPR at 0.1% and 1% FPR, TreeSHAP attribution, `www.` removal,
  structural audit and Tranco membership.
- `scripts/provenance_checks.py`: full-file containment with label agreement;
  Mendeley PhishURL's legitimate class is Kaggle Malicious's with
  `https://www.` prepended, and Kaggle Malicious contains PhishStorm with
  95.9% of its labels inverted.
- `scripts/kaggle_relabel_check.py`: transfer with the PhishStorm block of
  Kaggle Malicious removed or relabelled.
- `scripts/char_cnn.py` and `scripts/transformer_url.py`: character CNN and
  fine-tuned DistilBERT on raw URLs.
- `scripts/build_aligned_benchmark.py`: the procedure-aligned benchmark from
  Phishing.Database and Common Crawl, with its manifest.
- `scripts/fetch_extra_datasets.py`: two further Kaggle corpora, both found to
  duplicate existing benchmarks and excluded.
- `scripts/headline_stats.py` and `scripts/make_heatmap.py`: the paper's
  headline statistics and Fig. 3 from the committed results.
- `notebooks/10_q1_revision.ipynb`: the Colab run of the GPU models and the
  aligned benchmark.
- Results in `reports/tables/revision/` and `reports/tables/revision_colab/`.
- Checksums of the files the scripts read (`ebbu2017.csv`, `phishstorm.csv`)
  and of the Tranco list.

Changed:
- `reports/figures/fig_heatmap.pdf` / `.png` redrawn from the primary
  environment's matrix.
- README, requirements and citation metadata updated for the revision.

## v1.0.0 — paper submission release (2026-08-02)

Audit code, notebooks 02-09 and result tables for the original submission.
