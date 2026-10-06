# Colab results for the journal revision

Output of `notebooks/10_q1_revision.ipynb` on Google Colab (xgboost 3.4.1,
scikit-learn 1.6.1, NVIDIA GPU), final run completed 2026-10-05. Seven corpora:
the five core benchmarks (60k cap, same sampling as notebook 09) and the two
procedure-aligned sets built by `scripts/build_aligned_benchmark.py`.

| Corpus | Rows | Notes |
|---|---|---|
| aligned_natural | 16,336 | Phishing.Database ACTIVE vs Common Crawl CC-MAIN-2026-39 captures of 1,790 Tranco domains, 5 URLs per domain, balanced by count |
| aligned_matched | 10,060 | subset of aligned_natural with identical (HTTPS, `www.`, root page, query) strata in both classes |

Files:

- `matrix_results.csv`: XGBoost, XGBoost on `www.`-stripped URLs (`xgb_nowww`), logistic regression and character n-gram TF-IDF + LR; three feature sets; seeds 42-46; raw, URL-deduplicated and domain-deduplicated test sets; ROC-AUC, PR-AUC and TPR at 0.1% and 1% FPR.
- `char_cnn_results.csv` (seeds 42-44) and `transformer_results.csv` (DistilBERT, seeds 42-46): raw URL strings.
- `summary_models_core.csv`: within vs cross-dataset AUC per model on the five core corpora, with exact permutation p, Cliff's delta and inverted-cell counts. `summary_models.csv` adds the aligned sets. `summary_aligned.csv`: aligned sets against the core corpora. The pair `aligned_natural`/`aligned_matched` is nested and is excluded from every summary.
- `summary_overlap.csv`, `overlap_cells_xgb_full.csv`: transfer after removing test URLs or domains seen in training.
- `shap_*.csv`: TreeSHAP separation per feature for each cross cell, and the features behind each inverted cell.
- `structural_audit_all.csv`, `containment_all.csv`, `tranco_membership.csv`: class composition, URL containment (on the 60k samples) and Tranco top-1M membership. Full-file containment is in `../revision/containment_fullfile.csv`.
- `extra_provenance.csv`: the two optional Kaggle corpora are duplicates of core corpora (containment 1.0 with Mendeley PhishURL and with PhishStorm) and were not used. Its aligned rows refer to the first aligned build; the final sets' containment is in `containment_all.csv`.
- `aligned_manifest.json`: sources, crawl id, list checksum, strata and counts of the aligned sets. `corpus_fingerprints.json`: content hashes of every corpus in the run.

On the five core corpora the seed-averaged cells agree with the local run in `../revision/` (xgboost 3.2.0, scikit-learn 1.9.1) to within 0.013 AUC for XGBoost with all features, with the same six inverted cells.
