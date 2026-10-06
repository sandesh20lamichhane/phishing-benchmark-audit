# Procedure-aligned benchmark (Section 7 of the paper)

The two sets used in the paper, built on 2026-10-05 by
`scripts/build_aligned_benchmark.py` (run from `notebooks/10_q1_revision.ipynb`).
Their sources, crawl id, phishing-list checksum and counts are in
`reports/tables/revision_colab/aligned_manifest.json`.

| File | URLs | Phishing | SHA-256 |
|---|---|---|---|
| `aligned_natural.csv` | 16,336 | 8,168 | `a58a96d1135e597c03313f0ee877a33543a9986542dcb3cfb6c345a365cc275d` |
| `aligned_matched.csv` | 10,060 | 5,030 | `3061e772795981432f204367233f6a39ae3bd6ab6c4fbf924a7028337789d5ea` |

Columns: `url`, `label` (1 = phishing), `source` (`phishing.database` or
`commoncrawl`). `aligned_matched` is a subset of `aligned_natural` in which
HTTPS, the `www.` prefix, root pages and query strings are equally distributed
in both classes. Both files are the exact inputs of the reported runs: their
content hashes equal those in
`reports/tables/revision_colab/corpus_fingerprints.json`.

The phishing feed changes daily, so rebuilding gives a different set. To
rerun the experiments on these sets, copy both files into `EXTRA_DIR`
(default `data/raw/extra/`).

Phishing URLs come from the Phishing.Database project's
`phishing-links-ACTIVE` list and legitimate URLs from the Common Crawl
CC-MAIN-2026-39 URL index; both remain under their providers' terms and are not
covered by this repository's license. **The phishing URLs were live when
collected. Do not open them.**
