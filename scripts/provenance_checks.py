"""Full-file provenance checks behind the paper's corpus-independence claims.

The transfer experiments use 60k-row samples, and containment measured on those
samples understates how much two corpora share. This script works on the full
files and writes to OUT_DIR (default reports/tables/revision):

  containment_fullfile.csv  pairwise containment of normalised URLs (shared
                            URLs / size of the smaller corpus), with the share
                            of shared URLs on which the two labels agree
  provenance_checks.json    the two findings below, with counts

Findings checked:
  * Mendeley PhishURL's legitimate class is Kaggle Malicious's benign class with
    `https://www.` prepended (counted on shared legitimate URLs).
  * Kaggle Malicious contains PhishStorm with the labels inverted (crosstab of
    the two labels on shared URLs, plus a phishing-keyword rate under each
    labelling as an independent check of which labelling is right).

With --write-variants DIR it also writes two corrected copies of Kaggle
Malicious for scripts/kaggle_relabel_check.py:
  kaggle_nophishstorm.csv  rows whose URL is in PhishStorm removed
  kaggle_relabelled.csv    those rows kept with PhishStorm's label

Usage:
    RAW_DIR=/path/to/data/raw python scripts/provenance_checks.py [--write-variants DIR]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402

KEYWORDS = r"paypal|login|signin|verify|webscr|account|secure|bank|update|confirm"


def load_full(key):
    """One corpus, uncapped, with the loader settings of revision_experiments."""
    _, rel, ucol, lcol, phish = next(row for row in rx.LOAD_ORDER if row[0] == key)
    d = pd.read_csv(rx.RAW / rel, low_memory=False)
    uc = [c for c in d.columns if c.lower() == ucol.lower()][0]
    lc = [c for c in d.columns if c.lower() == lcol.lower()][0]
    lab = ((d[lc].astype(str).str.lower() == str(phish).lower()).astype(int)
           if isinstance(phish, str) else (d[lc] == phish).astype(int))
    df = pd.DataFrame({"url": d[uc].astype(str), "label": lab})
    df["norm"] = df.url.map(rx.normalise)
    return df


def containment(corpora):
    rows = []
    for a, b in itertools.combinations(corpora, 2):
        la = corpora[a].drop_duplicates("norm").set_index("norm").label
        lb = corpora[b].drop_duplicates("norm").set_index("norm").label
        shared = la.index.intersection(lb.index)
        rows.append({"a": a, "b": b, "n_a": len(la), "n_b": len(lb), "shared": len(shared),
                     "containment": round(len(shared) / max(min(len(la), len(lb)), 1), 4),
                     "label_agreement": round(float((la[shared] == lb[shared]).mean()), 4)
                     if len(shared) else None})
    return pd.DataFrame(rows)


def has_www(urls):
    return urls.str.lower().str.contains(r"^(?:https?://)?www\.", regex=True)


def mendeley_prefix(k, m):
    j = m.merge(k, on="norm", suffixes=("_md", "_kg")).drop_duplicates("norm")
    legit = j[(j.label_md == 0) & (j.label_kg == 0)]
    added = ~has_www(legit.url_kg) & legit.url_md.str.lower().str.startswith("https://www.")
    return {"shared_urls": len(j),
            "shared_share_of_mendeley": round(len(j) / m.norm.nunique(), 4),
            "label_agreement": round(float((j.label_md == j.label_kg).mean()), 4),
            "shared_legitimate_share": round(float((j.label_md == 0).mean()), 4),
            "shared_legitimate_both": len(legit),
            "kaggle_no_www_mendeley_https_www": int(added.sum()),
            "examples": legit[added].url_md.head(5).tolist()}


def phishstorm_flip(k, p):
    q = p.drop_duplicates("norm").merge(k, on="norm", suffixes=("_ps", "_kg"))
    tab = pd.crosstab(q.label_ps, q.label_kg)
    kw = q.url_ps.str.contains(KEYWORDS, case=False, regex=True)
    return {"shared_rows": len(q),
            "opposite_labels": int((q.label_ps != q.label_kg).sum()),
            "crosstab_phishstorm_rows_kaggle_cols": {int(r): {int(c): int(tab.loc[r, c]) for c in tab.columns}
                                                     for r in tab.index},
            "kaggle_phishing_rows": int((k.label == 1).sum()),
            "kaggle_benign_rows": int((k.label == 0).sum()),
            "keyword_rate_by_phishstorm_label": {int(g): round(float(kw[q.label_ps == g].mean()), 4) for g in (0, 1)},
            "keyword_rate_by_kaggle_label": {int(g): round(float(kw[q.label_kg == g].mean()), 4) for g in (0, 1)}}


def write_variants(k, p, out):
    out.mkdir(parents=True, exist_ok=True)
    ps = p.drop_duplicates("norm").set_index("norm").label
    inps = k.norm.isin(ps.index)
    k[~inps][["url", "label"]].to_csv(out / "kaggle_nophishstorm.csv", index=False)
    fixed = k.copy()
    fixed.loc[inps, "label"] = fixed.loc[inps, "norm"].map(ps).astype(int)
    fixed[["url", "label"]].to_csv(out / "kaggle_relabelled.csv", index=False)
    print(f"[variants] {out}: nophishstorm {int((~inps).sum()):,} rows, relabelled {len(fixed):,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-variants", type=Path, default=None)
    a = ap.parse_args()
    corpora = {key: load_full(key) for key in rx.CORE + ["phishstorm"]}
    for key, d in corpora.items():
        print(f"[load] {key}: {len(d):,} rows, {d.norm.nunique():,} unique URLs", flush=True)
    rx.OUT.mkdir(parents=True, exist_ok=True)
    cont = containment(corpora)
    cont.to_csv(rx.OUT / "containment_fullfile.csv", index=False)
    print(cont.to_string(index=False))
    k, m, p = corpora["kaggle_malicious"], corpora["mendeley_phishurl"], corpora["phishstorm"]
    checks = {"mendeley_is_kaggle_with_https_www": mendeley_prefix(k, m),
              "kaggle_contains_phishstorm_flipped": phishstorm_flip(k, p)}
    (rx.OUT / "provenance_checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    if a.write_variants:
        write_variants(k, p, a.write_variants)


if __name__ == "__main__":
    main()
