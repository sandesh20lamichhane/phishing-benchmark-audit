"""Fetch additional raw-URL phishing corpora into EXTRA_DIR as url,label CSVs.

Each source is downloaded with kagglehub (Kaggle credentials required, as in
notebook 06). Label values are mapped explicitly; rows with any other value
are dropped and counted, never guessed. A source that fails to download is
reported and skipped.

You can also add a corpus by hand: put a CSV with `url` and `label`
(1 = phishing, 0 = legitimate) columns into EXTRA_DIR. Every CSV there joins
the transfer matrix, so run the `audit` step of revision_experiments.py
afterwards and drop any corpus whose URL containment with an existing one
exceeds 0.5 (it is not independent).

Usage:
    python scripts/fetch_extra_datasets.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402

PHISH = {"bad", "phishing", "phish", "malicious", "1", "true"}
LEGIT = {"good", "benign", "legitimate", "legit", "0", "false"}

# key -> Kaggle dataset slug. Column names are detected, label values are not guessed.
SOURCES = {
    "kaggle_phishing_site_urls": "taruntiwarihp/phishing-site-urls",
    "kaggle_malicious_benign_urls": "siddharthkumar25/malicious-and-benign-urls",
}


def pick(cols, names):
    for c in cols:
        if c.strip().lower() in names:
            return c
    return None


def convert(key, csv):
    d = pd.read_csv(csv, low_memory=False)
    ucol = pick(d.columns, {"url", "urls", "domain"})
    lcol = pick(d.columns, {"label", "labels", "type", "class", "status", "result"})
    if ucol is None or lcol is None:
        print(f"[{key}] {csv.name}: no url/label columns in {list(d.columns)}")
        return None
    v = d[lcol].astype(str).str.strip().str.lower()
    keep = v.isin(PHISH | LEGIT)
    print(f"[{key}] {csv.name}: url column '{ucol}', label column '{lcol}', "
          f"values {v.value_counts().head(8).to_dict()}; dropped {int((~keep).sum()):,} rows")
    return pd.DataFrame({"url": d.loc[keep, ucol].astype(str),
                         "label": v[keep].isin(PHISH).astype(int)})


def main():
    import kagglehub
    rx.EXTRA.mkdir(parents=True, exist_ok=True)
    for key, slug in SOURCES.items():
        out = rx.EXTRA / f"{key}.csv"
        if out.exists():
            print(f"[{key}] already present")
            continue
        try:
            path = Path(kagglehub.dataset_download(slug))
        except Exception as e:
            print(f"[{key}] download failed ({type(e).__name__}: {e}); skipped")
            continue
        frames = [f for f in (convert(key, c) for c in sorted(path.rglob("*.csv"))) if f is not None]
        if not frames:
            print(f"[{key}] nothing usable; skipped")
            continue
        df = pd.concat(frames, ignore_index=True).drop_duplicates("url")
        df.to_csv(out, index=False)
        print(f"[{key}] wrote {len(df):,} rows, phishing rate {df.label.mean():.3f}")


if __name__ == "__main__":
    main()
