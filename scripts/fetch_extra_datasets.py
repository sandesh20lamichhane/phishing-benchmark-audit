"""Fetch additional raw-URL phishing corpora into EXTRA_DIR as url,label CSVs.

Each source is downloaded with kagglehub (Kaggle credentials required, as in
notebook 06). Label values are mapped explicitly; rows with any other value
are dropped and counted, never guessed. A source that fails to download is
reported and skipped.

You can also add a corpus by hand: put a CSV with `url` and `label`
(1 = phishing, 0 = legitimate) columns into EXTRA_DIR. Every CSV there joins
the transfer matrix.

After fetching, every extra corpus is checked against the full files of the
core corpora (and each other): normalised-URL containment above 0.5 means it
is not independent data, and its CSV is moved to RAW_DIR/duplicates so it
stays out of the matrix. The table is written to OUT_DIR/extra_provenance.csv.

Usage:
    python scripts/fetch_extra_datasets.py [--check-only]
"""
import shutil
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


def full_sets():
    """Normalised URL sets of the full core files and of every extra CSV."""
    sets = {}
    for key, rel, ucol, _, _ in rx.LOAD_ORDER:
        d = pd.read_csv(rx.RAW / rel, low_memory=False)
        col = [c for c in d.columns if c.lower() == ucol.lower()][0]
        sets[key] = set(d[col].astype(str).map(rx.normalise))
    extras = {p.stem: set(pd.read_csv(p, low_memory=False)["url"].astype(str).map(rx.normalise))
              for p in sorted(rx.EXTRA.glob("*.csv"))}
    return sets, extras


def provenance(threshold=0.5):
    """Keep an extra corpus only if no core corpus or already-kept extra contains it."""
    if not rx.EXTRA.is_dir() or not any(rx.EXTRA.glob("*.csv")):
        return
    core, extras = full_sets()
    kept, rows = {}, []
    dup_dir = rx.RAW / "duplicates"
    for e, es in extras.items():
        ours = e.startswith("aligned_")    # built by build_aligned_benchmark.py; matched is a subset of natural by design
        found = []
        for k, ks in (core if ours else {**core, **kept}).items():
            shared = len(es & ks)
            found.append({"extra": e, "other": k, "extra_urls": len(es), "other_urls": len(ks),
                          "shared": shared,
                          "containment": round(shared / max(min(len(es), len(ks)), 1), 4)})
        rows += found
        top = max(found, key=lambda r: r["containment"])
        if ours:
            print(f"[{e}] max containment {top['containment']:.3f} ({top['other']}): built here, kept")
        elif top["containment"] > threshold:
            dup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(rx.EXTRA / f"{e}.csv"), str(dup_dir / f"{e}.csv"))
            print(f"[{e}] containment {top['containment']:.3f} with {top['other']} "
                  f"({top['shared']:,} shared URLs): not independent, moved to {dup_dir}")
        else:
            kept[e] = es
            print(f"[{e}] max containment {top['containment']:.3f} ({top['other']}): kept")
    rx.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rx.OUT / "extra_provenance.csv", index=False)


def main():
    if "--check-only" in sys.argv:
        provenance()
        return
    import kagglehub
    rx.EXTRA.mkdir(parents=True, exist_ok=True)
    for key, slug in SOURCES.items():
        out = rx.EXTRA / f"{key}.csv"
        if out.exists() or (rx.RAW / "duplicates" / f"{key}.csv").exists():
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
    provenance()


if __name__ == "__main__":
    main()
