"""Do the Kaggle Malicious inversions survive once its PhishStorm block is fixed?

Kaggle Malicious contains every PhishStorm URL, with the label inverted on
95.9% of them (scripts/provenance_checks.py). All six inverted cells of the
published transfer matrix involve Kaggle Malicious, so this reruns its cells
with two corrected copies made by `provenance_checks.py --write-variants DIR`:

  kaggle_nophishstorm  the PhishStorm rows removed
  kaggle_relabelled    the PhishStorm rows kept with PhishStorm's labels

Same protocol as the main matrix: 60k cap and RNG sequence, domain-grouped
split, XGBoost on the full, targeted-ablation and host-only feature sets, seeds
42-46. Each variant is trained and tested within itself and in both directions
against the four other core corpora. Results go to
OUT_DIR/kaggle_relabel_results.csv; with the main matrix in
OUT_DIR/matrix_results.csv, a summary next to the original Kaggle Malicious
cells goes to OUT_DIR/kaggle_relabel_summary.csv.

Usage:
    RAW_DIR=... EXTRA_DIR=DIR python scripts/kaggle_relabel_check.py
(EXTRA_DIR must hold only the two variant files.)
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402

OTHERS = ["phiusiil", "mendeley_phishurl", "hannousse", "ebbu2017"]
VARIANTS = ["kaggle_nophishstorm", "kaggle_relabelled"]
FEATSETS = ["full", "targeted_ablation", "host_only"]
PATH = rx.OUT / "kaggle_relabel_results.csv"


def run(canon):
    done = rx.done_keys(PATH, ["model", "featset", "train", "test", "seed"])
    for fsn in FEATSETS:
        feats = rx.featsets(canon)[fsn]
        X = lambda k: canon[k]["X"][feats]  # noqa: E731
        for seed in rx.SEEDS:
            for v in VARIANTS:
                if ("xgb", fsn, v, v, str(seed)) not in done:
                    tr, te = rx.uf.grouped_split(canon[v]["X"], canon[v]["y"], canon[v]["domain"], seed=seed)
                    m = rx.make_xgb(seed).fit(X(v).iloc[tr], canon[v]["y"].iloc[tr])
                    rx.append(PATH, [{"model": "xgb", "featset": fsn, "train": v, "test": v, "seed": seed,
                                      "kind": "within", "variant": "raw",
                                      **rx.metrics(canon[v]["y"].iloc[te], m.predict_proba(X(v).iloc[te])[:, 1])}])
            for src in VARIANTS + OTHERS:
                dsts = OTHERS if src in VARIANTS else VARIANTS
                todo = [d for d in dsts if ("xgb", fsn, src, d, str(seed)) not in done]
                if not todo:
                    continue
                m = rx.make_xgb(seed).fit(X(src), canon[src]["y"])
                rx.append(PATH, [{"model": "xgb", "featset": fsn, "train": src, "test": d, "seed": seed,
                                  "kind": "cross", "variant": "raw",
                                  **rx.metrics(canon[d]["y"].to_numpy(), m.predict_proba(X(d))[:, 1])}
                                 for d in todo])
            print(f"  [relabel] {fsn} seed {seed} {time.strftime('%H:%M:%S')}", flush=True)


def summarise():
    new = pd.read_csv(PATH)
    parts = [(v, new, v) for v in VARIANTS]
    main = rx.OUT / "matrix_results.csv"
    if main.exists():
        old = pd.read_csv(main)
        parts.insert(0, ("kaggle_malicious", old[(old.model == "xgb") & (old.variant == "raw")], "kaggle_malicious"))
    rows = []
    for name, d, k in parts:
        for fsn in FEATSETS:
            f = d[d.featset == fsn]
            row = {"kaggle_version": name, "featset": fsn,
                   "within": f[(f.train == k) & (f.test == k)].roc_auc.mean()}
            for o in OTHERS:
                row[f"to_{o}"] = f[(f.train == k) & (f.test == o)].roc_auc.mean()
                row[f"from_{o}"] = f[(f.train == o) & (f.test == k)].roc_auc.mean()
            cells = [row[f"{d_}_{o}"] for d_ in ("to", "from") for o in OTHERS]
            row["cross_mean"] = sum(cells) / len(cells)
            row["inverted"] = sum(c < 0.5 for c in cells)
            rows.append(row)
    s = pd.DataFrame(rows).round(4)
    s.to_csv(rx.OUT / "kaggle_relabel_summary.csv", index=False)
    print(s.to_string(index=False))


def main():
    missing = [v for v in VARIANTS if v not in rx.EXTRAS]
    if missing:
        sys.exit(f"EXTRA_DIR={rx.EXTRA} lacks {missing}; run provenance_checks.py --write-variants first")
    rx.OUT.mkdir(parents=True, exist_ok=True)
    run(rx.build_canon())
    summarise()


if __name__ == "__main__":
    main()
