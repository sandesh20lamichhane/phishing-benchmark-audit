"""The paper's headline transfer results, from one environment.

Reads the full-feature XGBoost cells of OUT_DIR/matrix_results.csv (written by
revision_experiments.py) and writes to OUT_DIR:

  headline_transfer_matrix.csv  5x5 seed-mean ROC-AUC with per-cell SD
  headline_stats.csv            within vs cross over dataset-pair cells: means,
                                SDs, exact permutation test, Cliff's delta,
                                inverted-cell count
  headline_asymmetry.csv        each pair's two directions over seeds:
                                magnitude, Mann-Whitney U, Holm correction

The paper's heat map is drawn from headline_transfer_matrix.csv by
scripts/make_heatmap.py. With --baselines this script also loads the corpora (RAW_DIR / CACHE_DIR as for revision_experiments.py) and adds

  headline_delong.csv           XGBoost against naive Bayes, logistic regression
                                and random forest on the seed-42 domain-grouped
                                split of each dataset, paired DeLong, Holm
  headline_phishstorm_probe.csv PhiUSIIL -> PhishStorm transfer, five seeds

Usage:
    python scripts/headline_stats.py [--baselines]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402
from cell_level_stats import cliffs_delta, exact_permutation_gap  # noqa: E402

ORDER = ["phiusiil", "mendeley_phishurl", "kaggle_malicious", "hannousse", "ebbu2017"]


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def xgb_full_cells():
    d = pd.read_csv(rx.OUT / "matrix_results.csv")
    return d[(d.model == "xgb") & (d.featset == "full") & (d.variant == "raw")
             & d.train.isin(ORDER) & d.test.isin(ORDER)]


def headline(d):
    cells = d.groupby(["train", "test"]).roc_auc.agg(["mean", "std"]).reset_index()
    mat = cells.pivot(index="train", columns="test", values="mean").reindex(index=ORDER, columns=ORDER)
    sd = cells.pivot(index="train", columns="test", values="std").reindex(index=ORDER, columns=ORDER)
    out = mat.round(6).add_suffix("_mean").join(sd.round(6).add_suffix("_sd"))
    out.to_csv(rx.OUT / "headline_transfer_matrix.csv")
    diag = np.eye(len(ORDER), dtype=bool)
    w, c = mat.to_numpy()[diag], mat.to_numpy()[~diag]
    gap, hits, total = exact_permutation_gap(w, c)
    stats = {"within_mean": w.mean(), "within_sd": w.std(ddof=1),
             "cross_mean": c.mean(), "cross_sd": c.std(ddof=1), "gap": gap,
             "perm_hits": hits, "perm_total": total, "perm_p_one_sided": hits / total,
             "cliffs_delta": cliffs_delta(w, c), "cross_min": c.min(), "cross_max": c.max(),
             "n_inverted": int((c < 0.5).sum()), "n_below_0.46": int((c < 0.46).sum()),
             "max_sd_within": sd.to_numpy()[diag].max(), "max_sd_cross": sd.to_numpy()[~diag].max()}
    pd.DataFrame([stats]).round(4).to_csv(rx.OUT / "headline_stats.csv", index=False)
    print(mat.round(4).to_string())
    print(pd.Series(stats).round(4).to_string())
    return mat


def asymmetry(d):
    rows = []
    for a, b in itertools.combinations(ORDER, 2):
        ab = d[(d.train == a) & (d.test == b)].sort_values("seed").roc_auc.to_numpy()
        ba = d[(d.train == b) & (d.test == a)].sort_values("seed").roc_auc.to_numpy()
        p = mannwhitneyu(ab, ba, alternative="two-sided", method="exact").pvalue
        rows.append({"a": a, "b": b, "a_to_b": ab.mean(), "b_to_a": ba.mean(),
                     "magnitude": abs(ab.mean() - ba.mean()), "p_raw": p,
                     "complete_separation": bool(ab.min() > ba.max() or ba.min() > ab.max())})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_raw)
    out = out.sort_values("magnitude", ascending=False).round(4)
    out.to_csv(rx.OUT / "headline_asymmetry.csv", index=False)
    print(out.to_string(index=False))


def baselines():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, str(rx.REPO))
    import src.evaluation.stats as st

    canon = rx.build_canon()
    others = {"NaiveBayes": lambda s: make_pipeline(StandardScaler(), GaussianNB()),
              "LogisticReg": lambda s: make_pipeline(StandardScaler(),
                                                     LogisticRegression(max_iter=2000, random_state=s)),
              "RandomForest": lambda s: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=s)}
    rows = []
    for key in ORDER:
        X, y = canon[key]["X"], canon[key]["y"]
        tr, te = rx.uf.grouped_split(X, y, canon[key]["domain"], seed=42)
        yte = np.asarray(y.iloc[te])
        ref = rx.make_xgb(42).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        for name, make in others.items():
            s = make(42).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
            a, b, _, p = st.delong_test(yte, ref, s)
            rows.append({"dataset": key, "model": name, "auc_xgb": a, "auc_other": b,
                         "diff": a - b, "p_raw": p})
        print(f"  [delong] {key}", flush=True)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_raw)
    out.round(6).to_csv(rx.OUT / "headline_delong.csv", index=False)
    print(out.round(4).to_string(index=False))

    probe = []
    src, dst = canon["phiusiil"], canon["phishstorm"]
    for seed in rx.SEEDS:
        m = rx.make_xgb(seed).fit(src["X"], src["y"])
        probe.append({"train": "phiusiil", "test": "phishstorm", "seed": seed,
                      **rx.metrics(dst["y"].to_numpy(), m.predict_proba(dst["X"])[:, 1])})
    probe = pd.DataFrame(probe)
    probe.to_csv(rx.OUT / "headline_phishstorm_probe.csv", index=False)
    print(f"[probe] PhiUSIIL -> PhishStorm {probe.roc_auc.mean():.4f} (sd {probe.roc_auc.std():.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    d = xgb_full_cells()
    headline(d)
    asymmetry(d)
    if a.baselines:
        baselines()


if __name__ == "__main__":
    main()
