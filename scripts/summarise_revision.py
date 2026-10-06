"""Summarise reports/tables/revision/*.csv into the tables used in the paper."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_level_stats import cliffs_delta, exact_permutation_gap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REV = Path(os.environ.get("OUT_DIR", REPO / "reports" / "tables" / "revision"))
CORE = ["phiusiil", "mendeley_phishurl", "kaggle_malicious", "hannousse", "ebbu2017"]
# aligned_matched is a subsample of aligned_natural: transfer between them tests on training data
NESTED = {frozenset({"aligned_natural", "aligned_matched"})}
AUDITED = {"is_https", "url_length", "path_length", "query_length", "n_slashes",
           "n_question", "n_equals", "n_ampersand", "has_double_slash_path",
           "longest_token_path"}


def load():
    frames = [pd.read_csv(p) for p in (REV / "matrix_results.csv", REV / "char_cnn_results.csv",
                                       REV / "transformer_results.csv") if p.exists()]
    df = pd.concat(frames, ignore_index=True)
    nested = [frozenset({a, b}) in NESTED for a, b in zip(df.train, df.test)]
    return df[~pd.Series(nested, index=df.index)]


def core_only(df):
    return df[df.train.isin(CORE) & df.test.isin(CORE)]


def aligned_summary(c):
    """Within-dataset results on the aligned corpora and transfer between them and the core."""
    rows = []
    for (model, fs), g in c.groupby(["model", "featset"]):
        for a in sorted(set(g.train) - set(CORE)):
            w = g[(g.train == a) & (g.test == a)]
            to_core = g[(g.train == a) & g.test.isin(CORE)]
            from_core = g[g.train.isin(CORE) & (g.test == a)]
            rows.append({"model": model, "featset": fs, "aligned": a,
                         "within": round(w.roc_auc.mean(), 4),
                         "within_tpr_fpr_0.01": round(w["tpr_at_fpr_0.01"].mean(), 4),
                         "to_core": round(to_core.roc_auc.mean(), 4),
                         "to_core_inverted": int((to_core.roc_auc < 0.5).sum()),
                         "from_core": round(from_core.roc_auc.mean(), 4),
                         "from_core_inverted": int((from_core.roc_auc < 0.5).sum())})
    return pd.DataFrame(rows)


def cells(df, variant="raw"):
    d = df[df.variant == variant]
    return (d.groupby(["model", "featset", "train", "test", "kind"], as_index=False)
             [["roc_auc", "pr_auc", "tpr_at_fpr_0.001", "tpr_at_fpr_0.01", "n_test", "n_pos"]]
             .mean())


def gate(c):
    ref = pd.read_csv(REPO / "reports" / "tables" / "transfer_matrix_5way.csv")
    ref = ref.groupby(["train", "test"], as_index=False).roc_auc.mean()
    new = c[(c.model == "xgb") & (c.featset == "full")][["train", "test", "roc_auc"]]
    m = ref.merge(new, on=["train", "test"], suffixes=("_paper", "_rerun"))
    m["diff"] = (m.roc_auc_rerun - m.roc_auc_paper).abs()
    inv_p = set(map(tuple, m[m.roc_auc_paper < 0.5][["train", "test"]].to_numpy()))
    inv_r = set(map(tuple, m[m.roc_auc_rerun < 0.5][["train", "test"]].to_numpy()))
    return {"cells": len(m), "max_abs_diff": round(m["diff"].max(), 4),
            "mean_abs_diff": round(m["diff"].mean(), 4),
            "inverted_paper": len(inv_p), "inverted_rerun": len(inv_r),
            "same_inverted_cells": inv_p == inv_r}


def permutation_gap(w, x, n_random=200_000, seed=0):
    """Exact when feasible, otherwise Monte Carlo with the +1 correction."""
    if math.comb(len(w) + len(x), len(w)) <= n_random:
        _, hits, total = exact_permutation_gap(w, x)
        return hits / total
    rng = np.random.default_rng(seed)
    allv = np.concatenate([w, x])
    obs = w.mean() - x.mean()
    hits = 0
    for _ in range(n_random):
        perm = rng.permutation(allv)
        hits += perm[:len(w)].mean() - perm[len(w):].mean() >= obs - 1e-12
    return (hits + 1) / (n_random + 1)


def model_summary(c):
    rows = []
    for (model, fs), g in c.groupby(["model", "featset"]):
        w = g[g.kind == "within"].roc_auc.to_numpy()
        x = g[g.kind == "cross"].roc_auc.to_numpy()
        rows.append({"model": model, "featset": fs,
                     "within": round(w.mean(), 4), "cross": round(x.mean(), 4),
                     "gap": round(w.mean() - x.mean(), 4),
                     "n_within": len(w), "n_cross": len(x),
                     "perm_p": permutation_gap(w, x), "cliffs_delta": round(cliffs_delta(w, x), 3),
                     "inverted": int((x < 0.5).sum()), "min_cross": round(x.min(), 4),
                     "within_pr_auc": round(g[g.kind == "within"].pr_auc.mean(), 4),
                     "within_tpr_fpr_0.001": round(g[g.kind == "within"]["tpr_at_fpr_0.001"].mean(), 4),
                     "within_tpr_fpr_0.01": round(g[g.kind == "within"]["tpr_at_fpr_0.01"].mean(), 4),
                     "cross_pr_auc": round(g[g.kind == "cross"].pr_auc.mean(), 4),
                     "cross_tpr_fpr_0.01": round(g[g.kind == "cross"]["tpr_at_fpr_0.01"].mean(), 4)})
    return pd.DataFrame(rows)


def overlap_summary(df):
    rows = []
    for (model, fs), g in df[df.kind == "cross"].groupby(["model", "featset"]):
        piv = (g.groupby(["train", "test", "variant"]).agg(roc_auc=("roc_auc", "mean"),
                                                           n_test=("n_test", "mean"))
                .reset_index())
        for var, gv in piv.groupby("variant"):
            rows.append({"model": model, "featset": fs, "variant": var,
                         "cross_mean": round(gv.roc_auc.mean(), 4),
                         "inverted": int((gv.roc_auc < 0.5).sum()),
                         "min": round(gv.roc_auc.min(), 4),
                         "median_test_retained": round(gv.n_test.median(), 0)})
    return pd.DataFrame(rows)


def overlap_cells(df):
    g = df[(df.model == "xgb") & (df.featset == "full") & (df.kind == "cross")]
    piv = g.pivot_table(index=["train", "test"], columns="variant", values="roc_auc", aggfunc="mean")
    n = g.pivot_table(index=["train", "test"], columns="variant", values="n_test", aggfunc="mean")
    piv["retained_url"] = (n["url_dedup"] / n["raw"]).round(3)
    piv["retained_domain"] = (n["domain_dedup"] / n["raw"]).round(3)
    return piv.round(4).reset_index()


def shap_summary():
    p = REV / "shap_separation.csv"
    if not p.exists():
        return None, None
    s = pd.read_csv(p)
    inv = s[s.roc_auc < 0.5]
    top = (inv.sort_values("sep").groupby(["train", "test"]).head(3)
              [["train", "test", "roc_auc", "feature", "sep"]])
    rows = []
    for (a, b), g in inv.groupby(["train", "test"]):
        neg = g[g.sep < 0]
        total = -neg.sep.sum()
        audited = -neg[neg.feature.isin(AUDITED)].sep.sum()
        rows.append({"train": a, "test": b, "roc_auc": round(g.roc_auc.iloc[0], 4),
                     "neg_sep_total": round(total, 3),
                     "audited_share_of_neg_sep": round(audited / total, 3) if total else np.nan,
                     "is_https_sep": round(g[g.feature == "is_https"].sep.iloc[0], 3)})
    return top, pd.DataFrame(rows)


def main():
    df = load()
    c = cells(df)
    pd.set_option("display.width", 200)
    print("== gate (xgb full vs paper Table 3)")
    print(gate(c))
    ms = model_summary(c)
    ms.to_csv(REV / "summary_models.csv", index=False)
    print("\n== models, every corpus (raw test sets)\n", ms.to_string(index=False))
    if set(c.train) - set(CORE):
        mc = model_summary(core_only(c))
        mc.to_csv(REV / "summary_models_core.csv", index=False)
        print("\n== models, five core corpora only\n", mc.to_string(index=False))
        al = aligned_summary(c)
        al.to_csv(REV / "summary_aligned.csv", index=False)
        print("\n== aligned corpora\n", al.to_string(index=False))
    ov = overlap_summary(df)
    ov.to_csv(REV / "summary_overlap.csv", index=False)
    print("\n== overlap-controlled transfer\n", ov.to_string(index=False))
    oc = overlap_cells(df)
    oc.to_csv(REV / "overlap_cells_xgb_full.csv", index=False)
    print("\n== xgb full, per cell\n", oc.to_string(index=False))
    top, sh = shap_summary()
    if sh is not None:
        top.to_csv(REV / "shap_top_inverting.csv", index=False)
        sh.to_csv(REV / "shap_inverted_summary.csv", index=False)
        print("\n== SHAP: top inverting features\n", top.to_string(index=False))
        print("\n", sh.to_string(index=False))


if __name__ == "__main__":
    main()
