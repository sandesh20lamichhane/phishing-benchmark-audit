"""Revision experiments for the journal version of the paper.

Runs, on the five independent corpora and the PhishStorm probe, under the
protocol of notebook 09 (same loaders, same 60k cap and sampling sequence,
same XGBoost configuration, same domain-grouped within-dataset split):

  1. XGBoost transfer matrix with the full, targeted-ablation and host-only
     feature sets (five seeds), and logistic regression in the same spaces.
  2. Every cross-dataset cell evaluated three ways: on the full test corpus,
     with test URLs that also occur in the training corpus removed, and with
     test URLs whose registrable domain occurs in the training corpus removed.
  3. ROC-AUC, PR-AUC (average precision) and TPR at 0.1% and 1% FPR for
     every cell.
  4. A raw-string baseline: character n-gram TF-IDF with logistic regression.
  5. TreeSHAP attribution of the inverted cells.
  6. Tranco top-1M membership by class, and a check of the Mendeley PhishURL
     loader, which concatenates two distributed files.
  7. The structural audit and URL containment for every corpus, including
     any additional corpora.

Additional corpora: every CSV with `url` and `label` (1 = phishing) columns in
EXTRA_DIR (default data/raw/extra) joins the transfer matrix after the five
original corpora. The original corpora and their samples are unchanged.

Usage:
    RAW_DIR=/path/to/data/raw TRANCO=/path/to/tranco_top1m.csv \
        python scripts/revision_experiments.py [step ...]

Steps: load, audit, xgb, lr, tfidf, shap, tranco, mendeley (default: all).
Every step appends to its CSV under OUT_DIR (default
reports/tables/revision/) and resumes.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, DMatrix

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import src.features.urlfeat as uf  # noqa: E402

RAW = Path(os.environ.get("RAW_DIR", REPO / "data" / "raw"))
TRANCO = Path(os.environ.get("TRANCO", REPO / "data" / "external" / "tranco" / "tranco_top1m.csv"))
OUT = Path(os.environ.get("OUT_DIR", REPO / "reports" / "tables" / "revision"))
CACHE = Path(os.environ.get("CACHE_DIR", REPO / "data" / "cache"))
EXTRA = Path(os.environ.get("EXTRA_DIR", RAW / "extra"))

CAP = 60000
SEEDS = (42, 43, 44, 45, 46)
CORE = ["phiusiil", "mendeley_phishurl", "kaggle_malicious", "hannousse", "ebbu2017"]
EXTRAS = sorted(p.stem for p in EXTRA.glob("*.csv")) if EXTRA.is_dir() else []
KEYS = CORE + EXTRAS
LOAD_ORDER = [
    ("phiusiil",          "phiusiil/phiusiil.csv",                        "URL", "label",  0),
    ("hannousse",         "hannousse/hannousse.csv",                      "url", "status", "phishing"),
    ("kaggle_malicious",  "kaggle_malicious/kaggle_malicious_urls.csv",   "url", "label",  1),
    ("mendeley_phishurl", "mendeley_phishurl/mendeley_phishurl_urls.csv", "url", "label",  1),
    ("ebbu2017",          "ebbu2017/ebbu2017.csv",                        "url", "label",  1),
    ("phishstorm",        "phishstorm/phishstorm.csv",                    "url", "label",  1),
]
DROP_TARGETED = {"is_https", "url_length", "path_length", "query_length", "n_slashes",
                 "n_question", "n_equals", "n_ampersand", "has_double_slash_path",
                 "longest_token_path"}
HOST_ONLY = ["host_length", "n_subdomains", "subdomain_length", "domain_length",
             "tld_length", "is_ip_host", "is_hex_host", "has_port", "is_shortener",
             "has_punycode", "host_entropy", "has_https_token_in_host"]
VARIANTS = ("raw", "url_dedup", "domain_dedup")


# ------------------------------------------------------------------ data
def normalise(u):
    s = str(u).strip().lower()
    for pre in ("https://", "http://"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    if s.startswith("www."):
        s = s[4:]
    return s.rstrip("/")


def load_corpora():
    """Same loaders, cap and RNG sequence as notebook 09."""
    rng = np.random.default_rng(42)
    corpora = {}
    for key, rel, ucol, lcol, phish in LOAD_ORDER:
        d = pd.read_csv(RAW / rel, low_memory=False)
        uc = [c for c in d.columns if c.lower() == ucol.lower()][0]
        lc = [c for c in d.columns if c.lower() == lcol.lower()][0]
        lab = ((d[lc].astype(str).str.lower() == str(phish).lower()).astype(int)
               if isinstance(phish, str) else (d[lc] == phish).astype(int))
        df = pd.DataFrame({"url": d[uc].astype(str), "label": lab})
        if len(df) > CAP:
            idx = np.sort(rng.choice(len(df), CAP, replace=False))
            df = df.iloc[idx].reset_index(drop=True)
        corpora[key] = df
    for key in EXTRAS:
        d = pd.read_csv(EXTRA / f"{key}.csv", low_memory=False)
        df = pd.DataFrame({"url": d["url"].astype(str), "label": d["label"].astype(int)})
        if len(df) > CAP:
            idx = np.sort(rng.choice(len(df), CAP, replace=False))
            df = df.iloc[idx].reset_index(drop=True)
        corpora[key] = df
    return corpora


def build_canon_tag():
    """Cache key: the extra corpora's names, sizes and modification times, so a
    rebuilt corpus never reuses a stale cache."""
    if not EXTRAS:
        return "canon_core"
    sig = ",".join(f"{k}:{(EXTRA / f'{k}.csv').stat().st_size}:{int((EXTRA / f'{k}.csv').stat().st_mtime)}"
                   for k in EXTRAS)
    return "canon_" + hashlib.md5(sig.encode()).hexdigest()[:8]


RESULT_FILES = ("matrix_results.csv", "char_cnn_results.csv", "transformer_results.csv")


def drop_stale_results(canon):
    """Remove result rows for any corpus whose content changed since they were computed.

    Every runner skips (train, test, seed) cells already in its results file, so
    without this a rebuilt corpus would silently keep the old corpus's numbers.
    """
    fp_path = OUT / "corpus_fingerprints.json"
    fps = {k: hashlib.md5("\n".join(v["url"] + "\t" + v["y"].astype(str)).encode()).hexdigest()
           for k, v in canon.items()}
    old = json.loads(fp_path.read_text()) if fp_path.exists() else {}
    changed = [k for k in fps if k in old and old[k] != fps[k]]
    for name in RESULT_FILES if changed else ():
        p = OUT / name
        if p.exists():
            d = pd.read_csv(p)
            keep = ~(d.train.isin(changed) | d.test.isin(changed))
            if (~keep).any():
                d[keep].to_csv(p, index=False)
                print(f"[stale] {name}: removed {int((~keep).sum()):,} rows computed on the "
                      f"previous version of {', '.join(changed)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(json.dumps({**old, **fps}, indent=1))


def build_canon():
    path = CACHE / f"{build_canon_tag()}.pkl"
    if path.exists():
        canon = pd.read_pickle(path)
        drop_stale_results(canon)
        return canon
    CACHE.mkdir(parents=True, exist_ok=True)
    canon = {}
    for key, df in load_corpora().items():
        t0 = time.time()
        urls = df.url.astype(str)
        canon[key] = {
            "url": urls.reset_index(drop=True),
            "X": uf.build_canonical_matrix(urls).reset_index(drop=True),
            "y": df.label.reset_index(drop=True),
            "domain": urls.map(uf.registrable_domain).reset_index(drop=True),
            "norm": urls.map(normalise).reset_index(drop=True),
        }
        print(f"[load] {key}: n={len(df):,} phishing_rate={df.label.mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    pd.to_pickle(canon, path)
    drop_stale_results(canon)
    return canon


def test_masks(canon, src, dst):
    """Boolean masks over dst for each evaluation variant."""
    d = canon[dst]
    url_seen = d["norm"].isin(set(canon[src]["norm"]))
    dom_seen = d["domain"].isin(set(canon[src]["domain"]))
    return {"raw": np.ones(len(d["y"]), dtype=bool),
            "url_dedup": ~url_seen.to_numpy(),
            "domain_dedup": ~dom_seen.to_numpy()}


# ------------------------------------------------------------------ metrics
def tpr_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def metrics(y, s):
    y = np.asarray(y)
    n_pos = int(y.sum())
    row = {"n_test": len(y), "n_pos": n_pos}
    if n_pos == 0 or n_pos == len(y):
        return {**row, "roc_auc": np.nan, "pr_auc": np.nan,
                "tpr_at_fpr_0.001": np.nan, "tpr_at_fpr_0.01": np.nan}
    return {**row,
            "roc_auc": roc_auc_score(y, s),
            "pr_auc": average_precision_score(y, s),
            "tpr_at_fpr_0.001": tpr_at_fpr(y, s, 0.001),
            "tpr_at_fpr_0.01": tpr_at_fpr(y, s, 0.01)}


# ------------------------------------------------------------------ models
def make_xgb(seed):
    return XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
                         eval_metric="logloss", random_state=seed, verbosity=0)


def make_lr(seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=seed))


def make_tfidf_lr(seed):
    return make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True,
                        max_features=200_000, sublinear_tf=True, dtype=np.float32),
        LogisticRegression(max_iter=2000, C=4.0, random_state=seed))


def featsets(canon):
    allf = list(canon["phiusiil"]["X"].columns)
    return {"full": allf,
            "targeted_ablation": [f for f in allf if f not in DROP_TARGETED],
            "host_only": HOST_ONLY}


def done_keys(path, cols):
    if not path.exists():
        return set()
    prev = pd.read_csv(path)
    return set(map(tuple, prev[cols].astype(str).to_numpy()))


def append(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_matrix(canon, model_name, make, inputs, feat_names, seeds_cross):
    """One model family over all feature sets and all 25 cells.

    inputs(key, feats) returns the model input for corpus `key`.
    """
    path = OUT / "matrix_results.csv"
    keycols = ["model", "featset", "train", "test", "seed"]
    done = done_keys(path, keycols)
    t0 = time.time()
    for fsn in feat_names:
        feats = featsets(canon).get(fsn)
        for src in KEYS:
            for seed in SEEDS:
                # within: domain-grouped split
                if (model_name, fsn, src, src, str(seed)) not in done:
                    Xs, ys = inputs(src, feats), canon[src]["y"]
                    tr, te = uf.grouped_split(canon[src]["X"], ys, canon[src]["domain"], seed=seed)
                    Xtr = Xs.iloc[tr] if hasattr(Xs, "iloc") else Xs[tr]
                    Xte = Xs.iloc[te] if hasattr(Xs, "iloc") else Xs[te]
                    m = make(seed).fit(Xtr, ys.iloc[tr])
                    s = m.predict_proba(Xte)[:, 1]
                    append(path, [{"model": model_name, "featset": fsn, "train": src,
                                   "test": src, "seed": seed, "kind": "within",
                                   "variant": "raw", **metrics(ys.iloc[te], s)}])
                # cross: train once on all of src, evaluate every other corpus
                if seed not in seeds_cross:
                    continue
                todo = [d for d in KEYS if d != src
                        and (model_name, fsn, src, d, str(seed)) not in done]
                if not todo:
                    continue
                m = make(seed).fit(inputs(src, feats), canon[src]["y"])
                rows = []
                for dst in todo:
                    s = m.predict_proba(inputs(dst, feats))[:, 1]
                    y = canon[dst]["y"].to_numpy()
                    for var, mask in test_masks(canon, src, dst).items():
                        rows.append({"model": model_name, "featset": fsn, "train": src,
                                     "test": dst, "seed": seed, "kind": "cross",
                                     "variant": var, **metrics(y[mask], s[mask])})
                append(path, rows)
                print(f"  [{model_name}/{fsn}] {src} seed {seed} "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)


# ------------------------------------------------------------------ steps
def step_xgb(canon):
    run_matrix(canon, "xgb", make_xgb, lambda k, f: canon[k]["X"][f],
               ["full", "targeted_ablation", "host_only"], seeds_cross=SEEDS)


def step_lr(canon):
    run_matrix(canon, "lr", make_lr, lambda k, f: canon[k]["X"][f],
               ["full", "targeted_ablation", "host_only"], seeds_cross=(42,))


def step_tfidf(canon):
    run_matrix(canon, "tfidf_lr", make_tfidf_lr, lambda k, f: canon[k]["url"].to_numpy(),
               ["raw_string"], seeds_cross=(42,))


def step_shap(canon):
    """Per-feature class separation of TreeSHAP contributions on each cross cell.

    For a test corpus, sep_j = mean contribution of feature j on phishing URLs
    minus its mean on legitimate URLs. Positive sep pushes the classes apart in
    the right direction; negative sep is the feature driving inversion.
    """
    feats = featsets(canon)["full"]
    rows = []
    for src in KEYS:
        m = make_xgb(42).fit(canon[src]["X"][feats], canon[src]["y"])
        booster = m.get_booster()
        for dst in KEYS:
            if dst == src:
                continue
            X, y = canon[dst]["X"][feats], canon[dst]["y"].to_numpy()
            contrib = booster.predict(DMatrix(X), pred_contribs=True)[:, :-1]
            sep = contrib[y == 1].mean(axis=0) - contrib[y == 0].mean(axis=0)
            auc = roc_auc_score(y, m.predict_proba(X)[:, 1])
            for f, v in zip(feats, sep):
                rows.append({"train": src, "test": dst, "roc_auc": auc,
                             "feature": f, "sep": float(v)})
        print(f"  [shap] {src}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "shap_separation.csv", index=False)


def strip_www(urls):
    """Remove a leading www./wwwN. label from the host, keeping any scheme."""
    return urls.str.replace(r"^((?:[a-z][a-z0-9+.-]*://)?)www\d*\.(?=[^/:?#]*\.)", r"\1",
                            regex=True, case=False)


def step_www(canon):
    """XGBoost on features recomputed after removing a leading www. from every host.

    The legitimate classes of PhiUSIIL and Mendeley PhishURL carry the prefix
    almost without exception and Kaggle Malicious's almost never does; this
    tests whether that convention drives the host-level inversions. Split,
    domains and overlap masks are those of the unmodified corpora.
    """
    path = CACHE / f"{build_canon_tag()}_nowww.pkl"
    if path.exists():
        nowww = pd.read_pickle(path)
    else:
        nowww = {}
        for k in KEYS:
            X = uf.build_canonical_matrix(strip_www(canon[k]["url"])).reset_index(drop=True)
            nowww[k] = {**canon[k], "X": X}
            print(f"  [www] features recomputed for {k}", flush=True)
        pd.to_pickle(nowww, path)
    run_matrix(nowww, "xgb_nowww", make_xgb, lambda k, f: nowww[k]["X"][f],
               ["host_only", "targeted_ablation", "full"], seeds_cross=SEEDS)


def step_tranco(canon):
    ranks = pd.read_csv(TRANCO, header=None, names=["rank", "domain"])
    top = {n: set(ranks.domain[ranks["rank"] <= n]) for n in (10_000, 100_000, 1_000_000)}
    rows = []
    for key in KEYS + ["phishstorm"]:
        d = canon[key]
        for cls, name in ((0, "legit"), (1, "phish")):
            dom = d["domain"][d["y"] == cls]
            path_empty = d["X"]["path_length"][d["y"] == cls] <= 1
            row = {"dataset": key, "class": name, "n": len(dom),
                   "bare_root_pct": round(100 * path_empty.mean(), 2)}
            for n, s in top.items():
                row[f"in_tranco_top{n // 1000}k_pct"] = round(100 * dom.isin(s).mean(), 2)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tranco_membership.csv", index=False)
    print(out.to_string(index=False))


def step_mendeley(_canon):
    base = RAW / "mendeley_phishurl"
    merged = pd.read_csv(base / "mendeley_phishurl_urls.csv")
    a = pd.read_csv(base / "URL dataset.csv")
    b = pd.read_csv(base / "Phishing URLs.csv")
    a_ph = set(a.url[a.type.str.lower() == "phishing"].map(normalise))
    b_all = set(b.url.map(normalise))
    row = {
        "merged_rows": len(merged), "merged_unique_urls": merged.url.nunique(),
        "url_dataset_rows": len(a), "url_dataset_phishing": int((a.type.str.lower() == "phishing").sum()),
        "phishing_urls_rows": len(b),
        "phishing_urls_in_url_dataset_phishing": len(b_all & a_ph),
        "phishing_urls_containment": round(len(b_all & a_ph) / max(len(b_all), 1), 4),
        "merged_duplicate_rows": int(merged.duplicated("url").sum()),
        "merged_conflicting_labels": int(merged.groupby("url").label.nunique().gt(1).sum()),
    }
    out = pd.DataFrame([row])
    out.to_csv(OUT / "mendeley_loader_check.csv", index=False)
    print(out.T.to_string(header=False))


def step_audit(canon):
    """Structural composition by class, and URL containment between corpora."""
    rows = []
    for key in KEYS + ["phishstorm"]:
        d = canon[key]
        url = d["url"].str.strip().str.lower()
        X = d["X"]
        for cls, name in ((0, "legit"), (1, "phish")):
            m = (d["y"] == cls).to_numpy()
            rows.append({"dataset": key, "class": name, "n": int(m.sum()),
                         "scheme_present_pct": round(100 * url[m].str.contains("://", regex=False).mean(), 2),
                         "https_pct": round(100 * url[m].str.startswith("https://").mean(), 2),
                         "empty_path_pct": round(100 * (X["path_length"][m] == 0).mean(), 2),
                         "query_pct": round(100 * (X["query_length"][m] > 0).mean(), 2),
                         "www_pct": round(100 * strip_www(url[m]).ne(url[m]).mean(), 2)})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "structural_audit_all.csv", index=False)
    print(audit.to_string(index=False))
    sets = {k: set(canon[k]["norm"]) for k in KEYS + ["phishstorm"]}
    pairs = []
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = len(sets[a] & sets[b])
            pairs.append({"a": a, "b": b, "shared": inter,
                          "containment": round(inter / max(min(len(sets[a]), len(sets[b])), 1), 4)})
    cont = pd.DataFrame(pairs)
    cont.to_csv(OUT / "containment_all.csv", index=False)
    print(cont[cont.containment > 0.05].to_string(index=False))


STEPS = {"audit": step_audit, "xgb": step_xgb, "lr": step_lr, "tfidf": step_tfidf, "shap": step_shap,
         "tranco": step_tranco, "mendeley": step_mendeley, "www": step_www}


def purge(keys):
    """Remove every result row that trains or tests on one of `keys`, so the runners recompute them."""
    for name in RESULT_FILES:
        p = OUT / name
        if p.exists():
            d = pd.read_csv(p)
            hit = d.train.isin(keys) | d.test.isin(keys)
            d[~hit].to_csv(p, index=False)
            print(f"[purge] {name}: removed {int(hit.sum()):,} rows", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:2] == ["purge"]:
        purge(sys.argv[2:])
        return
    steps = sys.argv[1:] or ["load", *STEPS]
    canon = build_canon()
    for name in steps:
        if name == "load":
            continue
        print(f"== {name}", flush=True)
        STEPS[name](canon)


if __name__ == "__main__":
    main()
