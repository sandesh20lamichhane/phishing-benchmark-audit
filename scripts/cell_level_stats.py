"""Aggregate within- vs cross-dataset gap with the dataset-pair cell as the unit.

Seeds of one cell share training and evaluation data, so they are not
independent observations. This script averages seeds per cell and runs an
exact permutation test over all assignments of cells to the within group.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path(__file__).resolve().parents[1] / "reports" / "tables"


def cliffs_delta(a, b):
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def exact_permutation_gap(w, c):
    """One-sided exact p for mean(w) - mean(c) over all relabellings."""
    values = np.concatenate([w, c])
    k, n = len(w), len(values)
    observed = w.mean() - c.mean()
    hits = total = 0
    for idx in itertools.combinations(range(n), k):
        mask = np.zeros(n, dtype=bool)
        mask[list(idx)] = True
        gap = values[mask].mean() - values[~mask].mean()
        hits += gap >= observed - 1e-12
        total += 1
    return observed, hits, total


def main():
    m = pd.read_csv(TABLES / "transfer_matrix_5way.csv")
    cells = m.groupby(["train", "test", "kind"], as_index=False).roc_auc.mean()
    w = cells.loc[cells.kind == "within", "roc_auc"].to_numpy()
    c = cells.loc[cells.kind == "cross", "roc_auc"].to_numpy()
    gap, hits, total = exact_permutation_gap(w, c)
    row = {
        "n_within_cells": len(w), "n_cross_cells": len(c),
        "within_mean": round(w.mean(), 4), "within_sd": round(w.std(ddof=1), 4),
        "cross_mean": round(c.mean(), 4), "cross_sd": round(c.std(ddof=1), 4),
        "gap": round(gap, 4), "perm_hits": hits, "perm_total": total,
        "perm_p_one_sided": hits / total,
        "cliffs_delta": round(cliffs_delta(w, c), 4),
        "cross_min": round(c.min(), 4), "cross_max": round(c.max(), 4),
        "n_inverted": int((c < 0.5).sum()),
    }
    out = pd.DataFrame([row])
    out.to_csv(TABLES / "cell_level_stats.csv", index=False)
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
