"""The paper's transfer heat map (Fig. 3), drawn from the committed results.

Reads the seed-mean ROC-AUC matrix in
reports/tables/revision/headline_transfer_matrix.csv (written by
scripts/headline_stats.py) and writes reports/figures/fig_heatmap.pdf and .png
in the style of notebook 09. Cells below 0.5 are bold and outlined.

Usage:
    python scripts/make_heatmap.py [--matrix CSV] [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ORD = ["phiusiil", "mendeley_phishurl", "kaggle_malicious", "hannousse", "ebbu2017"]
NAMES = {"phiusiil": "PhiUSIIL", "mendeley_phishurl": "Mendeley PhishURL",
         "kaggle_malicious": "Kaggle Malicious", "hannousse": "Hannousse", "ebbu2017": "Ebbu2017"}


def load_matrix(path):
    d = pd.read_csv(path, index_col="train")
    m = d[[f"{k}_mean" for k in ORD]]
    m.columns = ORD
    return m.reindex(index=ORD)


def draw(mat, out):
    plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300, "axes.linewidth": 0.8})
    labels = [NAMES[k] for k in ORD]
    v = mat.to_numpy()
    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    im = ax.imshow(v, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(ORD)))
    ax.set_yticks(range(len(ORD)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.grid(False)
    for i in range(len(ORD)):
        for j in range(len(ORD)):
            ax.text(j, i, f"{v[i, j]:.3f}", ha="center", va="center", fontsize=6,
                    fontweight="bold" if v[i, j] < 0.5 else "normal")
            if v[i, j] < 0.5:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="blue", lw=1.6))
    ax.set_xlabel("Evaluation dataset", fontsize=7)
    ax.set_ylabel("Training dataset", fontsize=7)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("ROC-AUC", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)
    off = v[~np.eye(len(ORD), dtype=bool)]
    print(f"[heatmap] {out}/fig_heatmap.pdf, .png ({int((off < 0.5).sum())} inverted cells outlined)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path,
                    default=REPO / "reports" / "tables" / "revision" / "headline_transfer_matrix.csv")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "figures")
    a = ap.parse_args()
    draw(load_matrix(a.matrix), a.out)


if __name__ == "__main__":
    main()
