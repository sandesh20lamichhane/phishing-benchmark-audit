"""Character-level CNN on raw URL strings, under the transfer protocol.

A URLNet-style character branch (Le et al., 2018): embedding, parallel
convolutions of width 3-6, global max-pooling, one hidden layer. It reads the
raw string, scheme included, so it can exploit any structural convention the
feature-based models can, plus any it cannot.

Usage (same environment variables as revision_experiments.py):
    python scripts/char_cnn.py [cpu_threads]
Uses a GPU when one is available. Results append to
OUT_DIR/char_cnn_results.csv and resume.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402

MAX_LEN = 200
SEEDS = (42, 43, 44)
EPOCHS = 5
BATCH = 256
PATH = rx.OUT / "char_cnn_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(urls):
    """Printable ASCII -> 1..95, anything else -> 96, padding 0. Case kept."""
    out = np.zeros((len(urls), MAX_LEN), dtype=np.int64)
    for i, u in enumerate(urls):
        codes = [ord(c) - 31 if 32 <= ord(c) < 127 else 96 for c in str(u)[:MAX_LEN]]
        out[i, :len(codes)] = codes
    return torch.from_numpy(out)


class CharCNN(nn.Module):
    def __init__(self, vocab=97, emb=32, filters=128, widths=(3, 4, 5, 6)):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, w) for w in widths)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(filters * len(widths), 64),
                                  nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([torch.relu(c(e)).amax(dim=2) for c in self.convs], dim=1)
        return self.head(h).squeeze(1)


def fit(x, y, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CharCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    y = torch.tensor(np.asarray(y), dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        order = rng.permutation(len(y))
        for i in range(0, len(order), BATCH):
            b = torch.from_numpy(order[i:i + BATCH])
            opt.zero_grad()
            loss_fn(model(x[b].to(DEVICE)), y[b].to(DEVICE)).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x):
    return np.concatenate([torch.sigmoid(model(x[i:i + 2048].to(DEVICE))).cpu().numpy()
                           for i in range(0, len(x), 2048)])


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    print(f"device: {DEVICE}", flush=True)
    rx.OUT.mkdir(parents=True, exist_ok=True)
    canon = rx.build_canon()
    enc = {k: encode(canon[k]["url"]) for k in rx.KEYS}
    done = rx.done_keys(PATH, ["train", "test", "seed"])
    t0 = time.time()
    for src in rx.KEYS:
        y = canon[src]["y"]
        for seed in SEEDS:
            if (src, src, str(seed)) not in done:
                tr, te = rx.uf.grouped_split(canon[src]["X"], y, canon[src]["domain"], seed=seed)
                m = fit(enc[src][tr], y.iloc[tr], seed)
                rx.append(PATH, [{"model": "char_cnn", "featset": "raw_string", "train": src,
                                  "test": src, "seed": seed, "kind": "within", "variant": "raw",
                                  **rx.metrics(y.iloc[te], predict(m, enc[src][te]))}])
            todo = [d for d in rx.KEYS if d != src and (src, d, str(seed)) not in done]
            if not todo:
                continue
            m = fit(enc[src], y, seed)
            rows = []
            for dst in todo:
                s = predict(m, enc[dst])
                yd = canon[dst]["y"].to_numpy()
                for var, mask in rx.test_masks(canon, src, dst).items():
                    rows.append({"model": "char_cnn", "featset": "raw_string", "train": src,
                                 "test": dst, "seed": seed, "kind": "cross", "variant": var,
                                 **rx.metrics(yd[mask], s[mask])})
            rx.append(PATH, rows)
            print(f"  [char_cnn] {src} seed {seed} [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
