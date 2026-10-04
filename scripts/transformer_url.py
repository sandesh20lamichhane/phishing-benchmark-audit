"""Fine-tuned pretrained transformer on raw URL strings, under the transfer protocol.

The model reads the URL as text through its own subword tokenizer, scheme
included. Default: distilbert-base-uncased (set MODEL_NAME to change it).
Needs a GPU for reasonable run time (about 2-4 minutes per fit on a T4).

Usage (same environment variables as revision_experiments.py):
    python scripts/transformer_url.py [seed ...]        # default seed 42
Results append to OUT_DIR/transformer_results.csv and resume.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402

MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
MAX_LEN = 128
EPOCHS = 2
BATCH = 64
LR = 5e-5
PATH = rx.OUT / "transformer_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = MODEL_NAME.split("/")[-1]


def tokenize(tok, urls):
    enc = tok(list(map(str, urls)), truncation=True, max_length=MAX_LEN,
              padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def fit(ids, mask, y, seed):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    data = TensorDataset(ids, mask, torch.tensor(np.asarray(y), dtype=torch.long))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(data, batch_size=BATCH, shuffle=True, generator=gen)
    total = EPOCHS * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: max(0.0, 1 - step / total))
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    model.train()
    for _ in range(EPOCHS):
        for i, m, t in loader:
            opt.zero_grad()
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = model(input_ids=i.to(DEVICE), attention_mask=m.to(DEVICE),
                             labels=t.to(DEVICE)).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, ids, mask):
    out = []
    for k in range(0, len(ids), 512):
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(input_ids=ids[k:k + 512].to(DEVICE),
                           attention_mask=mask[k:k + 512].to(DEVICE)).logits
        out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    seeds = tuple(int(s) for s in sys.argv[1:]) or (42,)
    print(f"model {MODEL_NAME} on {DEVICE}, seeds {seeds}", flush=True)
    rx.OUT.mkdir(parents=True, exist_ok=True)
    canon = rx.build_canon()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = {k: tokenize(tok, canon[k]["url"]) for k in rx.KEYS}
    done = rx.done_keys(PATH, ["train", "test", "seed"])
    t0 = time.time()
    for src in rx.KEYS:
        y = canon[src]["y"]
        ids, mask = enc[src]
        for seed in seeds:
            if (src, src, str(seed)) not in done:
                tr, te = rx.uf.grouped_split(canon[src]["X"], y, canon[src]["domain"], seed=seed)
                m = fit(ids[tr], mask[tr], y.iloc[tr], seed)
                rx.append(PATH, [{"model": TAG, "featset": "raw_string", "train": src,
                                  "test": src, "seed": seed, "kind": "within", "variant": "raw",
                                  **rx.metrics(y.iloc[te], predict(m, ids[te], mask[te]))}])
            todo = [d for d in rx.KEYS if d != src and (src, d, str(seed)) not in done]
            if not todo:
                continue
            m = fit(ids, mask, y, seed)
            rows = []
            for dst in todo:
                s = predict(m, *enc[dst])
                yd = canon[dst]["y"].to_numpy()
                for var, msk in rx.test_masks(canon, src, dst).items():
                    rows.append({"model": TAG, "featset": "raw_string", "train": src,
                                 "test": dst, "seed": seed, "kind": "cross", "variant": var,
                                 **rx.metrics(yd[msk], s[msk])})
            rx.append(PATH, rows)
            print(f"  [{TAG}] {src} seed {seed} [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
