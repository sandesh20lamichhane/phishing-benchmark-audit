import json, time, hashlib, subprocess
from pathlib import Path
import joblib


def _git_hash(repo="/content/phishing-detection"):
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


class Checkpoint:
    # Resumable checkpointing for Colab. Survives runtime resets via Drive.

    def __init__(self, root="/content/phishing-detection/checkpoints",
                 run_id=None, verbose=True):
        self.root = Path(root)
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = (json.loads(self.manifest_path.read_text())
                         if self.manifest_path.exists() else {})

    @staticmethod
    def _key(params):
        if params is None:
            return "default"
        blob = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:10]

    def _path(self, stage, params=None):
        return self.dir / (stage + "__" + self._key(params) + ".joblib")

    def exists(self, stage, params=None):
        return self._path(stage, params).exists()

    def save(self, stage, obj, params=None, meta=None):
        p = self._path(stage, params)
        joblib.dump(obj, p, compress=3)
        size_mb = round(p.stat().st_size / 1e6, 3)
        self.manifest[p.name] = {
            "stage": stage, "params": params, "meta": meta or {},
            "git_commit": _git_hash(),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "size_mb": size_mb,
        }
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, default=str))
        if self.verbose:
            print("[ckpt] saved " + p.name + "  (" + str(size_mb) + " MB)")
        return p

    def load(self, stage, params=None):
        p = self._path(stage, params)
        if not p.exists():
            raise FileNotFoundError("No checkpoint: " + str(p))
        if self.verbose:
            print("[ckpt] loaded " + p.name)
        return joblib.load(p)

    def cache(self, stage, fn, params=None, meta=None, force=False):
        if self.exists(stage, params) and not force:
            return self.load(stage, params)
        if self.verbose:
            print("[ckpt] computing " + stage + " ...")
        out = fn()
        self.save(stage, out, params=params, meta=meta)
        return out

    def summary(self):
        import pandas as pd
        if not self.manifest:
            return pd.DataFrame()
        return pd.DataFrame(self.manifest).T[
            ["stage", "saved_at", "size_mb", "git_commit"]]

    @classmethod
    def latest_run(cls, root="/content/phishing-detection/checkpoints", **kw):
        runs = sorted(p.name for p in Path(root).glob("run_*") if p.is_dir())
        if not runs:
            raise FileNotFoundError("No previous runs.")
        return cls(root=root, run_id=runs[-1], **kw)
