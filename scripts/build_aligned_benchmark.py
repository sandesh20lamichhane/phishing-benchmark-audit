"""Build a procedure-aligned phishing URL benchmark.

Both classes are collected the same way: as complete URLs observed in the
wild, with the same per-domain cap.

  phishing:   Phishing.Database `phishing-links-ACTIVE.txt` (GitHub mirror)
  legitimate: Common Crawl captures (HTTP 200, text/html) for domains sampled
              evenly from three Tranco rank bands (1-10k, 10k-100k, 100k-1M)

Two corpora are written to EXTRA_DIR (default data/raw/extra):

  aligned_natural.csv  both classes as collected, balanced by count
  aligned_matched.csv  additionally subsampled so that the joint distribution
                       of (HTTPS, empty path, query present) is identical in
                       both classes: the audited properties carry no
                       information by construction

plus aligned_manifest.json (sources, crawl id, list checksum, counts).

Needs network access to raw.githubusercontent.com and index.commoncrawl.org.
Common Crawl queries are cached per domain and the script resumes.

Usage:
    TRANCO=/path/to/tranco_top1m.csv python scripts/build_aligned_benchmark.py \
        [--domains 3000] [--per-domain 5] [--workers 4]
"""
import argparse
import hashlib
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402

PHISH_URL = ("https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/"
             "master/phishing-links-ACTIVE.txt")
CC_INFO = "https://index.commoncrawl.org/collinfo.json"
UA = {"User-Agent": "phishing-benchmark-audit/1.0 (academic research)"}
BANDS = ((1, 10_000), (10_001, 100_000), (100_001, 1_000_000))


def get(url, timeout=60, retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            time.sleep(2 ** attempt + random.random())
        except Exception:
            time.sleep(2 ** attempt + random.random())
    return None


def cap_per_domain(df, k, seed):
    return (df.sample(frac=1, random_state=seed)
              .groupby("domain", group_keys=False).head(k)
              .reset_index(drop=True))


def phishing_urls(out_dir, k, seed):
    raw = get(PHISH_URL, timeout=300)
    if not raw:
        sys.exit("could not download the Phishing.Database list")
    (out_dir / "phishing-links-ACTIVE.txt").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    urls = [u.strip() for u in raw.decode("utf-8", "replace").splitlines()]
    urls = [u for u in urls if u.lower().startswith(("http://", "https://"))]
    df = pd.DataFrame({"url": urls})
    df["norm"] = df.url.map(rx.normalise)
    df = df.drop_duplicates("norm")
    df["domain"] = df.url.map(rx.uf.registrable_domain)
    df = cap_per_domain(df[df.domain != ""], k, seed)
    print(f"[phish] {len(urls):,} http(s) links -> {len(df):,} after dedup and "
          f"per-domain cap {k}", flush=True)
    return df, sha


def crawl_api():
    info = json.loads(get(CC_INFO))
    return info[0]["id"], info[0]["cdx-api"]


def cc_urls(api, domain, cache_dir):
    path = cache_dir / f"{domain}.json"
    if path.exists():
        return json.loads(path.read_text())
    q = urllib.parse.urlencode({"url": domain, "matchType": "domain", "output": "json",
                                "fl": "url,status,mime", "filter": "status:200",
                                "limit": "300"})
    body = get(f"{api}?{q}", timeout=90)
    if body is None:
        return None                     # transient failure: not cached, retried next run
    urls = []
    for line in body.decode("utf-8", "replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "html" in rec.get("mime", "") and rec.get("url"):
            urls.append(rec["url"])
    path.write_text(json.dumps(urls))
    return urls


def legit_urls(tranco, n_domains, k, workers, cache_dir, seed):
    ranks = pd.read_csv(tranco, header=None, names=["rank", "domain"])
    rng = random.Random(seed)
    domains = []
    for lo, hi in BANDS:
        band = ranks[(ranks["rank"] >= lo) & (ranks["rank"] <= hi)].domain.tolist()
        domains += rng.sample(band, n_domains // len(BANDS))
    crawl_id, api = crawl_api()
    print(f"[legit] {len(domains):,} Tranco domains, crawl {crawl_id}", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    got, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cc_urls, api, d, cache_dir): d for d in domains}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            if res:
                got[futures[f]] = res
            if i % 200 == 0:
                print(f"  {i:,}/{len(domains):,} domains queried, {len(got):,} with "
                      f"captures [{(time.time() - t0) / 60:.1f} min]", flush=True)
    rows = []
    for d, urls in got.items():
        uniq = sorted(set(u for u in urls if not u.lower().endswith(("robots.txt", ".xml"))))
        for u in random.Random(f"{seed}-{d}").sample(uniq, min(k, len(uniq))):
            rows.append({"url": u, "tranco_domain": d})
    df = pd.DataFrame(rows)
    df["norm"] = df.url.map(rx.normalise)
    df = df.drop_duplicates("norm")
    df["domain"] = df.url.map(rx.uf.registrable_domain)
    print(f"[legit] {len(df):,} URLs from {df.tranco_domain.nunique():,} domains", flush=True)
    return df, crawl_id


def strata(urls):
    X = rx.uf.build_canonical_matrix(urls.reset_index(drop=True))
    return (X.is_https.astype(str) + (X.path_length == 0).astype(int).astype(str)
            + (X.query_length > 0).astype(int).astype(str)).to_numpy()


def matched(df, seed):
    df = df.assign(stratum=strata(df.url))
    parts = []
    for _, g in df.groupby("stratum"):
        n = min((g.label == 0).sum(), (g.label == 1).sum())
        if n:
            parts += [g[g.label == c].sample(n, random_state=seed) for c in (0, 1)]
    return pd.concat(parts).sample(frac=1, random_state=seed).drop(columns="stratum")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=int, default=3000)
    ap.add_argument("--per-domain", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = rx.EXTRA
    out.mkdir(parents=True, exist_ok=True)
    work = out.parent / "aligned_work"
    work.mkdir(parents=True, exist_ok=True)

    phish, sha = phishing_urls(work, a.per_domain, a.seed)
    legit, crawl_id = legit_urls(rx.TRANCO, a.domains, a.per_domain, a.workers,
                                 work / "cc_cache", a.seed)
    legit = legit[~legit.norm.isin(set(phish.norm))]
    n = min(len(phish), len(legit))
    natural = pd.concat([
        legit.sample(n, random_state=a.seed).assign(label=0, source="commoncrawl"),
        phish.sample(n, random_state=a.seed).assign(label=1, source="phishing.database"),
    ])[["url", "label", "source"]].sample(frac=1, random_state=a.seed)
    match = matched(natural, a.seed)
    natural.to_csv(out / "aligned_natural.csv", index=False)
    match.to_csv(out / "aligned_matched.csv", index=False)
    manifest = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phishing_source": PHISH_URL, "phishing_list_sha256": sha,
        "common_crawl_id": crawl_id, "tranco_bands": BANDS,
        "domains_requested": a.domains, "per_domain_cap": a.per_domain, "seed": a.seed,
        "natural": {"n": len(natural), "phish": int(natural.label.sum())},
        "matched": {"n": len(match), "phish": int(match.label.sum())},
    }
    (out.parent / "aligned_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
