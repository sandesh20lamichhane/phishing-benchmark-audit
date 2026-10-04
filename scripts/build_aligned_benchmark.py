"""Build a procedure-aligned phishing URL benchmark.

Both classes are collected the same way: as complete URLs observed in the
wild, with the same per-domain cap.

  phishing:   Phishing.Database `phishing-links-ACTIVE.txt` (GitHub mirror)
  legitimate: Common Crawl captures (HTTP 200, HTML) for domains sampled
              evenly from three Tranco rank bands (1-10k, 10k-100k, 100k-1M)

Legitimate URLs are read straight from the crawl's own URL index files on
data.commoncrawl.org (the `cluster.idx` table and byte ranges of the
`cdx-*.gz` shards), not through the CDX API at index.commoncrawl.org, which
rate-limits bulk lookups to the point of refusing nearly all of them.

Two corpora are written to EXTRA_DIR (default data/raw/extra):

  aligned_natural.csv  both classes as collected, balanced by count
  aligned_matched.csv  additionally subsampled so that the joint distribution
                       of (HTTPS, www. prefix, empty path, query present) is
                       identical in both classes: those properties carry no
                       information by construction

plus aligned_manifest.json (sources, crawl id, list checksum, counts).

Needs network access to raw.githubusercontent.com, index.commoncrawl.org (one
request, for the crawl list) and data.commoncrawl.org. Index lookups are
cached per domain and the script resumes.

Usage:
    TRANCO=/path/to/tranco_top1m.csv python scripts/build_aligned_benchmark.py \
        [--domains 3000] [--per-domain 5] [--workers 4] [--crawl CC-MAIN-YYYY-WW]
"""
import argparse
import bisect
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_experiments as rx  # noqa: E402

PHISH_URL = ("https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/"
             "master/phishing-links-ACTIVE.txt")
CC_INFO = "https://index.commoncrawl.org/collinfo.json"
CC_DATA = os.environ.get("CC_DATA", "https://data.commoncrawl.org")
MAX_BLOCKS = 3                          # index blocks read per domain (~3,000 captures each)
UA = {"User-Agent": "phishing-benchmark-audit/1.0 (academic research)"}
BANDS = ((1, 10_000), (10_001, 100_000), (100_001, 1_000_000))


def get(url, timeout=60, retries=6, headers=None):
    """Body bytes; b"" for a 404; None after repeated failures.

    A ranged request whose answer is not a 206 counts as a failure, so a server
    ignoring the range never streams a whole index shard.
    """
    headers = headers or {}
    req = urllib.request.Request(url, headers={**UA, **headers})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if "Range" in headers and r.status != 206:
                    return None
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            wait = e.headers.get("Retry-After") if e.headers else None
            time.sleep(min(float(wait), 120) if wait and wait.isdigit()
                       else 2 ** attempt + random.random())
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


def crawl_ids():
    body = get(CC_INFO)
    if not body:
        sys.exit("could not read the Common Crawl crawl list; pass --crawl CC-MAIN-YYYY-WW")
    return [c["id"] for c in json.loads(body)]


def load_cluster(crawl, work):
    """The crawl's cluster.idx: first SURT key, shard, offset and length of every index block."""
    path = work / f"cluster-{crawl}.idx"
    if not path.exists():
        url = f"{CC_DATA}/cc-index/collections/{crawl}/indexes/cluster.idx"
        body = get(url, timeout=600)
        if not body:
            return None
        path.write_bytes(body)
    keys, blocks = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            keys.append(parts[0].split(" ")[0])
            blocks.append((parts[1], int(parts[2]), int(parts[3])))
    return keys, blocks


def surt_prefix(domain):
    return ",".join(reversed(domain.lower().strip(".").split(".")))


def owns(key, prefix):
    """True when SURT key `key` belongs to the registrable domain with this prefix."""
    return key.startswith(prefix) and key[len(prefix):len(prefix) + 1] in (")", ",")


def domain_blocks(keys, prefix):
    """Indexes of up to MAX_BLOCKS index blocks holding the domain, spread over its range."""
    i = bisect.bisect_left(keys, prefix)
    first = max(i - 1, 0)               # captures may start inside the preceding block
    last = i
    while last < len(keys) and owns(keys[last], prefix):
        last += 1
    span = list(range(first, max(last, first + 1)))
    if len(span) <= MAX_BLOCKS:
        return span
    step = (len(span) - 1) / (MAX_BLOCKS - 1)
    return sorted({span[round(k * step)] for k in range(MAX_BLOCKS)})


def read_block(crawl, shard, offset, length):
    url = f"{CC_DATA}/cc-index/collections/{crawl}/indexes/{shard}"
    body = get(url, timeout=120, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
    if not body:
        return None
    try:
        return zlib.decompress(body, 16 + zlib.MAX_WBITS).decode("utf-8", "replace")
    except zlib.error:
        return None


def cc_urls(crawl, index, domain, cache_dir):
    path = cache_dir / f"{domain}.json"
    if path.exists():
        return json.loads(path.read_text())
    keys, blocks = index
    prefix = surt_prefix(domain)
    urls = []
    for b in domain_blocks(keys, prefix):
        text = read_block(crawl, *blocks[b])
        if text is None:
            return None                 # transient failure: not cached, retried next run
        for line in text.splitlines():
            key, _, rest = line.partition(" ")
            if not owns(key, prefix):
                continue
            try:
                rec = json.loads(rest.partition(" ")[2])
            except ValueError:
                continue
            mime = rec.get("mime-detected") or rec.get("mime", "")
            if rec.get("status") == "200" and "html" in mime and rec.get("url"):
                urls.append(rec["url"])
    path.write_text(json.dumps(urls))
    return urls


def legit_urls(tranco, n_domains, k, workers, work, seed, crawl=None):
    ranks = pd.read_csv(tranco, header=None, names=["rank", "domain"])
    rng = random.Random(seed)
    domains = []
    for lo, hi in BANDS:
        band = ranks[(ranks["rank"] >= lo) & (ranks["rank"] <= hi)].domain.tolist()
        domains += rng.sample(band, n_domains // len(BANDS))
    index = None
    for cid in ([crawl] if crawl else crawl_ids()[:3]):
        index = load_cluster(cid, work)
        if index:
            crawl = cid
            break
        print(f"[legit] no URL index for {cid}; trying the previous crawl", flush=True)
    if not index:
        sys.exit("could not download a Common Crawl cluster.idx from data.commoncrawl.org")
    print(f"[legit] {len(domains):,} Tranco domains, crawl {crawl}, "
          f"{len(index[0]):,} index blocks", flush=True)
    cache_dir = work / f"cc_cache_{crawl}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    got, failed, empty, t0 = {}, 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cc_urls, crawl, index, d, cache_dir): d for d in domains}
        for i, f in enumerate(as_completed(futures), 1):
            if f.cancelled():
                continue
            res = f.result()
            if res is None:
                failed += 1
            elif res:
                got[futures[f]] = res
            else:
                empty += 1
            if i % 200 == 0 or i == len(domains):
                print(f"  {i:,}/{len(domains):,} domains: {len(got):,} with captures, "
                      f"{empty:,} without, {failed:,} failed "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
            if i == 200 and failed > 150:
                for g in futures:
                    g.cancel()
                sys.exit("most index reads are failing: data.commoncrawl.org is refusing "
                         "requests from this machine. Re-run later or with --workers 1.")
    if failed:
        print(f"[legit] {failed:,} domains failed; re-run to retry only those", flush=True)
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
    stats = {"domains_with_captures": len(got), "domains_without": empty, "domains_failed": failed}
    return df, crawl, stats


def strata(urls):
    urls = urls.reset_index(drop=True)
    X = rx.uf.build_canonical_matrix(urls)
    www = rx.strip_www(urls.str.strip()).ne(urls.str.strip())
    return (X.is_https.astype(str) + www.astype(int).astype(str)
            + (X.path_length == 0).astype(int).astype(str)
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
    ap.add_argument("--crawl", default=None, help="Common Crawl id; default: the newest with an index")
    a = ap.parse_args()

    out = rx.EXTRA
    out.mkdir(parents=True, exist_ok=True)
    work = out.parent / "aligned_work"
    work.mkdir(parents=True, exist_ok=True)

    phish, sha = phishing_urls(work, a.per_domain, a.seed)
    legit, crawl_id, cc_stats = legit_urls(rx.TRANCO, a.domains, a.per_domain, a.workers,
                                           work, a.seed, a.crawl)
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
        "common_crawl_id": crawl_id, "common_crawl_lookup": cc_stats,
        "matched_strata": "https x www. x empty path x query", "tranco_bands": BANDS,
        "domains_requested": a.domains, "per_domain_cap": a.per_domain, "seed": a.seed,
        "natural": {"n": len(natural), "phish": int(natural.label.sum())},
        "matched": {"n": len(match), "phish": int(match.label.sum())},
    }
    (out.parent / "aligned_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
