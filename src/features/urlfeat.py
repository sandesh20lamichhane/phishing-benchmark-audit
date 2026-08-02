"""Canonical URL-lexical features + split strategies.

Hardened against malformed URLs. Real corpora contain entries that make
urllib.parse raise: non-numeric or out-of-range ports, unbalanced square
brackets that look like broken IPv6 literals, and embedded control
characters. urlparse() is lazy about this - it raises on the .hostname
and .port properties rather than at parse time - so every access is
guarded here.
"""
import re, math
from urllib.parse import urlparse, SplitResult
import numpy as np
import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "account", "update", "secure", "webscr",
    "banking", "confirm", "password", "credential", "invoice", "payment",
    "billing", "suspend", "unlock", "recover", "wallet", "support",
]
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorte.st", "rebrand.ly",
}
IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
HEX_HOST = re.compile(r"^0x[0-9a-f]+$", re.I)
CTRL = re.compile(r"[\x00-\x1f\x7f]")

_EMPTY = SplitResult(scheme="", netloc="", path="", query="", fragment="")


def safe_parse(url):
    """Parse a URL, never raising. Returns a SplitResult-like object.

    Falls back through three stages: parse as given; strip control
    characters and square brackets and retry; return an empty result.
    """
    s = str(url).strip()
    if "://" not in s:
        s = "http://" + s
    try:
        p = urlparse(s)
        _ = p.netloc          # force any lazy validation we can
        return p
    except ValueError:
        pass
    cleaned = CTRL.sub("", s).replace("[", "").replace("]", "")
    try:
        return urlparse(cleaned)
    except ValueError:
        return _EMPTY


def safe_hostname(p):
    """netloc-derived hostname, tolerating malformed authority sections."""
    try:
        h = p.hostname
        if h is not None:
            return h.lower()
    except ValueError:
        pass
    netloc = getattr(p, "netloc", "") or ""
    netloc = netloc.rsplit("@", 1)[-1]          # drop userinfo
    netloc = CTRL.sub("", netloc).replace("[", "").replace("]", "")
    host = netloc.split(":", 1)[0]
    return host.lower()


def safe_port(p):
    """Port if it is a valid integer in range, else None. Never raises."""
    try:
        return p.port
    except ValueError:
        return None


def is_malformed(url):
    """True if the URL needed the fallback path. Report this rate."""
    s = str(url).strip()
    if "://" not in s:
        s = "http://" + s
    try:
        p = urlparse(s)
        _ = p.hostname
        _ = p.port
        return False
    except ValueError:
        return True


def _entropy(s):
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def canonical_url_features(url, extractor=None):
    """~32 lexical features computable from a URL alone, identically for any source."""
    url = str(url).strip()
    if "://" not in url:
        url = "http://" + url

    p = safe_parse(url)
    host = safe_hostname(p)
    path = getattr(p, "path", "") or ""
    query = getattr(p, "query", "") or ""

    subdomain = domain = suffix = ""
    if extractor is not None:
        try:
            ext = extractor(url)
            subdomain, domain, suffix = ext.subdomain, ext.domain, ext.suffix
        except Exception:
            extractor = None
    if not (domain or suffix):
        parts = [x for x in host.split(".") if x]
        suffix = parts[-1] if len(parts) > 1 else ""
        domain = parts[-2] if len(parts) > 2 else (parts[0] if parts else "")
        subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""

    registrable = f"{domain}.{suffix}".strip(".")
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "n_dots": url.count("."),
        "n_hyphens": url.count("-"),
        "n_underscores": url.count("_"),
        "n_slashes": path.count("/"),
        "n_question": url.count("?"),
        "n_equals": url.count("="),
        "n_at": url.count("@"),
        "n_ampersand": url.count("&"),
        "n_percent": url.count("%"),
        "n_digits": digits,
        "n_letters": letters,
        "digit_ratio": digits / max(len(url), 1),
        "n_subdomains": len([s for s in subdomain.split(".") if s]),
        "subdomain_length": len(subdomain),
        "domain_length": len(domain),
        "tld_length": len(suffix),
        "is_ip_host": int(bool(IPV4.match(host))),
        "is_hex_host": int(bool(HEX_HOST.match(host))),
        "is_https": int(str(getattr(p, "scheme", "")).lower() == "https"),
        "has_port": int(safe_port(p) is not None),
        "is_shortener": int(registrable in SHORTENERS),
        "has_punycode": int("xn--" in host),
        "has_double_slash_path": int("//" in path),
        "n_suspicious_words": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        "url_entropy": _entropy(url),
        "host_entropy": _entropy(host),
        "longest_token_path": max([len(t) for t in re.split(r"[/\-_.]", path) if t] or [0]),
        "has_https_token_in_host": int("https" in host),
    }


def build_canonical_matrix(urls, use_tldextract=True):
    extractor = None
    if use_tldextract:
        extractor = get_extractor()
    rows = [canonical_url_features(u, extractor) for u in urls]
    return pd.DataFrame(rows, index=getattr(urls, "index", None))


_DEFAULT_EXTRACTOR = None


def get_extractor():
    """Cached tldextract instance using the bundled PSL snapshot (no network)."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        try:
            import tldextract
            _DEFAULT_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())
        except Exception:
            _DEFAULT_EXTRACTOR = False
    return _DEFAULT_EXTRACTOR or None


def registrable_domain(url, extractor="auto"):
    """eTLD+1. Returns the bare IP for IP-literal hosts. Never raises."""
    s = str(url).strip()
    if "://" not in s:
        s = "http://" + s
    host = safe_hostname(safe_parse(s))
    if not host:
        return ""
    if IPV4.match(host):
        return host
    if extractor == "auto":
        extractor = get_extractor()
    if extractor is not None:
        try:
            e = extractor(s)
            reg = f"{e.domain}.{e.suffix}".strip(".").strip("[]")
            if reg:
                return reg
        except Exception:
            pass
    parts = [x for x in host.strip("[]").split(".") if x]
    return ".".join(parts[-2:]) if len(parts) >= 2 else host.strip("[]")


# ------------------------------------------------------------------ splits
def grouped_split(X, y, groups, test_size=0.2, seed=42):
    """No registrable domain appears in both train and test."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    return tr, te


def stratified_split(X, y, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)


def rebalance_to_base_rate(y, target_rate, seed=42, index=None):
    """Downsample the phishing class to a realistic prevalence."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)) if index is None else np.asarray(index)
    pos = idx[np.asarray(y) == 1]
    neg = idx[np.asarray(y) == 0]
    n_pos = int(round(target_rate * len(neg) / (1 - target_rate)))
    n_pos = min(n_pos, len(pos))
    keep = np.concatenate([neg, rng.choice(pos, size=n_pos, replace=False)])
    rng.shuffle(keep)
    return keep
