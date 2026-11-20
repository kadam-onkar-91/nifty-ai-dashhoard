"""fundamentals_hub.py -- ONE place that answers "give me the fundamentals of NSE stock X", fast and without hammering Yahoo.

Why this file exists (from the live logs): Streamlit Cloud's IP is rejected by Yahoo's `.info` endpoint -- every call returns
`HTTP Error 401 ... Invalid Crumb`, hundreds of times per refresh (50 Nifty stocks + banks + the open stock, each retried).
That made the page wait for dead calls and left the Fundamentals box on "unavailable".

What it does instead:
  1. Disk + memory cache (12 h fresh; any age is served as a last resort and labelled with its age).  Fundamentals change over
     quarters, not seconds, so a cached copy is correct and costs 0 ms.
  2. Yahoo `.info` only while a CIRCUIT BREAKER is closed.  Two failures in a row -> Yahoo is skipped for 30 min (doubling up to 4 h),
     so the 401 storm stops after the first couple of calls instead of repeating forever.
  3. screener.in (public company page, no login) as the fallback source: P/E, P/B, ROE, ROCE, dividend yield, market cap,
     sales / profit growth.  Parsed defensively -- a field that can't be read stays None (never invented).
  4. Everything is returned in the SAME key names Yahoo uses (trailingPE, priceToBook, returnOnEquity ...), so every existing caller
     (bank scoring, Nifty-50 aggregate, stock panel) works unchanged.  `_source` / `_age_s` tell where / how old the numbers are.
"""
import json
import logging
import os
import re
import html as _html
import tempfile
import threading
import time
import urllib.parse

try:                                    # yfinance prints every 401 through its own logger -> hundreds of identical lines per refresh
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
except Exception:
    pass

logger = logging.getLogger(__name__)

FRESH_S = 12 * 3600
STALE_OK_S = 14 * 24 * 3600
_DISK = os.path.join(tempfile.gettempdir(), "fundamentals_hub_cache_v1.json")
_LOCK = threading.Lock()
_MEM = {}                                # SYMBOL -> {"ts": float, "data": dict}
_DISK_LOADED = {"done": False}
_NET_SLOTS = threading.Semaphore(5)      # at most 5 simultaneous screener requests

FIELDS = ("longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "profitMargins",
          "revenueGrowth", "earningsGrowth", "debtToEquity", "dividendYield", "beta", "52WeekChange")
_NUMERIC = ("marketCap", "trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "profitMargins", "revenueGrowth", "earningsGrowth",
            "debtToEquity", "dividendYield", "beta", "52WeekChange")

# ------------------------------------------------------------------------------------------------------------ Yahoo circuit breaker
_Y = {"fails": 0, "open_until": 0.0, "opens": 0}


def yahoo_open():
    """True while the breaker lets a Yahoo `.info` call through."""
    return time.time() >= _Y["open_until"]


def yahoo_state():
    return {"fails": _Y["fails"], "blocked_for_s": max(0.0, _Y["open_until"] - time.time())}


def _yahoo_failed():
    with _LOCK:
        _Y["fails"] += 1
        if _Y["fails"] >= 2:
            _Y["opens"] += 1
            _Y["open_until"] = time.time() + min(4 * 3600, 1800 * (2 ** min(_Y["opens"] - 1, 3)))
            _Y["fails"] = 0
            logger.warning("fundamentals_hub: Yahoo .info is rejected (401) -> skipped for %.0f min", (_Y["open_until"] - time.time()) / 60)


def _yahoo_ok():
    with _LOCK:
        _Y["fails"] = 0
        _Y["opens"] = 0
        _Y["open_until"] = 0.0


def _from_yahoo(sym):
    if not yahoo_open():
        return None
    try:
        import yfinance as yf
        info = yf.Ticker(sym + ".NS").info or {}
    except Exception:
        _yahoo_failed()
        return None
    out = {k: info.get(k) for k in FIELDS}
    if not any(out.get(k) is not None for k in _NUMERIC):
        _yahoo_failed()
        return None
    _yahoo_ok()
    out["_source"] = "Yahoo Finance"
    return out


# ------------------------------------------------------------------------------------------------------------ screener.in fallback
_TAG = re.compile(r"<[^>]+>")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _num(txt):
    m = _NUM.search(_html.unescape(txt or "").replace("\u20b9", " "))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None


def _clean(html):
    return re.sub(r"\s+", " ", _html.unescape(_TAG.sub(" ", html or ""))).strip()


def parse_screener(html):
    """screener.in company page -> Yahoo-style dict (values we cannot read stay None).  Pure function (unit tested)."""
    if not html:
        return {}
    ratios = {}
    for name, val in re.findall(r'<li[^>]*>\s*<span class="name">(.*?)</span>\s*<span class="nowrap value">(.*?)</span>\s*</li>', html, flags=re.S):
        ratios[_clean(name).lower()] = _clean(val)
    out = {k: None for k in FIELDS}
    mc = _num(ratios.get("market cap"))
    if mc is not None:
        out["marketCap"] = mc * 1e7                                     # screener shows Rs crore
    out["trailingPE"] = _num(ratios.get("stock p/e"))
    price, book = _num(ratios.get("current price")), _num(ratios.get("book value"))
    if price and book and book > 0:
        out["priceToBook"] = round(price / book, 2)
    roe = _num(ratios.get("roe"))
    if roe is not None:
        out["returnOnEquity"] = roe / 100.0
    dy = _num(ratios.get("dividend yield"))
    if dy is not None:
        out["dividendYield"] = dy / 100.0

    def growth(title):
        m = re.search(re.escape(title) + r".*?</table>", html, flags=re.S | re.I)
        if not m:
            return None
        rows = re.findall(r"<td[^>]*>\s*(.*?)\s*</td>\s*<td[^>]*>\s*(.*?)\s*</td>", m.group(0), flags=re.S)
        vals = {_clean(a).lower().rstrip(":"): _num(_clean(b)) for a, b in rows}
        for k in ("ttm", "3 years", "5 years"):
            if vals.get(k) is not None:
                return vals[k] / 100.0
        return None

    out["revenueGrowth"] = growth("Compounded Sales Growth")
    out["earningsGrowth"] = growth("Compounded Profit Growth")
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.S)
    if h1:
        out["longName"] = _clean(h1.group(1)) or None
    sec = re.search(r'<a[^>]*href="/market/[^"]*"[^>]*title="Sector"[^>]*>(.*?)</a>', html, flags=re.S)
    if sec:
        out["sector"] = _clean(sec.group(1)) or None
    ind = re.search(r'<a[^>]*href="/market/[^"]*"[^>]*title="Industry"[^>]*>(.*?)</a>', html, flags=re.S)
    if ind:
        out["industry"] = _clean(ind.group(1)) or None
    return out


def _from_screener(sym):
    try:
        import requests
    except Exception:
        return None
    slug = urllib.parse.quote(sym.upper(), safe="&-")
    hdr = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
           "Accept": "text/html,application/xhtml+xml"}
    with _NET_SLOTS:
        for path in ("consolidated/", ""):
            try:
                r = requests.get(f"https://www.screener.in/company/{slug}/{path}", headers=hdr, timeout=7)
                if r.status_code != 200:
                    continue
                out = parse_screener(r.text)
                if any(out.get(k) is not None for k in _NUMERIC):
                    out["_source"] = "screener.in"
                    return out
            except Exception:
                continue
    return None


# ------------------------------------------------------------------------------------------------------------ cache
def _load_disk():
    if _DISK_LOADED["done"]:
        return
    _DISK_LOADED["done"] = True
    try:
        with open(_DISK, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        if isinstance(j, dict):
            for k, v in j.items():
                if isinstance(v, dict) and isinstance(v.get("data"), dict):
                    _MEM.setdefault(k, v)
    except Exception:
        pass


def _save_disk():
    try:
        tmp = _DISK + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_MEM, fh)
        os.replace(tmp, _DISK)
    except Exception:
        pass


def cached(sym):
    """(data, age_s) from memory/disk, or (None, None).  Never touches the network."""
    sym = str(sym).upper().strip()
    with _LOCK:
        _load_disk()
        e = _MEM.get(sym)
    if not e:
        return None, None
    return e["data"], time.time() - e["ts"]


def get_info(sym, network=True, max_age=FRESH_S):
    """Yahoo-style fundamentals dict for NSE symbol `sym` ({} when nothing is known).  Fast path = cache."""
    sym = str(sym).upper().strip()
    if not sym:
        return {}
    data, age = cached(sym)
    if data is not None and age is not None and age <= max_age:
        return dict(data, _age_s=age)
    if network:
        fresh = _from_yahoo(sym) or _from_screener(sym)
        if fresh:
            with _LOCK:
                _MEM[sym] = {"ts": time.time(), "data": fresh}
                _save_disk()
            return dict(fresh, _age_s=0.0)
    if data is not None and age is not None and age <= STALE_OK_S:
        return dict(data, _age_s=age)                 # old but real -- always labelled with its age by the caller
    return {}
