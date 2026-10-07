"""gemini_pool.py -- ONE shared, quota-aware Gemini caller for the whole dashboard.

Why this exists
---------------
* Chat said "AI se jawab nahi mila: 503" -- the old rotation only rotated on 429; a 503
  ("model overloaded") crashed straight to the user instead of trying another model/key.
* "Saari 5 keys ka quota khatam" -- every call hammered key #1 first, re-sent the *whole*
  dashboard each time, and the trade-review cache key contained the live price, so the same
  setup was re-reviewed every 30 s refresh.
* Trade review was hard-wired to gemini-2.5-flash (being retired -> 404 -> "HTTPError").

What it does
------------
* Spreads calls round-robin over all keys and over a MODEL CHAIN (each model has its own quota
  bucket, so when flash-lite is exhausted, 3.5-flash / flash-latest still work).
* 429  -> that (key, model) pair rests (until the stated retry time; ~2 h if it is a *daily* limit).
* 503/5xx -> that MODEL rests ~20 s for every key (it is an overload, not your quota).
* 404 -> model marked retired for 6 h.   401/403 -> that key rests 6 h.
* Module-level state => shared by all browser sessions and all call sites (chat + trade review).
* Short response cache so identical requests never spend quota twice.
"""
from app_logging import get_logger
logger = get_logger(__name__)

import re
import time
import hashlib
import threading

_LOCK = threading.Lock()
_COOL = {}        # (key_hash|'*', model|'*') -> resume_ts
_DEAD = {}        # model -> ts when found retired
_CACHE = {}       # sig -> (ts, value)
_RR = {"n": 0}
_STATS = {"calls": 0, "ok": 0, "cache_hits": 0, "errors": 0, "last_model": None, "last_error": ""}

DEAD_TTL = 6 * 3600
CHAT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-lite-latest", "gemini-flash-latest"]
REVIEW_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


class PoolHTTPError(Exception):
    def __init__(self, status, message=""):
        super().__init__(f"HTTP {status} {message}")
        self.status = status
        self.message = message


class GeminiUnavailable(Exception):
    def __init__(self, last_exc, wait_s=0.0, attempts=None, details=None):
        super().__init__(str(last_exc))
        self.last_exc = last_exc
        self.wait_s = wait_s
        self.attempts = attempts or []
        self.details = details or []


def _kh(key):
    return hashlib.sha1(str(key).encode()).hexdigest()[:8]


def parse_error(exc):
    """-> (status:int|None, message:str) for requests-style PoolHTTPError and google-genai errors."""
    if isinstance(exc, PoolHTTPError):
        return exc.status, exc.message
    msg = str(exc)
    status = None
    for attr in ("code", "status_code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            status = v
            break
    if status is None:
        m = re.search(r"\b([45]\d\d)\b", msg)
        if m:
            status = int(m.group(1))
    low = msg.lower()
    if status is None:
        if "resource_exhausted" in low or "quota" in low:
            status = 429
        elif "unavailable" in low or "overloaded" in low or "high demand" in low:
            status = 503
    return status, msg


def report_error(key, model, status, msg):
    now = time.time()
    low = (msg or "").lower()
    with _LOCK:
        _STATS["errors"] += 1
        _STATS["last_error"] = f"{model}: {status} {low[:120]}"
        if status == 429:
            m = re.search(r"retry(?: in|delay'?:? ?'?)\s*([\d.]+)\s*s", low)
            daily = ("perday" in low.replace(" ", "")) or "per day" in low
            wait = float(m.group(1)) + 2 if m else (7200.0 if daily else 60.0)
            if daily and m is None:
                wait = 7200.0
            _COOL[(_kh(key), model)] = now + max(wait, 20.0)
        elif status in (500, 502, 503, 504):
            _COOL[("*", model)] = now + 20.0
        elif status == 404 or (status == 400 and any(t in low for t in ("no longer available", "not found", "not supported", "deprecated"))):
            _DEAD[model] = now
        elif status in (401, 403):
            _COOL[(_kh(key), "*")] = now + 6 * 3600


def _blocked(key, model, now):
    kh = _kh(key)
    for k in ((kh, model), ("*", model), (kh, "*")):
        if _COOL.get(k, 0) > now:
            return True
    return False


def plan(api_keys, models):
    """Pass 1: ONE free key for every model (each model has its own quota bucket, so a 429 on
    model #1 no longer burns all attempts before model #2 is ever tried). Pass 2: remaining combos."""
    now = time.time()
    keys = [k for k in (api_keys or []) if k]
    first, rest = [], []
    with _LOCK:
        _RR["n"] += 1
        start = _RR["n"] % max(len(keys), 1)
        for model in models:
            if now - _DEAD.get(model, 0) < DEAD_TTL:
                continue
            seen_first = False
            for i in range(len(keys)):
                key = keys[(start + i) % len(keys)]
                if _blocked(key, model, now):
                    continue
                if not seen_first:
                    first.append((key, model)); seen_first = True
                else:
                    rest.append((key, model))
    return first + rest


def seconds_until_available(api_keys, models):
    now = time.time()
    best = None
    with _LOCK:
        for model in models:
            if now - _DEAD.get(model, 0) < DEAD_TTL:
                continue
            for key in api_keys or []:
                kh = _kh(key)
                t = max(_COOL.get(k, 0) for k in ((kh, model), ("*", model), (kh, "*")))
                rem = max(0.0, t - now)
                best = rem if best is None else min(best, rem)
    return best or 0.0


def cache_get(sig, ttl):
    with _LOCK:
        hit = _CACHE.get(sig)
        if hit and time.time() - hit[0] < ttl:
            _STATS["cache_hits"] += 1
            return hit[1]
    return None


def cache_set(sig, value):
    with _LOCK:
        _CACHE[sig] = (time.time(), value)
        if len(_CACHE) > 60:
            for k in sorted(_CACHE, key=lambda z: _CACHE[z][0])[:20]:
                _CACHE.pop(k, None)


def run(api_keys, models, attempt_fn, max_attempts=30, deadline_s=None):
    """attempt_fn(key, model) -> value, or raises. Returns (value, key_index, model).
    deadline_s: stop trying new key/model combos after this many seconds in total (None = no limit)."""
    _t0 = time.time()
    keys = [k for k in (api_keys or []) if k]
    last, attempts, details = None, [], []
    combos = plan(keys, models)
    if not combos:
        raise GeminiUnavailable(RuntimeError("all keys/models are cooling down"),
                                seconds_until_available(keys, models), attempts)
    quota_hits = {}
    tried = 0
    for key, model in combos:
        if tried >= max_attempts:
            break
        if deadline_s is not None and tried > 0 and (time.time() - _t0) >= deadline_s:
            break
        tried += 1
        with _LOCK:
            _STATS["calls"] += 1
        try:
            val = attempt_fn(key, model)
            with _LOCK:
                _STATS["ok"] += 1
                _STATS["last_model"] = model
            return val, keys.index(key), model
        except Exception as exc:
            status, msg = parse_error(exc)
            report_error(key, model, status, msg)
            last = exc
            attempts.append(f"{model}: {status or type(exc).__name__}")
            details.append(f"{model} key#{keys.index(key) + 1}: {status or type(exc).__name__} {str(msg).replace(chr(10), ' ')[:90]}")
            if status == 429:
                quota_hits[model] = quota_hits.get(model, 0) + 1
            logger.warning("Gemini attempt failed %s status=%s", model, status)
            if status == 503:
                time.sleep(0.6)
    raise GeminiUnavailable(last, seconds_until_available(keys, models), attempts, details)


def stats():
    with _LOCK:
        return dict(_STATS)


def sdk_generate(api_keys, contents, models, genai, genai_types=None, grounding=False, max_attempts=30):
    """google-genai SDK call through the pool. Returns (text, key_index, model)."""
    def _one(key, model):
        client = genai.Client(api_key=key)
        kwargs = {"model": model, "contents": contents}
        if grounding and genai_types is not None:
            try:
                kwargs["config"] = genai_types.GenerateContentConfig(
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())])
            except Exception:
                kwargs.pop("config", None)
        try:
            resp = client.models.generate_content(**kwargs)
        except Exception as exc:
            status, _ = parse_error(exc)
            # search/grounding problem (400/403/429/5xx) -> same key/model once without it; do NOT burn another key
            if "config" in kwargs and status != 404:
                resp = client.models.generate_content(model=model, contents=contents)
            else:
                raise
        text = getattr(resp, "text", None)
        if not text:
            raise PoolHTTPError(503, "empty reply")
        return text
    return run(api_keys, models, _one, max_attempts=max_attempts)
