"""
safe_io.py -- keep ONE slow data source from freezing the whole dashboard.

The dashboard is a single Streamlit fragment that re-runs every 30 s and calls ~20 data sources one after the other.
If any one of them stalls (Yahoo rate-limit, broker API slow, ...) the page sits on "Stop" and nothing renders.

guarded()   : run a call in a worker thread and wait at most `timeout` seconds.  On timeout / error the LAST GOOD value
              (or `default`) is returned immediately, while the call keeps running in the background; when it finishes
              the fresh value is stored, so the next refresh picks it up.  No duplicate jobs are started for one key.
Stepper     : tiny helper that remembers how long each step of a refresh took, so the page can show the slowest steps
              (and, while loading, which step is running right now).
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

logger = logging.getLogger(__name__)

_POOL = ThreadPoolExecutor(max_workers=10, thread_name_prefix="safe-io")
_LOCK = threading.RLock()     # RLock: add_done_callback() runs the callback inline when the job already finished
_STATE = {}          # key -> {"has": bool, "value": any, "ts": float, "fut": Future|None, "last_ms": float, "last_status": str}


def _attach_streamlit_ctx(ctx):
    try:
        if ctx is not None:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            add_script_run_ctx(threading.current_thread(), ctx)
    except Exception:
        pass


def guarded(key, fn, *args, timeout=15.0, ttl=0.0, default=None, **kwargs):
    """Call fn(*args, **kwargs) but never wait longer than `timeout` seconds.  See module docstring."""
    now = time.time()
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None
    with _LOCK:
        st = _STATE.setdefault(key, {"has": False, "value": None, "ts": 0.0, "fut": None, "last_ms": 0.0, "last_status": "new"})
        if ttl and st["has"] and now - st["ts"] < ttl:
            return st["value"]
        fut = st["fut"]
        if fut is None or fut.done():
            t0 = time.time()

            def _job():
                _attach_streamlit_ctx(ctx)
                return fn(*args, **kwargs)

            fut = _POOL.submit(_job)
            st["fut"] = fut

            def _done(f, _t0=t0, _key=key):
                ms = (time.time() - _t0) * 1000.0
                with _LOCK:
                    s = _STATE[_key]
                    s["last_ms"] = ms
                    try:
                        s["value"] = f.result()
                        s["has"] = True
                        s["ts"] = time.time()
                        s["last_status"] = "ok"
                    except Exception as exc:                 # keep the old good value
                        s["last_status"] = f"error: {type(exc).__name__}"
                        logger.warning("safe_io %s failed: %s", _key, exc)
            fut.add_done_callback(_done)
    try:
        fut.result(timeout=timeout)
    except _FutTimeout:
        with _LOCK:
            _STATE[key]["last_status"] = f"timeout>{timeout:.0f}s (running in background)"
        logger.warning("safe_io %s timed out after %.0fs; using %s", key, timeout,
                       "last good value" if _STATE[key]["has"] else "default")
    except Exception:
        pass                                                   # the done-callback already recorded it
    with _LOCK:
        st = _STATE[key]
        return st["value"] if st["has"] else default


def status(key):
    with _LOCK:
        s = _STATE.get(key)
        return None if s is None else {"last_ms": s["last_ms"], "status": s["last_status"], "has_value": s["has"]}


class Stepper:
    """Records step durations of one refresh and renders them.  Usage:
         sp = Stepper(placeholder); sp.step("Option chain"); ...; sp.step("Strategy engine"); ...; sp.done()
    """

    def __init__(self, placeholder=None):
        self.ph = placeholder
        self.t0 = time.time()
        self.last_t = self.t0
        self.cur = None
        self.rows = []

    def step(self, label):
        now = time.time()
        if self.cur is not None:
            self.rows.append((self.cur, now - self.last_t))
        self.cur, self.last_t = label, now
        if self.ph is not None:
            try:
                self.ph.caption(f"⏳ Loading: **{label}** ... (total {now - self.t0:.0f}s so far)")
            except Exception:
                pass

    def done(self):
        now = time.time()
        if self.cur is not None:
            self.rows.append((self.cur, now - self.last_t))
            self.cur = None
        total = now - self.t0
        slow = sorted(self.rows, key=lambda r: -r[1])[:3]
        txt = f"✅ Page {total:.1f}s me load hua"
        if slow and total >= 8:
            txt += " | slowest: " + ", ".join(f"{n} {d:.1f}s" for n, d in slow)
        if self.ph is not None:
            try:
                self.ph.caption(txt)
            except Exception:
                pass
        return total, self.rows
