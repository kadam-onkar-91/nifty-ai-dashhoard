"""tick_guard.py -- stop-loss / target are checked on the LIVE PRICE every ~2 seconds.

WHY
  The dashboard's heavy refresh (data + 40 strategies + AI engine) takes 12-20 s and runs every 30 s, and the old code only looked at
  stop-loss / target inside it (plus the completed 5-min candles).  So a stop was recorded 30-45 s -- sometimes a whole candle -- AFTER
  price touched it.  This small background thread does ONE cheap thing: fetch the Upstox LTP and compare it with the open trade's
  SL / target.  It does not run any strategy, does not open trades and does not touch Streamlit.

HOW
  ensure_started() is idempotent (one daemon thread per server process; app.py calls it on every run).  The thread only works while
  NSE is OPEN and a broker token exists.  Every error is swallowed and logged -- it can never break the dashboard.
"""
from __future__ import annotations

import threading
import time

from app_logging import get_logger

logger = get_logger(__name__)

TICK_SECONDS = 2.0          # LTP poll interval while the market is open
SHADOW_EVERY_S = 10.0       # paper (shadow) trades are followed a little slower
_STATE = {"thread": None, "last_tick": 0.0, "last_price": None, "closed_by_tick": 0, "errors": 0}
_LOCK = threading.Lock()


def status():
    return dict(_STATE, alive=bool(_STATE["thread"] and _STATE["thread"].is_alive()))


def _loop():
    import market_data
    import market_status
    import shadow_trades
    import trade_learning
    import upstox_auth

    last_shadow = 0.0
    while True:
        try:
            time.sleep(TICK_SECONDS)
            if market_status.get_market_status().get("state") != "OPEN":
                time.sleep(20)
                continue
            token = upstox_auth._stored_token()
            if not token:
                continue
            ltp = market_data._fetch_upstox_ltp(token)
            if not ltp:
                continue
            _STATE["last_tick"] = time.time()
            _STATE["last_price"] = float(ltp)
            trade_learning.resolve_tick(float(ltp))
            if time.time() - last_shadow >= SHADOW_EVERY_S:
                last_shadow = time.time()
                try:
                    shadow_trades.resolve(float(ltp), None)
                except Exception:
                    logger.exception("tick shadow resolve failed (ignored)")
        except Exception:
            _STATE["errors"] += 1
            logger.exception("tick_guard loop error (ignored)")
            time.sleep(5)


def ensure_started():
    """Start the guard thread once per process.  Safe to call on every Streamlit run."""
    with _LOCK:
        t = _STATE["thread"]
        if t is not None and t.is_alive():
            return False
        t = threading.Thread(target=_loop, name="tick-guard", daemon=True)
        _STATE["thread"] = t
        t.start()
        return True
