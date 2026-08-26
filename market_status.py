from datetime import datetime, time as dtime

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = None

"""
MARKET STATE + DATA FRESHNESS CHECKER
--------------------------------------
Fixes the exact gap the in-app AI chat flagged: the dashboard had no
concept of "is NSE actually open right now" or "is this data fresh",
so it kept showing old numbers labeled as if they were live, even after
market close / weekends / when the feed silently stopped updating.

This module is standalone and additive -- it only reads a timestamp and
tells you the truth about it. It doesn't touch any fetch/scoring logic.
"""

NSE_OPEN = dtime(9, 15)
NSE_CLOSE = dtime(15, 30)
STALE_THRESHOLD_MINUTES = 15  # while market is OPEN, data older than this is flagged


def get_ist_now():
    """Current time in IST, timezone-aware if zoneinfo is available."""
    if IST is not None:
        return datetime.now(IST)
    return datetime.now()  # best-effort fallback if zoneinfo unavailable


def get_market_status():
    """
    Returns NSE Nifty 50 cash-market status based on real IST wall-clock
    time: OPEN (Mon-Fri 09:15-15:30 IST), PRE_OPEN, or CLOSED.
    Note: doesn't account for exchange holidays -- add a holiday list
    later if needed, this covers the daily/weekend case which was the
    actual bug reported.
    """
    now = get_ist_now()
    is_weekday = now.weekday() < 5  # Mon=0 ... Sun=6
    t = now.time()

    if is_weekday and NSE_OPEN <= t <= NSE_CLOSE:
        state, label, is_open = "OPEN", "🟢 Market OPEN — Live Data", True
    elif is_weekday and t < NSE_OPEN:
        state, label, is_open = "PRE_OPEN", "🟡 Pre-Market — Not Open Yet (opens 09:15 IST)", False
    elif not is_weekday:
        state, label, is_open = "CLOSED", "🔴 Market CLOSED (Weekend)", False
    else:
        state, label, is_open = "CLOSED", "🔴 Market CLOSED — Showing Last Traded Price", False

    return {"state": state, "label": label, "is_open": is_open, "now_ist": now}


def _to_ist_naive(ts):
    """Normalizes a (possibly tz-aware) timestamp to naive IST wall-clock
    for safe subtraction, regardless of what timezone the data source
    (Upstox / yfinance) returned it in."""
    if ts is None:
        return None
    try:
        if getattr(ts, 'tzinfo', None) is not None and IST is not None:
            return ts.astimezone(IST).replace(tzinfo=None)
        return ts.replace(tzinfo=None) if getattr(ts, 'tzinfo', None) is not None else ts
    except Exception:
        return ts


def check_data_freshness(last_candle_time, market_state, stale_minutes=STALE_THRESHOLD_MINUTES):
    """
    Tells you whether the latest candle timestamp is actually fresh.

    - Market OPEN: data older than `stale_minutes` means the feed is
      likely frozen/cached -- this is the exact "Ghost Data" bug the
      in-app AI flagged (old LTP shown as if it were live).
    - Market CLOSED/PRE_OPEN: an old last-candle timestamp is EXPECTED
      (that's just the final traded price of the last session), so it is
      never flagged stale in that case.
    """
    now_naive = _to_ist_naive(get_ist_now())
    lct_naive = _to_ist_naive(last_candle_time)

    if lct_naive is None or now_naive is None:
        return {"is_stale": False, "age_minutes": None, "note": "No timestamp available to check freshness."}

    try:
        age_minutes = (now_naive - lct_naive).total_seconds() / 60.0
    except Exception:
        return {"is_stale": False, "age_minutes": None, "note": "Could not compute data age."}

    if market_state == "OPEN" and age_minutes > stale_minutes:
        return {
            "is_stale": True, "age_minutes": round(age_minutes, 1),
            "note": (f"⚠️ Feed hasn't updated in {round(age_minutes, 1)} min while the market is open — "
                     f"this looks like frozen/cached data, not a true live tick.")
        }

    return {
        "is_stale": False, "age_minutes": round(age_minutes, 1),
        "note": "Data timestamp looks consistent with the current market state."
    }
