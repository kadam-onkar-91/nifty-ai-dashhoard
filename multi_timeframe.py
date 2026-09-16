import pandas as pd
import smart_money

"""
PHASE 4 -- MULTI-TIMEFRAME MARKET STRUCTURE (spec section 2)
------------------------------------------------------------
Fetches REAL candles from Upstox at several native timeframes (1D, 1H,
30M, 15M, 5M -- all via market_data.fetch_candles_for_timeframe, which
reuses the already-fixed V3 candle endpoint) and classifies each one's
trend using the same BOS/CHoCH logic already used elsewhere
(smart_money.detect_market_structure), plus a simple EMA20-vs-EMA50
trend read. 4H is not a native Upstox interval, so it's built by
resampling the real 1H candles (standard, honest technique -- not
invented data).

If a timeframe's candles can't be fetched (no token, API failure, not
enough history), that timeframe reports "DATA UNAVAILABLE" and is
excluded from the alignment string rather than guessed at.
"""

TIMEFRAMES = [
    ("1D", "days", "1", 60),
    ("4H", None, None, None),   # built by resampling 1H, see below
    ("1H", "minutes", "60", 15),
    ("30M", "minutes", "30", 8),
    ("15M", "minutes", "15", 5),
    ("5M", "minutes", "5", 3),
]


def _classify_trend(df):
    """Simple, honest trend read: EMA20 vs EMA50 slope + relative position."""
    if df is None or len(df) < 25:
        return None
    close = df['Close']
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean() if len(df) >= 50 else ema20
    last_close = float(close.iloc[-1])
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    e20_prev = float(ema20.iloc[-5]) if len(ema20) > 5 else e20
    slope_up = e20 > e20_prev

    if last_close > e20 > e50 and slope_up:
        return "Bullish"
    if last_close < e20 < e50 and not slope_up:
        return "Bearish"
    if last_close > e50 and not (last_close > e20 > e50):
        return "Bullish pullback"
    if last_close < e50 and not (last_close < e20 < e50):
        return "Bearish pullback"
    return "Neutral/Range"


def _resample_to_4h(df_1h):
    if df_1h is None or df_1h.empty:
        return None
    try:
        agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        if 'Volume' in df_1h.columns:
            agg['Volume'] = 'sum'
        df_4h = df_1h.resample('4h').agg(agg).dropna()
        return df_4h if not df_4h.empty else None
    except Exception:
        return None


def get_multi_timeframe_structure(access_token):
    """
    Returns {'timeframes': {label: {...}}, 'alignment': str, 'status': str}.
    Each timeframe entry has 'trend' and 'structure_event' (BOS/CHoCH/Range)
    if data was available, else 'status': 'DATA_UNAVAILABLE'.
    """
    if not access_token:
        return {"status": "DATA_UNAVAILABLE", "reason": "No Upstox access token -- login required.", "timeframes": {}, "alignment": None}

    results = {}
    df_1h_cache = None

    for label, unit, interval, days_back in TIMEFRAMES:
        if label == "4H":
            df = _resample_to_4h(df_1h_cache)
        else:
            df = _fetch(access_token, unit, interval, days_back)
            if label == "1H":
                df_1h_cache = df

        if df is None or df.empty or len(df) < 25:
            results[label] = {"status": "DATA_UNAVAILABLE"}
            continue

        trend = _classify_trend(df)
        structure_list = smart_money.detect_market_structure(df)
        structure_event = structure_list[0]["Market Event"] if structure_list else "Unknown"

        results[label] = {
            "status": "OK", "trend": trend, "structure_event": structure_event,
            "last_close": round(float(df['Close'].iloc[-1]), 2), "candle_count": len(df),
        }

    available = [(lbl, r) for lbl, r in results.items() if r.get("status") == "OK"]
    if not available:
        return {"status": "DATA_UNAVAILABLE", "reason": "No timeframe returned usable candles.", "timeframes": results, "alignment": None}

    parts = [f"{lbl}={r['trend']}" for lbl, r in available]
    alignment_str = ", ".join(parts)

    htf_labels = [lbl for lbl in ["1D", "4H", "1H"] if results.get(lbl, {}).get("status") == "OK"]
    ltf_labels = [lbl for lbl in ["30M", "15M", "5M"] if results.get(lbl, {}).get("status") == "OK"]
    htf_trends = [results[l]["trend"] for l in htf_labels]
    ltf_trends = [results[l]["trend"] for l in ltf_labels]

    def _bias(trends):
        if not trends:
            return None
        bulls = sum(1 for t in trends if t and "Bullish" in t)
        bears = sum(1 for t in trends if t and "Bearish" in t)
        if bulls > bears:
            return "Bullish"
        if bears > bulls:
            return "Bearish"
        return "Mixed/Neutral"

    htf_bias = _bias(htf_trends)
    ltf_bias = _bias(ltf_trends)

    if htf_bias and ltf_bias:
        if htf_bias == ltf_bias:
            classification = f"HTF {htf_bias}, LTF {ltf_bias} -- fully aligned"
        elif "Neutral" in (htf_bias, ltf_bias) or "Mixed" in (htf_bias, ltf_bias):
            classification = f"HTF {htf_bias}, LTF {ltf_bias} -- one side unclear, treat with extra caution"
        else:
            classification = f"HTF {htf_bias}, LTF {ltf_bias} pullback/counter-trend -- do NOT treat all timeframes equally, HTF bias should dominate"
    else:
        classification = "Not enough timeframe data for an HTF/LTF classification"

    return {
        "status": "OK", "timeframes": results, "alignment": alignment_str,
        "htf_bias": htf_bias, "ltf_bias": ltf_bias, "classification": classification,
    }


def _fetch(access_token, unit, interval, days_back):
    import market_data as _md
    return _md.fetch_candles_for_timeframe(access_token, unit, interval, days_back)
