import requests
import yfinance as yf
import streamlit as st

# Bank Nifty carries ~35% effective weight inside Nifty 50 (via the bank
# stocks that sit in both indices) and is the single biggest driver of
# Nifty option chain sentiment on any given day. Sensex is tracked as a
# broader market cross-check. Both follow the same Upstox-first (live,
# most accurate) -> Yahoo Finance fallback (delayed) pattern used
# everywhere else in this app, for consistency and best-available accuracy.

INDEX_TARGETS = [
    {"name": "Bank Nifty", "instrument_key": "NSE_INDEX|Nifty Bank", "yf": "^NSEBANK"},
    {"name": "Sensex", "instrument_key": "BSE_INDEX|SENSEX", "yf": "^BSESN"},
]


def _fetch_upstox_index(access_token, instrument_key):
    if not access_token:
        return None
    try:
        import urllib.parse
        encoded_key = urllib.parse.quote(instrument_key, safe="|")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_key}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        res = requests.get(url, headers=headers, timeout=6)
        res_json = res.json()
        if res_json.get("status") != "success":
            return None
        data = res_json.get("data", {})
        if not data:
            return None
        # Upstox keys responses by a symbol-ish key, not always the exact
        # instrument_key string -- take the first (only) entry returned.
        match = next(iter(data.values()), None)
        if match is None:
            return None
        ltp = match.get("last_price")
        prev_close = match.get("ohlc", {}).get("close")
        if ltp is None or prev_close is None or prev_close == 0:
            return None
        change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)
        return round(float(ltp), 2), change_pct
    except Exception:
        return None


@st.cache_data(ttl=25, show_spinner=False)
def _fetch_yfinance_index(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if hist.empty or len(hist) < 1:
            return None
        current = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
        if prev == 0:
            return None
        change_pct = round(((current - prev) / prev) * 100, 2)
        return round(current, 2), change_pct
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def _get_nifty_prev_close():
    try:
        hist = yf.Ticker("^NSEI").history(period="5d")
        if hist.empty or len(hist) < 2:
            return None
        return float(hist['Close'].iloc[-2])
    except Exception:
        return None


def get_nifty_change_pct(live_price):
    """Approximate Nifty 50 day change % using live_price vs the last
    completed session's close (from Yahoo Finance), used only for the
    Bank Nifty divergence comparison below."""
    prev_close = _get_nifty_prev_close()
    if prev_close is None or prev_close == 0 or live_price is None:
        return None
    return round(((live_price - prev_close) / prev_close) * 100, 2)


def get_bank_nifty_sensex(access_token=None, nifty_change_pct=None):
    """
    Returns a list of dicts: [{name, price, change_pct, source}, ...] for
    Bank Nifty and Sensex, plus a correlation note vs Nifty 50's own
    change (passed in as nifty_change_pct) since a Bank Nifty vs Nifty
    divergence is a well-known real-world fakeout warning signal.
    """
    results = []
    for target in INDEX_TARGETS:
        value = _fetch_upstox_index(access_token, target["instrument_key"])
        source = "Upstox (Live)"
        if value is None:
            value = _fetch_yfinance_index(target["yf"])
            source = "Yahoo Finance (delayed)"
        if value is None:
            results.append({"name": target["name"], "price": None, "change_pct": None, "source": "Unavailable"})
        else:
            price, change_pct = value
            results.append({"name": target["name"], "price": price, "change_pct": change_pct, "source": source})

    correlation_note = None
    bn = next((r for r in results if r["name"] == "Bank Nifty"), None)
    if bn and bn["change_pct"] is not None and nifty_change_pct is not None:
        diff = bn["change_pct"] - nifty_change_pct
        if abs(diff) < 0.15:
            correlation_note = "CONFIRMED — Bank Nifty and Nifty 50 are moving together (healthy participation)."
        elif diff <= -0.15 and nifty_change_pct > 0:
            correlation_note = "DIVERGENCE WARNING — Nifty is up but Bank Nifty is lagging/negative. Classic fake-breakout risk."
        elif diff >= 0.15 and nifty_change_pct < 0:
            correlation_note = "DIVERGENCE WARNING — Nifty is down but Bank Nifty is holding up. Possible fake-breakdown risk."
        else:
            correlation_note = f"Mild divergence ({diff:+.2f} pts gap) — not a strong confirmation either way."

    return results, correlation_note
