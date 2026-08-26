import requests
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

# Bank Nifty carries ~35% effective weight inside Nifty 50 (via the bank
# stocks that sit in both indices) and is the single biggest driver of
# Nifty option chain sentiment on any given day. Sensex is tracked as a
# broader market cross-check. Both are sourced Upstox-first end-to-end
# (live price AND previous close) with Yahoo Finance only as a last-resort
# fallback if Upstox is unavailable.

INDEX_TARGETS = [
    {"name": "Bank Nifty", "instrument_key": "NSE_INDEX|Nifty Bank", "yf": "^NSEBANK"},
    {"name": "Sensex", "instrument_key": "BSE_INDEX|SENSEX", "yf": "^BSESN"},
]

NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"


@st.cache_data(ttl=14400, show_spinner=False)
def _get_upstox_prev_close(access_token, instrument_key):
    """Previous close sourced from Upstox's own historical-candle API
    (day interval) -- NOT Upstox's live-quote 'ohlc.close' field, which
    was confirmed to mirror the live last_price during market hours
    (causing every index/stock to show a false 0.00% change). Cached for
    hours since a prior session's close can't change intraday."""
    if not access_token:
        return None
    try:
        import urllib.parse
        encoded_key = urllib.parse.quote(instrument_key, safe="|")
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_date}/{from_date}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        res = requests.get(url, headers=headers, timeout=6)
        res_json = res.json()
        if res_json.get("status") != "success":
            return None
        candles = res_json.get("data", {}).get("candles", [])
        if not candles:
            return None
        non_today = [c for c in candles if not str(c[0]).startswith(to_date)]
        chosen = non_today[0] if non_today else candles[0]
        return float(chosen[4])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _get_index_prev_close_yahoo(ticker):
    """Last-resort fallback only -- used if Upstox's own historical-candle
    call fails for some reason (no login, API hiccup, etc)."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        closes = hist['Close'].dropna()
        if len(closes) >= 2:
            return float(closes.iloc[-2])
        elif len(closes) == 1:
            return float(closes.iloc[-1])
        return None
    except Exception:
        return None


def _fetch_upstox_index(access_token, instrument_key, yf_ticker):
    if not access_token:
        return None, None
    try:
        import urllib.parse
        encoded_key = urllib.parse.quote(instrument_key, safe="|")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_key}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        res = requests.get(url, headers=headers, timeout=6)
        res_json = res.json()
        if res_json.get("status") != "success":
            return None, None
        data = res_json.get("data", {})
        if not data:
            return None, None
        # Upstox keys responses by a symbol-ish key, not always the exact
        # instrument_key string -- take the first (only) entry returned.
        match = next(iter(data.values()), None)
        if match is None:
            return None, None
        ltp = match.get("last_price")

        prev_close = _get_upstox_prev_close(access_token, instrument_key)
        prevclose_source = "Upstox"
        if prev_close is None:
            prev_close = _get_index_prev_close_yahoo(yf_ticker)
            prevclose_source = "Yahoo (fallback)"

        if ltp is None or prev_close is None or prev_close == 0:
            return None, None
        change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)
        return (round(float(ltp), 2), change_pct), prevclose_source
    except Exception:
        return None, None


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


def get_nifty_change_pct(access_token, live_price):
    """Nifty 50's own day change % -- tries Upstox's own previous-close
    first (same fix as above), Yahoo only as a last resort."""
    if live_price is None:
        return None
    prev_close = _get_upstox_prev_close(access_token, NIFTY_INSTRUMENT_KEY) if access_token else None
    if prev_close is None:
        prev_close = _get_index_prev_close_yahoo("^NSEI")
    if prev_close is None or prev_close == 0:
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
        value, prevclose_source = _fetch_upstox_index(access_token, target["instrument_key"], target["yf"])
        source = "Upstox (Live)" if prevclose_source is None else f"Upstox (Live, prev-close: {prevclose_source})"
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
