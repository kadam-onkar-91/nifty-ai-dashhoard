import pandas as pd
import requests
import urllib.parse
import yfinance as yf
import streamlit as st

HEAVYWEIGHTS = [
    {"Stock": "HDFC BANK", "Weight (%)": 13.2, "instrument_key": "NSE_EQ|INE040A01034", "symbol": "HDFCBANK", "yf": "HDFCBANK.NS"},
    {"Stock": "RELIANCE", "Weight (%)": 10.5, "instrument_key": "NSE_EQ|INE002A01018", "symbol": "RELIANCE", "yf": "RELIANCE.NS"},
    {"Stock": "ICICI BANK", "Weight (%)": 7.8, "instrument_key": "NSE_EQ|INE090A01021", "symbol": "ICICIBANK", "yf": "ICICIBANK.NS"},
    {"Stock": "INFOSYS", "Weight (%)": 6.1, "instrument_key": "NSE_EQ|INE009A01021", "symbol": "INFY", "yf": "INFY.NS"},
    {"Stock": "TCS", "Weight (%)": 4.5, "instrument_key": "NSE_EQ|INE467B01029", "symbol": "TCS", "yf": "TCS.NS"},
    # Extended to the next best-weighted names for a more representative
    # proxy (previously only 5 stocks ~42% of index weight, now ~65%+).
    # Real Upstox instrument keys added for all of them too, so the live
    # Upstox path (100% accurate, real-time) covers the full 12-stock list
    # when you're logged in -- not just the original 5. Weights are
    # approximate (NSE rebalances the index periodically).
    {"Stock": "BHARTI AIRTEL", "Weight (%)": 4.3, "instrument_key": "NSE_EQ|INE397D01024", "symbol": "BHARTIARTL", "yf": "BHARTIARTL.NS"},
    {"Stock": "ITC", "Weight (%)": 3.9, "instrument_key": "NSE_EQ|INE154A01025", "symbol": "ITC", "yf": "ITC.NS"},
    {"Stock": "LARSEN & TOUBRO", "Weight (%)": 3.6, "instrument_key": "NSE_EQ|INE018A01030", "symbol": "LT", "yf": "LT.NS"},
    {"Stock": "KOTAK BANK", "Weight (%)": 3.0, "instrument_key": "NSE_EQ|INE237A01028", "symbol": "KOTAKBANK", "yf": "KOTAKBANK.NS"},
    {"Stock": "AXIS BANK", "Weight (%)": 3.0, "instrument_key": "NSE_EQ|INE238A01034", "symbol": "AXISBANK", "yf": "AXISBANK.NS"},
    {"Stock": "SBI", "Weight (%)": 2.9, "instrument_key": "NSE_EQ|INE062A01020", "symbol": "SBIN", "yf": "SBIN.NS"},
    {"Stock": "HUL", "Weight (%)": 2.2, "instrument_key": "NSE_EQ|INE030A01027", "symbol": "HINDUNILVR", "yf": "HINDUNILVR.NS"},
]

# Full Nifty 50 constituent list (NSE tickers) used for REAL, broad market
# breadth -- not just the 5-stock proxy above. Pulled from Yahoo Finance in
# one batched call so it stays fast even on a 30-second auto-refresh.
NIFTY50_YF = [
    "HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "M&M.NS", "SUNPHARMA.NS",
    "NTPC.NS", "HCLTECH.NS", "MARUTI.NS", "TITAN.NS", "TATAMOTORS.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "BAJAJFINSV.NS", "ASIANPAINT.NS",
    "NESTLEIND.NS", "ONGC.NS", "ADANIPORTS.NS", "COALINDIA.NS", "WIPRO.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "INDUSINDBK.NS", "GRASIM.NS", "TECHM.NS",
    "HINDALCO.NS", "CIPLA.NS", "SBILIFE.NS", "DRREDDY.NS", "BRITANNIA.NS",
    "EICHERMOT.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "DIVISLAB.NS",
    "TATACONSUM.NS", "HDFCLIFE.NS", "SHRIRAMFIN.NS", "TRENT.NS",
    "ADANIENT.NS", "LTIM.NS", "HEROMOTOCO.NS",
]

# ROOT CAUSE of "DATA UNAVAILABLE": this used to depend ENTIRELY on a live
# Upstox login (token expires daily, needs manual re-login every session).
# Fixed by adding a real Yahoo Finance fallback below, so breadth now works
# even without logging into Upstox each time.
# Set to True only if you need to inspect the raw Upstox response again.
DEBUG_MODE = False


def _find_match(data_dict, symbol):
    for key, val in data_dict.items():
        if symbol.upper() in key.upper():
            return val
    return None


def _fetch_real_heavyweights(access_token):
    """Primary source: live Upstox quotes (most accurate, needs login)."""
    if not access_token:
        return None

    try:
        keys_param = ",".join([h["instrument_key"] for h in HEAVYWEIGHTS])
        encoded_keys = urllib.parse.quote(keys_param, safe=",|")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_keys}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        res = requests.get(url, headers=headers, timeout=6)
        res_json = res.json()

        if DEBUG_MODE:
            with st.expander("DEBUG: Raw Upstox breadth response", expanded=False):
                st.json(res_json)

        if res_json.get("status") != "success":
            return None

        data = res_json.get("data", {})
        rows = []
        for h in HEAVYWEIGHTS:
            match = _find_match(data, h["symbol"])
            if match is None:
                return None

            ltp = match.get("last_price")
            prev_close = match.get("ohlc", {}).get("close")
            if ltp is None or prev_close is None or prev_close == 0:
                return None

            change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)
            rows.append({"Stock": h["Stock"], "Weight (%)": h["Weight (%)"], "Change (%)": change_pct})

        return pd.DataFrame(rows)
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"DEBUG: Upstox breadth fetch exception - {e}")
        return None


@st.cache_data(ttl=25, show_spinner=False)
def _fetch_heavyweights_yfinance():
    """Fallback source: real (slightly delayed) Yahoo Finance quotes.
    This is what actually fixes 'DATA UNAVAILABLE' -- no login required."""
    try:
        tickers = [h["yf"] for h in HEAVYWEIGHTS]
        data = yf.download(tickers=tickers, period="5d", interval="1d",
                            group_by="ticker", progress=False, threads=True)
        rows = []
        for h in HEAVYWEIGHTS:
            t_data = data[h["yf"]] if len(tickers) > 1 else data
            closes = t_data["Close"].dropna()
            if len(closes) < 2:
                return None
            change_pct = round(((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100, 2)
            rows.append({"Stock": h["Stock"], "Weight (%)": h["Weight (%)"], "Change (%)": change_pct})
        return pd.DataFrame(rows)
    except Exception:
        return None


def get_nifty_internal_breadth(access_token=None):
    df_heavyweights = _fetch_real_heavyweights(access_token)
    source = "Upstox (Live Broker Feed)"

    if df_heavyweights is None:
        df_heavyweights = _fetch_heavyweights_yfinance()
        source = "Yahoo Finance (delayed ~15-20 min)"

    if df_heavyweights is None:
        placeholder = pd.DataFrame([
            {"Stock": h["Stock"], "Weight (%)": h["Weight (%)"], "Change (%)": None}
            for h in HEAVYWEIGHTS
        ])
        return placeholder, None, None, None, "DATA UNAVAILABLE (both Upstox and Yahoo Finance failed -- check internet/network access)"

    advances = int((df_heavyweights["Change (%)"] > 0).sum())
    declines = int((df_heavyweights["Change (%)"] <= 0).sum())
    breadth_ratio = round(advances / declines, 2) if declines > 0 else float(advances)

    avg_change = df_heavyweights["Change (%)"].mean()
    if avg_change > 0.3:
        label = "BULLISH HEAVYWEIGHTS"
    elif avg_change < -0.3:
        label = "BEARISH HEAVYWEIGHTS"
    else:
        label = "MIXED HEAVYWEIGHTS"

    breadth_status = f"{label} (5-stock proxy, not full Nifty 50) -- Source: {source}"
    return df_heavyweights, advances, declines, breadth_ratio, breadth_status


@st.cache_data(ttl=90, show_spinner=False)
def get_full_nifty50_breadth():
    """
    REAL market breadth across all 50 Nifty index constituents (not just the
    5-stock proxy above), sourced from Yahoo Finance. This is a genuine
    advance/decline count of the actual index membership.
    Returns: df_full, advances, declines, ratio, status_text
    """
    try:
        data = yf.download(tickers=NIFTY50_YF, period="5d", interval="1d",
                            group_by="ticker", progress=False, threads=True)
        advances, declines, unchanged = 0, 0, 0
        rows = []
        for sym in NIFTY50_YF:
            try:
                closes = data[sym]["Close"].dropna()
                if len(closes) < 2:
                    continue
                chg = ((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100
                if chg > 0.05:
                    advances += 1
                elif chg < -0.05:
                    declines += 1
                else:
                    unchanged += 1
                rows.append({"Symbol": sym.replace(".NS", ""), "Change (%)": round(chg, 2)})
            except Exception:
                continue

        total = advances + declines + unchanged
        if total < 30:
            return None, None, None, None, "DATA UNAVAILABLE (too few Nifty 50 stocks returned data from Yahoo Finance)"

        ratio = round(advances / declines, 2) if declines > 0 else float(advances)
        if advances > declines * 1.5:
            status = f"BROAD-BASED BULLISH ({advances}/{total} advancing)"
        elif declines > advances * 1.5:
            status = f"BROAD-BASED BEARISH ({declines}/{total} declining)"
        else:
            status = f"MIXED / NARROW BREADTH ({advances} up, {declines} down, {unchanged} flat of {total})"

        df_full = pd.DataFrame(rows).sort_values("Change (%)", ascending=False).reset_index(drop=True)
        return df_full, advances, declines, ratio, status
    except Exception as e:
        return None, None, None, None, f"DATA UNAVAILABLE ({e})"
