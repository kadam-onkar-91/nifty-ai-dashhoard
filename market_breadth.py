import pandas as pd
import requests
import urllib.parse
import streamlit as st

HEAVYWEIGHTS = [
    {"Stock": "RELIANCE", "Weight (%)": 10.5, "instrument_key": "NSE_EQ|INE002A01018", "symbol": "RELIANCE"},
    {"Stock": "HDFC BANK", "Weight (%)": 13.2, "instrument_key": "NSE_EQ|INE040A01034", "symbol": "HDFCBANK"},
    {"Stock": "ICICI BANK", "Weight (%)": 7.8, "instrument_key": "NSE_EQ|INE090A01021", "symbol": "ICICIBANK"},
    {"Stock": "INFOSYS", "Weight (%)": 6.1, "instrument_key": "NSE_EQ|INE009A01021", "symbol": "INFY"},
    {"Stock": "TCS", "Weight (%)": 4.5, "instrument_key": "NSE_EQ|INE467B01029", "symbol": "TCS"},
]

# TEMPORARY DEBUG SWITCH — set to True to see the raw Upstox response on the
# dashboard so we can see the exact key format and fix matching for real.
# Set back to False once breadth data is showing correctly.
DEBUG_MODE = True


def _find_match(data_dict, symbol):
    for key, val in data_dict.items():
        if symbol.upper() in key.upper():
            return val
    return None


def _fetch_real_heavyweights(access_token):
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
            with st.expander("🔧 DEBUG: Raw Upstox breadth response (remove after fixing)", expanded=False):
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
            st.error(f"🔧 DEBUG: Breadth fetch exception — {e}")
        return None


def get_nifty_internal_breadth(access_token=None):
    df_heavyweights = _fetch_real_heavyweights(access_token)

    if df_heavyweights is None:
        placeholder = pd.DataFrame([
            {"Stock": h["Stock"], "Weight (%)": h["Weight (%)"], "Change (%)": None}
            for h in HEAVYWEIGHTS
        ])
        return placeholder, None, None, None, "DATA UNAVAILABLE (Live broker feed required for breadth)"

    advances = int((df_heavyweights["Change (%)"] > 0).sum())
    declines = int((df_heavyweights["Change (%)"] <= 0).sum())
    breadth_ratio = round(advances / declines, 2) if declines > 0 else float(advances)

    avg_change = df_heavyweights["Change (%)"].mean()
    if avg_change > 0.3:
        breadth_status = "BULLISH HEAVYWEIGHTS (5-stock proxy, not full Nifty 50 breadth)"
    elif avg_change < -0.3:
        breadth_status = "BEARISH HEAVYWEIGHTS (5-stock proxy, not full Nifty 50 breadth)"
    else:
        breadth_status = "MIXED HEAVYWEIGHTS (5-stock proxy, not full Nifty 50 breadth)"

    return df_heavyweights, advances, declines, breadth_ratio, breadth_status
    
