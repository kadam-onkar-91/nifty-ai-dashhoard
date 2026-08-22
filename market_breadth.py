import pandas as pd
import requests
import urllib.parse

# Nifty 50 heavyweight stocks with approximate index weights.
# NOTE: Upstox's response dict is typically keyed by "EXCHANGE:TRADINGSYMBOL"
# (e.g. "NSE_EQ:RELIANCE"), not by instrument_key. Verify this against
# Upstox's current API docs — response key formats have changed before.
HEAVYWEIGHTS = [
    {"Stock": "RELIANCE", "Weight (%)": 10.5, "instrument_key": "NSE_EQ|INE002A01018", "symbol": "RELIANCE"},
    {"Stock": "HDFC BANK", "Weight (%)": 13.2, "instrument_key": "NSE_EQ|INE040A01034", "symbol": "HDFCBANK"},
    {"Stock": "ICICI BANK", "Weight (%)": 7.8, "instrument_key": "NSE_EQ|INE090A01021", "symbol": "ICICIBANK"},
    {"Stock": "INFOSYS", "Weight (%)": 6.1, "instrument_key": "NSE_EQ|INE009A01021", "symbol": "INFY"},
    {"Stock": "TCS", "Weight (%)": 4.5, "instrument_key": "NSE_EQ|INE467B01029", "symbol": "TCS"},
]


def _find_match(data_dict, symbol):
    """Tries a few reasonable key patterns since exact Upstox key format can vary."""
    for key, val in data_dict.items():
        if symbol.upper() in key.upper():
            return val
    return None


def _fetch_real_heavyweights(access_token):
    """
    Fetches REAL LTP + previous close (for Change%) from Upstox's quotes
    endpoint. Returns None on ANY failure or partial match — we do not mix
    real data for some stocks with fabricated data for others.
    """
    if not access_token:
        return None

    try:
        keys_param = ",".join([h["instrument_key"] for h in HEAVYWEIGHTS])
        encoded_keys = urllib.parse.quote(keys_param, safe=",|")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_keys}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        res = requests.get(url, headers=headers, timeout=6)
        res_json = res.json()

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
    except Exception:
        return None


def get_nifty_internal_breadth(access_token=None):
    """
    Reports real heavyweight movement as a breadth PROXY (5 stocks, not the
    full 50 — being upfront about that limitation matters). If live data
    can't be fetched or fully matched, this clearly reports
    'DATA UNAVAILABLE' rather than generating random numbers.
    """
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
