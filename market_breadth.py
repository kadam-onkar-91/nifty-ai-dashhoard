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
    {"Stock": "BHARTI AIRTEL", "Weight (%)": 4.3, "instrument_key": "NSE_EQ|INE397D01024", "symbol": "BHARTIARTL", "yf": "BHARTIARTL.NS"},
    {"Stock": "ITC", "Weight (%)": 3.9, "instrument_key": "NSE_EQ|INE154A01025", "symbol": "ITC", "yf": "ITC.NS"},
    {"Stock": "LARSEN & TOUBRO", "Weight (%)": 3.6, "instrument_key": "NSE_EQ|INE018A01030", "symbol": "LT", "yf": "LT.NS"},
    {"Stock": "KOTAK BANK", "Weight (%)": 3.0, "instrument_key": "NSE_EQ|INE237A01028", "symbol": "KOTAKBANK", "yf": "KOTAKBANK.NS"},
    {"Stock": "AXIS BANK", "Weight (%)": 3.0, "instrument_key": "NSE_EQ|INE238A01034", "symbol": "AXISBANK", "yf": "AXISBANK.NS"},
    {"Stock": "SBI", "Weight (%)": 2.9, "instrument_key": "NSE_EQ|INE062A01020", "symbol": "SBIN", "yf": "SBIN.NS"},
    {"Stock": "HUL", "Weight (%)": 2.2, "instrument_key": "NSE_EQ|INE030A01027", "symbol": "HINDUNILVR", "yf": "HINDUNILVR.NS"},
]

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


def _find_match(data_dict, symbol):
    for key, val in data_dict.items():
        if symbol.upper() in key.upper():
            return val
    return None


def _fetch_real_heavyweights(access_token):
    """Primary source: live Upstox quotes. Returns (df_or_None, debug_reason)
    -- the reason is ALWAYS captured, even on success, so the caller (and the
    UI) can show exactly why a fallback happened instead of failing silently.

    NOTE: instrument_key is now resolved live via Upstox's own instrument
    master (_load_upstox_nse_equity_map, shared with the full-50 breadth
    fetch below) instead of the hand-typed ISINs in HEAVYWEIGHTS. The
    hand-typed keys were exactly the risk flagged in that function's own
    docstring -- if even one ISIN was stale/mistyped, that stock's quote
    request would just silently fail to match and this whole 12-stock
    table would fall back to Yahoo Finance, even while the full-50 table
    (which never trusted hand-typed keys) succeeded on Upstox Live right
    next to it."""
    if not access_token:
        return None, "No Upstox access_token in session (not logged in)."

    symbol_map, map_debug = _load_upstox_nse_equity_map()
    if not symbol_map:
        return None, f"Could not load Upstox instrument master to resolve heavyweight symbols: {map_debug}"

    key_to_stock = {}
    missing_syms = []
    for h in HEAVYWEIGHTS:
        ik = symbol_map.get(h["symbol"])
        if not ik:
            missing_syms.append(h["symbol"])
            continue
        key_to_stock[ik] = h

    if len(key_to_stock) < len(HEAVYWEIGHTS) - 1:
        return None, (f"Only {len(key_to_stock)}/{len(HEAVYWEIGHTS)} heavyweight symbols matched in "
                      f"Upstox instrument master. Missing: {missing_syms}")

    try:
        encoded_keys = urllib.parse.quote(",".join(key_to_stock.keys()), safe=",|")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_keys}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        res = requests.get(url, headers=headers, timeout=6)
        res_json = res.json()

        if res_json.get("status") != "success":
            return None, f"Upstox API returned non-success status (HTTP {res.status_code}): {res_json}"

        data = res_json.get("data", {})
        if not data:
            return None, f"Upstox API returned success but an EMPTY data payload (raw: {res_json})"

        rows = []
        missing = []
        for h in key_to_stock.values():
            match = _find_match(data, h["symbol"])
            if match is None:
                missing.append(h["symbol"])
                continue
            ltp = match.get("last_price")
            prev_close = match.get("ohlc", {}).get("close")
            if ltp is None or prev_close is None or prev_close == 0:
                missing.append(f"{h['symbol']} (missing ltp/ohlc)")
                continue
            change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)
            rows.append({"Stock": h["Stock"], "Weight (%)": h["Weight (%)"], "Change (%)": change_pct})

        if len(rows) < len(HEAVYWEIGHTS) - 1:
            return None, f"Upstox response missing/incomplete for: {missing}. Raw data keys: {list(data.keys())}"

        return pd.DataFrame(rows), "OK (Upstox live)"
    except Exception as e:
        return None, f"Upstox request raised an exception: {type(e).__name__}: {e}"


@st.cache_data(ttl=25, show_spinner=False)
def _fetch_heavyweights_yfinance():
    """Fallback source: real (slightly delayed) Yahoo Finance quotes."""
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
    df_heavyweights, upstox_debug = _fetch_real_heavyweights(access_token)
    source = "Upstox (Live Broker Feed)"

    if df_heavyweights is None:
        df_heavyweights = _fetch_heavyweights_yfinance()
        source = "Yahoo Finance (delayed ~15-20 min)"

    if df_heavyweights is None:
        placeholder = pd.DataFrame([
            {"Stock": h["Stock"], "Weight (%)": h["Weight (%)"], "Change (%)": None}
            for h in HEAVYWEIGHTS
        ])
        return placeholder, None, None, None, "DATA UNAVAILABLE (both Upstox and Yahoo Finance failed)", upstox_debug

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

    breadth_status = f"{label} ({len(HEAVYWEIGHTS)}-stock proxy, not full Nifty 50) -- Source: {source}"
    return df_heavyweights, advances, declines, breadth_ratio, breadth_status, upstox_debug


@st.cache_data(ttl=3600, show_spinner=False)
def _load_upstox_nse_equity_map():
    """
    Downloads Upstox's official NSE instrument master file and builds a
    tradingsymbol -> instrument_key map. This avoids hand-typing 50 ISINs
    (risky -- one typo would silently pull the WRONG stock's data). Returns
    (map_or_None, debug_reason).
    """
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
        df = pd.read_csv(url, compression="gzip")
        if 'tradingsymbol' not in df.columns or 'instrument_key' not in df.columns:
            return None, f"Instrument master downloaded but columns unexpected: {list(df.columns)}"

        eq_df = pd.DataFrame()
        matched_via = None

        # Try 1: segment == 'NSE_EQ' (matches the "NSE_EQ|<ISIN>" instrument_key
        # prefix already used everywhere else in this app -- the most reliable
        # signal if the column exists).
        if 'segment' in df.columns:
            candidate = df[df['segment'].astype(str).str.upper() == 'NSE_EQ']
            if not candidate.empty:
                eq_df, matched_via = candidate, "segment == 'NSE_EQ'"

        # Try 2: instrument_type, case-insensitive, either 'EQ' or 'EQUITY'.
        if eq_df.empty and 'instrument_type' in df.columns:
            it = df['instrument_type'].astype(str).str.upper()
            candidate = df[it.isin(['EQ', 'EQUITY'])]
            if not candidate.empty:
                eq_df, matched_via = candidate, "instrument_type in ['EQ','EQUITY'] (case-insensitive)"

        if eq_df.empty:
            sample_cols = [c for c in ['segment', 'instrument_type', 'exchange'] if c in df.columns]
            sample = {c: df[c].astype(str).unique().tolist()[:8] for c in sample_cols}
            return None, (f"Instrument master downloaded ({len(df)} rows, columns={list(df.columns)}) "
                          f"but no equity rows matched any known filter. Sample observed values: {sample}")

        return dict(zip(eq_df['tradingsymbol'], eq_df['instrument_key'])), f"OK ({len(eq_df)} equity symbols loaded via {matched_via})"
    except Exception as e:
        return None, f"Instrument master download/parse failed: {type(e).__name__}: {e}"


def _fetch_full50_upstox(access_token):
    """Batched live Upstox quotes for all 50 Nifty constituents.
    Returns (rows_or_None, debug_reason)."""
    if not access_token:
        return None, "No Upstox access_token in session (not logged in)."

    symbol_map, map_debug = _load_upstox_nse_equity_map()
    if not symbol_map:
        return None, map_debug

    base_symbols = [s.replace(".NS", "") for s in NIFTY50_YF]
    found = {s: symbol_map[s] for s in base_symbols if s in symbol_map}
    missing_syms = [s for s in base_symbols if s not in symbol_map]
    keys = list(found.values())

    if len(keys) < 40:
        return None, f"Only {len(keys)}/50 symbols matched in Upstox instrument master. Missing: {missing_syms}"

    try:
        rows = []
        chunk_size = 25
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i:i + chunk_size]
            encoded_keys = urllib.parse.quote(",".join(chunk), safe=",|")
            url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_keys}"
            headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
            res = requests.get(url, headers=headers, timeout=8)
            res_json = res.json()
            if res_json.get("status") != "success":
                return None, f"Upstox quotes API (chunk {i//chunk_size + 1}) returned non-success (HTTP {res.status_code}): {res_json}"
            data = res_json.get("data", {})
            for match in data.values():
                symbol = match.get("symbol") or match.get("trading_symbol")
                ltp = match.get("last_price")
                prev_close = match.get("ohlc", {}).get("close")
                if symbol is None or ltp is None or prev_close is None or prev_close == 0:
                    continue
                chg = round(((ltp - prev_close) / prev_close) * 100, 2)
                rows.append({"Symbol": symbol, "Change (%)": chg})

        if len(rows) < 40:
            return None, f"Upstox quotes returned only {len(rows)}/50 usable rows (rest had missing ltp/ohlc)."
        return rows, f"OK ({len(rows)}/50 from Upstox live)"
    except Exception as e:
        return None, f"Upstox batched quotes request raised an exception: {type(e).__name__}: {e}"


@st.cache_data(ttl=90, show_spinner=False)
def _fetch_full50_yfinance():
    try:
        data = yf.download(tickers=NIFTY50_YF, period="5d", interval="1d",
                            group_by="ticker", progress=False, threads=True)
        rows = []
        for sym in NIFTY50_YF:
            try:
                closes = data[sym]["Close"].dropna()
                if len(closes) < 2:
                    continue
                chg = ((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100
                rows.append({"Symbol": sym.replace(".NS", ""), "Change (%)": round(chg, 2)})
            except Exception:
                continue
        return rows if len(rows) >= 30 else None
    except Exception:
        return None


def get_full_nifty50_breadth(access_token=None):
    """
    REAL market breadth across all 50 Nifty index constituents. Tries live
    Upstox quotes first (100% accurate, real-time -- needs login), then
    falls back to Yahoo Finance (delayed ~15-20 min).
    Returns: df_full, advances, declines, ratio, status_text, debug_reason
    """
    rows, upstox_debug = _fetch_full50_upstox(access_token)
    source = "Upstox (Live)"

    if rows is None:
        rows = _fetch_full50_yfinance()
        source = "Yahoo Finance (delayed ~15-20 min)"

    if rows is None:
        return None, None, None, None, "DATA UNAVAILABLE (both Upstox and Yahoo Finance failed to return enough Nifty 50 data)", upstox_debug

    advances, declines, unchanged = 0, 0, 0
    for r in rows:
        if r["Change (%)"] > 0.05:
            advances += 1
        elif r["Change (%)"] < -0.05:
            declines += 1
        else:
            unchanged += 1
    total = advances + declines + unchanged

    ratio = round(advances / declines, 2) if declines > 0 else float(advances)
    if advances > declines * 1.5:
        status = f"BROAD-BASED BULLISH ({advances}/{total} advancing)"
    elif declines > advances * 1.5:
        status = f"BROAD-BASED BEARISH ({declines}/{total} declining)"
    else:
        status = f"MIXED / NARROW BREADTH ({advances} up, {declines} down, {unchanged} flat of {total})"
    status = f"{status} -- Source: {source}"

    df_full = pd.DataFrame(rows).sort_values("Change (%)", ascending=False).reset_index(drop=True)
    return df_full, advances, declines, ratio, status, upstox_debug
