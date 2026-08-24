import pandas as pd
import requests
import urllib.parse
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

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


@st.cache_data(ttl=14400, show_spinner=False)
def _get_upstox_prev_close_map(access_token, symbol_key_pairs):
    """
    Previous-close reference sourced from UPSTOX's OWN historical-candle API
    (day interval) -- not Yahoo Finance. This means the whole calculation
    (live price AND previous close) comes from Upstox for every symbol this
    succeeds for. Requires one API call per symbol, so it's cached for 4
    hours (previous close never changes intraday, so this is safe) to avoid
    hammering the API on every 30-second auto-refresh.
    symbol_key_pairs: tuple of (symbol, instrument_key) tuples (hashable for caching).
    """
    if not access_token:
        return {}
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    result = {}
    for symbol, instrument_key in symbol_key_pairs:
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe="|")
            url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_date}/{from_date}"
            res = requests.get(url, headers=headers, timeout=5)
            res_json = res.json()
            if res_json.get("status") != "success":
                continue
            candles = res_json.get("data", {}).get("candles", [])
            if not candles:
                continue
            # Upstox returns candles newest-first; each candle is
            # [timestamp, open, high, low, close, volume, oi]. Skip today's
            # candle if present (it's still forming) and take the most
            # recent COMPLETE prior session's close.
            non_today = [c for c in candles if not str(c[0]).startswith(to_date)]
            chosen = non_today[0] if non_today else candles[0]
            result[symbol] = float(chosen[4])
        except Exception:
            continue
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def _get_prev_close_reference_map():
    """
    Upstox's live quote 'ohlc.close' field turned out to mirror the LIVE
    last_price during market hours instead of the prior session's closing
    price (confirmed: every single stock showed exactly 0.00% change,
    which is not real market behaviour). Rather than guess at Upstox's
    field semantics again, we use a definitively unambiguous reference:
    yesterday's actual closing print, sourced once per hour from Yahoo
    Finance's daily bar (this is just a static prior-day print, not a
    live/delayed feed -- every data vendor agrees on it). Upstox's LIVE
    last_price is still used as the CURRENT price everywhere; only the
    "previous close" denominator comes from this reference.
    """
    try:
        all_tickers = list(set(NIFTY50_YF))
        data = yf.download(tickers=all_tickers, period="5d", interval="1d",
                            group_by="ticker", progress=False, threads=True)
        ref = {}
        for t in all_tickers:
            try:
                closes = data[t]["Close"].dropna()
                if len(closes) >= 2:
                    ref[t.replace(".NS", "")] = float(closes.iloc[-2])
                elif len(closes) == 1:
                    ref[t.replace(".NS", "")] = float(closes.iloc[-1])
            except Exception:
                continue
        return ref
    except Exception:
        return {}


def _fetch_real_heavyweights(access_token):
    """Primary source: live Upstox quotes. Returns (df_or_None, debug_reason)
    -- the reason is ALWAYS captured, even on success, so the caller (and the
    UI) can show exactly why a fallback happened instead of failing silently."""
    if not access_token:
        return None, "No Upstox access_token in session (not logged in)."

    try:
        keys_param = ",".join([h["instrument_key"] for h in HEAVYWEIGHTS])
        encoded_keys = urllib.parse.quote(keys_param, safe=",|")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_keys}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}
        res = requests.get(url, headers=headers, timeout=6)
        res_json = res.json()

        if res_json.get("status") != "success":
            return None, f"Upstox API returned non-success status (HTTP {res.status_code}): {res_json}"

        data = res_json.get("data", {})
        if not data:
            return None, f"Upstox API returned success but an EMPTY data payload (raw: {res_json})"

        symbol_key_pairs = tuple((h["symbol"], h["instrument_key"]) for h in HEAVYWEIGHTS)
        upstox_prev_close = _get_upstox_prev_close_map(access_token, symbol_key_pairs)
        yahoo_prev_close = _get_prev_close_reference_map()

        rows = []
        missing = []
        used_yahoo_prevclose_for = []
        for h in HEAVYWEIGHTS:
            match = _find_match(data, h["symbol"])
            if match is None:
                missing.append(h)
                continue
            ltp = match.get("last_price")
            prev_close = upstox_prev_close.get(h["symbol"])
            if prev_close is None:
                prev_close = yahoo_prev_close.get(h["symbol"])
                if prev_close is not None:
                    used_yahoo_prevclose_for.append(h["symbol"])
            if ltp is None or prev_close is None or prev_close == 0:
                missing.append(h)
                continue
            change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)
            rows.append({"Stock": h["Stock"], "Weight (%)": h["Weight (%)"], "Change (%)": change_pct})

        topped_up_note = ""
        if missing:
            yf_fallback = _fetch_heavyweights_yfinance()
            if yf_fallback is not None:
                yf_map = dict(zip(yf_fallback["Stock"], yf_fallback["Change (%)"]))
                still_missing = []
                for h in missing:
                    if h["Stock"] in yf_map:
                        rows.append({"Stock": h["Stock"], "Weight (%)": h["Weight (%)"], "Change (%)": yf_map[h["Stock"]]})
                    else:
                        still_missing.append(h["symbol"])
                if still_missing:
                    return None, f"Upstox missing {[h['symbol'] for h in missing]}, and Yahoo top-up also failed for: {still_missing}"
                topped_up_note = f" (+{len(missing)} topped up from Yahoo Finance: {[h['symbol'] for h in missing]})"
            else:
                return None, f"Upstox response missing/incomplete for: {[h['symbol'] for h in missing]}, and Yahoo Finance top-up also failed."

        prevclose_note = f" (prev-close via Yahoo for: {used_yahoo_prevclose_for})" if used_yahoo_prevclose_for else " (prev-close 100% via Upstox)"
        return pd.DataFrame(rows), f"OK (Upstox live){topped_up_note}{prevclose_note}"
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
        symbol_key_pairs = tuple(found.items())
        upstox_prev_close = _get_upstox_prev_close_map(access_token, symbol_key_pairs)
        yahoo_prev_close = _get_prev_close_reference_map()
        yahoo_used_count = 0

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
                prev_close = upstox_prev_close.get(symbol)
                if prev_close is None:
                    prev_close = yahoo_prev_close.get(symbol)
                    if prev_close is not None:
                        yahoo_used_count += 1
                if symbol is None or ltp is None or prev_close is None or prev_close == 0:
                    continue
                chg = round(((ltp - prev_close) / prev_close) * 100, 2)
                rows.append({"Symbol": symbol, "Change (%)": chg})

        if len(rows) < 40:
            return None, f"Upstox quotes returned only {len(rows)}/50 usable rows (rest had missing ltp/prev_close)."
        prevclose_note = f", prev-close via Yahoo for {yahoo_used_count} symbol(s)" if yahoo_used_count else ", prev-close 100% via Upstox"
        return rows, f"OK ({len(rows)}/50 from Upstox live{prevclose_note})"
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
    else:
        # Upstox sometimes doesn't return every single one of the 50 in its
        # response (thin/illiquid instruments, brief API hiccups, etc).
        # Top up any missing ones with Yahoo Finance so the breadth count
        # always accounts for all 50 -- never silently drops stocks.
        have = {r["Symbol"] for r in rows}
        base_symbols = [s.replace(".NS", "") for s in NIFTY50_YF]
        missing = [s for s in base_symbols if s not in have]
        if missing:
            yf_rows = _fetch_full50_yfinance() or []
            yf_map = {r["Symbol"]: r for r in yf_rows}
            topped_up = [sym for sym in missing if sym in yf_map]
            for sym in topped_up:
                rows.append(yf_map[sym])
            if topped_up:
                source = f"Upstox (Live) + Yahoo Finance for {len(topped_up)} symbol(s) Upstox didn't return"

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
    breakdown = f"{advances} up, {declines} down, {unchanged} flat -- {total}/50 stocks fetched"
    if advances > declines * 1.5:
        status = f"BROAD-BASED BULLISH ({breakdown})"
    elif declines > advances * 1.5:
        status = f"BROAD-BASED BEARISH ({breakdown})"
    else:
        status = f"MIXED / NARROW BREADTH ({breakdown})"
    status = f"{status} -- Source: {source}"

    df_full = pd.DataFrame(rows).sort_values("Change (%)", ascending=False).reset_index(drop=True)
    return df_full, advances, declines, ratio, status, upstox_debug
