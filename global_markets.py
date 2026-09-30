from app_logging import get_logger
logger = get_logger(__name__)
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# SPEED FIX: this function used to fetch 18 tickers ONE BY ONE (no cache,
# no timeout) on every 30s refresh -> a single slow ticker froze the whole
# page ("ghumta rehta hai"). Now: all tickers are fetched in PARALLEL with a
# hard overall deadline, the result is cached for a short time, and if a
# refresh fails we keep serving the last good table instead of blanking it.
# ---------------------------------------------------------------------
_GM_CACHE = {"ts": 0.0, "df": None}
_GM_LOCK = threading.Lock()
GLOBAL_MARKETS_TTL = 45        # seconds a good result is reused
GLOBAL_MARKETS_DEADLINE = 9    # hard cap for the whole parallel fetch
_GM_LAST_GOOD = {}             # ticker -> (ts, price, change_pct): reused if a ticker is stuck
GM_LAST_GOOD_MAX_AGE = 1800


def _fetch_one_ticker(ticker):
    """Returns (current_price, change_pct) or None. Runs inside a worker thread."""
    try:
        hist = yf.Ticker(ticker).history(period="2d", timeout=8)
    except TypeError:
        hist = yf.Ticker(ticker).history(period="2d")
    if hist is None or hist.empty:
        return None
    current_price = float(hist['Close'].iloc[-1])
    prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
    if prev_close == 0:
        return None
    return current_price, ((current_price - prev_close) / prev_close) * 100

def get_global_market_indices():
    """
    Fetches real-time prices and percentage changes for global stock markets 
    and macro indicators with official country flags/logos.
    """
    indices_data = [
        {"name": "S&P 500 (US)", "ticker": "^GSPC", "code": "us"},
        {"name": "Nasdaq Composite (US)", "ticker": "^IXIC", "code": "us"},
        {"name": "Dow Jones (US)", "ticker": "^DJI", "code": "us"},
        {"name": "Nikkei 225 (Japan)", "ticker": "^N225", "code": "jp"},
        {"name": "Shanghai Composite (China)", "ticker": "000001.SS", "code": "cn"},
        {"name": "Hang Seng (Hong Kong)", "ticker": "^HSI", "code": "hk"},
        {"name": "KOSPI (South Korea)", "ticker": "^KS11", "code": "kr"},
        {"name": "FTSE 100 (UK)", "ticker": "^FTSE", "code": "gb"},
        {"name": "DAX (Germany)", "ticker": "^GDAXI", "code": "de"},
        {"name": "CAC 40 (France)", "ticker": "^FCHI", "code": "fr"},
        {"name": "ASX 200 (Australia)", "ticker": "^AXJO", "code": "au"},
        {"name": "Straits Times (Singapore)", "ticker": "^STI", "code": "sg"},
        {"name": "India VIX", "ticker": "^INDIAVIX", "code": "in"},
        {"name": "Crude Oil (WTI)", "ticker": "CL=F", "code": "oil"},
        {"name": "USD/INR", "ticker": "USDINR=X", "code": "in"},
        # Phase 2 — expanded macro set
        {"name": "US Dollar Index (DXY)", "ticker": "DX-Y.NYB", "code": "us"},
        {"name": "Gold", "ticker": "GC=F", "code": "gold"},
        {"name": "US 10-Year Treasury Yield", "ticker": "^TNX", "code": "us"},
    ]
    
    now = time.time()
    with _GM_LOCK:
        if _GM_CACHE["df"] is not None and now - _GM_CACHE["ts"] < GLOBAL_MARKETS_TTL:
            return _GM_CACHE["df"].copy()

    def _flag(code):
        if code == "oil":
            return "https://img.icons8.com/color/48/oil-industry.png"
        if code == "gold":
            return "https://img.icons8.com/color/48/gold-bars.png"
        return f"https://flagcdn.com/w40/{code}.png"

    executor = ThreadPoolExecutor(max_workers=9)
    futures = {}
    try:
        for item in indices_data:
            futures[item["ticker"]] = executor.submit(_fetch_one_ticker, item["ticker"])
        _futures_wait(list(futures.values()), timeout=GLOBAL_MARKETS_DEADLINE)
    finally:
        # never block on stragglers -- they are simply reported as Neutral
        executor.shutdown(wait=False, cancel_futures=True)

    data = []
    got_any = False
    for item in indices_data:
        name, ticker, code = item["name"], item["ticker"], item["code"]
        flag_url = _flag(code)
        res = None
        fut = futures.get(ticker)
        if fut is not None and fut.done():
            try:
                res = fut.result()
            except Exception:
                logger.exception("Global market ticker fetch failed: %s", ticker)
                res = None
        if res is not None:
            _GM_LAST_GOOD[ticker] = (time.time(), res[0], res[1])
        elif ticker in _GM_LAST_GOOD and time.time() - _GM_LAST_GOOD[ticker][0] < GM_LAST_GOOD_MAX_AGE:
            # ticker stuck/failed this refresh -> reuse its last real value (max 30 min old)
            res = (_GM_LAST_GOOD[ticker][1], _GM_LAST_GOOD[ticker][2])
        if res is not None:
            got_any = True
            current_price, change_pct = res
            data.append({
                "Logo": flag_url,
                "Global Market / Asset": name,
                "Latest Price": round(current_price, 2),
                "Change (%)": round(change_pct, 2),
                "Status": "Bullish 🟢" if change_pct >= 0 else "Bearish 🔴"
            })
        else:
            data.append({
                "Logo": flag_url,
                "Global Market / Asset": name,
                "Latest Price": 0.0,
                "Change (%)": 0.0,
                "Status": "Neutral 🟡"
            })

    df_markets = pd.DataFrame(data)
    with _GM_LOCK:
        if got_any:
            _GM_CACHE["ts"], _GM_CACHE["df"] = time.time(), df_markets.copy()
        elif _GM_CACHE["df"] is not None:
            # total failure this time -> keep showing the last good table
            return _GM_CACHE["df"].copy()
    return df_markets

def get_global_market_summary(df_markets):
    """
    Calculates an automatic Global Market Sentiment Score based on live data.
    """
    if df_markets.empty:
        return "Neutral / Mixed 🟡", 0.0
    
    bullish_count = len(df_markets[df_markets['Status'].str.contains('Bullish')])
    bearish_count = len(df_markets[df_markets['Status'].str.contains('Bearish')])
    avg_change = df_markets['Change (%)'].mean()
    
    if bullish_count >= bearish_count + 4:
        score = "🚀 Strong Bullish"
    elif bullish_count > bearish_count:
        score = "🟢 Mild Bullish"
    elif bearish_count >= bullish_count + 4:
        score = "🚨 Strong Bearish"
    elif bearish_count > bullish_count:
        score = "🔴 Mild Bearish"
    else:
        score = "🟡 Neutral / Sideways"
        
    return score, round(avg_change, 2)


def get_live_vix(df_markets):
    """
    NEW: Pulls the real India VIX value out of the global markets table so
    it can be fed into ml_engine.py as an actual model feature — instead of
    being just a display number that never touches the model (which is
    what was happening before).
    Returns None if VIX couldn't be fetched, so the model can honestly
    drop the feature rather than use a fake constant.
    """
    if df_markets is None or df_markets.empty:
        return None
    try:
        vix_row = df_markets[df_markets["Global Market / Asset"] == "India VIX"]
        if vix_row.empty:
            return None
        value = float(vix_row["Latest Price"].iloc[0])
        return value if value > 0 else None
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return None
