from app_logging import get_logger
logger = get_logger(__name__)
import pandas as pd
import requests
import urllib.parse
import yfinance as yf
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import time
import indicators  # Custom indicators module

NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# The Upstox instrument master is static intraday. Re-downloading it every
# 30-second dashboard refresh was unnecessary network/latency and made the
# lower dashboard wait behind strategy/volume work. Refresh it periodically
# (and automatically on a new expiry/session if needed).
_FUTURE_KEY_CACHE = {"ts": 0.0, "key": None}
_FUTURE_KEY_CACHE_SECONDS = 900
_MODEL_CACHE = {"fingerprint": None, "model": None, "feature_cols": []}


def _load_upstox_nifty_future_key():
    """Finds the current near-month Nifty 50 index futures contract from
    Upstox's official instrument master. Returns (instrument_key_or_None,
    debug_reason).

    Why this exists: NSE_INDEX|Nifty 50 (the spot index) genuinely has
    ZERO traded volume in Upstox's own data -- an index isn't a
    tradeable instrument, only its derivatives are. So any "Volume
    Profile" or volume-weighted VWAP built purely on the spot index
    candles is, by definition, not really volume-weighted (this app's
    existing code already correctly falls back to a price-based proxy
    when Volume sums to zero -- that fallback was NOT a bug, it was the
    honest thing to do given zero real volume). The standard real fix,
    same as what professional platforms do for index order-flow
    analysis, is to use the near-month INDEX FUTURES contract instead --
    futures are genuinely, continuously traded and carry real volume.
    """
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
        df = pd.read_csv(url, compression="gzip")

        if 'segment' not in df.columns or 'instrument_type' not in df.columns:
            return None, f"Instrument master missing expected columns: {list(df.columns)}"

        fo_df = df[df['segment'].astype(str).str.upper() == 'NSE_FO']
        fo_df = fo_df[fo_df['instrument_type'].astype(str).str.upper() == 'FUTIDX']

        # Match the NIFTY (not BANKNIFTY/FINNIFTY/MIDCPNIFTY) index future.
        # Prefer an exact 'name' column match if present (most reliable);
        # fall back to a tradingsymbol pattern that requires digits right
        # after "NIFTY" (e.g. NIFTY24MARFUT) so it can't accidentally match
        # NIFTYNXT50FUT or similar.
        nifty_rows = pd.DataFrame()
        if 'name' in fo_df.columns:
            candidate = fo_df[fo_df['name'].astype(str).str.upper() == 'NIFTY']
            if not candidate.empty:
                nifty_rows = candidate
        if nifty_rows.empty and 'tradingsymbol' in fo_df.columns:
            candidate = fo_df[fo_df['tradingsymbol'].astype(str).str.match(r'^NIFTY\d', na=False)]
            if not candidate.empty:
                nifty_rows = candidate

        if nifty_rows.empty:
            return None, f"No NIFTY FUTIDX rows matched (found {len(fo_df)} other index futures)"

        if 'expiry' in nifty_rows.columns:
            nifty_rows = nifty_rows.copy()
            nifty_rows['expiry_parsed'] = pd.to_datetime(nifty_rows['expiry'], errors='coerce')
            nifty_rows = nifty_rows.dropna(subset=['expiry_parsed']).sort_values('expiry_parsed')
            upcoming = nifty_rows[nifty_rows['expiry_parsed'] >= pd.Timestamp.now().normalize()]
            nifty_rows = upcoming if not upcoming.empty else nifty_rows

        if nifty_rows.empty or 'instrument_key' not in nifty_rows.columns:
            return None, "Matched NIFTY future rows but couldn't resolve a valid near-month instrument_key"

        return nifty_rows.iloc[0]['instrument_key'], "OK"
    except Exception as e:
        logger.exception("Broad exception caught; fallback path executed")
        return None, f"Futures instrument lookup failed: {type(e).__name__}: {e}"


def get_nifty_futures_instrument_key():
    """Cached public wrapper for the current near-month NIFTY futures key."""
    now = time.time()
    if _FUTURE_KEY_CACHE["key"] and now - _FUTURE_KEY_CACHE["ts"] < _FUTURE_KEY_CACHE_SECONDS:
        return _FUTURE_KEY_CACHE["key"]
    key, _debug = _load_upstox_nifty_future_key()
    if key:
        _FUTURE_KEY_CACHE.update({"ts": now, "key": key})
    return key


def fetch_nifty_futures_volume(access_token):
    """Real traded Volume from the near-month Nifty futures contract, as a
    Timestamp-indexed Series. Returns None if unavailable for any reason
    (no access token, lookup failure, no candles) -- callers should treat
    that as 'no real volume available' and keep using the existing
    price-based proxy, not raise an error."""
    if not access_token:
        return None
    fut_key = get_nifty_futures_instrument_key()
    if not fut_key:
        return None
    fut_df = _fetch_upstox_candles(access_token, instrument_key=fut_key)
    if fut_df is None or fut_df.empty or 'Volume' not in fut_df.columns:
        return None
    return fut_df['Volume']


def fetch_candles_for_timeframe(access_token, unit, interval, days_back=5, instrument_key=NIFTY_INSTRUMENT_KEY):
    """Public wrapper -- PHASE 4's multi_timeframe.py needs real candles at
    several different unit/interval combinations (1D, 1H, 30M, 15M, 5M),
    so it reuses this module's existing, already-fixed V3 candle fetch
    instead of duplicating the historical+intraday stitching logic."""
    return _fetch_upstox_candles(access_token, instrument_key=instrument_key, unit=unit, interval=interval, days_back=days_back)


def _fetch_upstox_ltp(access_token, instrument_key=NIFTY_INSTRUMENT_KEY):
    """Real live LTP from Upstox. Previously this hardcoded the response
    dict key as 'NSE_Index:Nifty 50' (wrong case -- Upstox actually returns
    'NSE_INDEX:Nifty 50'), so the lookup silently KeyError'd and fell
    through to the except-pass every single time, even when the request
    itself succeeded. Now we just take the first (only) value returned,
    the same safe pattern used elsewhere in this app."""
    try:
        encoded_key = urllib.parse.quote(instrument_key, safe='')
        url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={encoded_key}"
        headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
        res = requests.get(url, headers=headers, timeout=6).json()
        if res.get('status') != 'success':
            return None
        data = res.get('data', {})
        match = next(iter(data.values()), None)
        if match is None or match.get('last_price') is None:
            return None
        return float(match['last_price'])
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return None


def _fetch_upstox_candles(access_token, instrument_key=NIFTY_INSTRUMENT_KEY, unit="minutes", interval="5", days_back=5):
    """Real live 5-minute candles from Upstox.

    Previously this called the V2 `/historical-candle/.../5minute/...`
    endpoint -- but the V2 historical-candle API does NOT support a
    '5minute' interval at all (V2 only supports 1minute, 30minute, day,
    week, month), so every request here returned an error and silently
    fell back to Yahoo Finance. Fixed by using the V3 historical-candle
    API (which supports custom minute intervals) for past days, combined
    with the V3 intraday-candle API for today's live candles -- both
    using the correct 'NSE_INDEX' instrument key casing (the old code
    used 'NSE_Index', also wrong)."""
    if not access_token:
        return None
    encoded_key = urllib.parse.quote(instrument_key, safe='')
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    candles = []

    # Past days, up to (not including) today
    try:
        to_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        hist_url = f"https://api.upstox.com/v3/historical-candle/{encoded_key}/{unit}/{interval}/{to_date}/{from_date}"
        res = requests.get(hist_url, headers=headers, timeout=8)
        res_json = res.json()
        if res_json.get('status') == 'success':
            candles.extend(res_json.get('data', {}).get('candles', []))
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        pass

    # Today's live/intraday candles
    try:
        intraday_url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/{unit}/{interval}"
        res = requests.get(intraday_url, headers=headers, timeout=8)
        res_json = res.json()
        if res_json.get('status') == 'success':
            candles.extend(res_json.get('data', {}).get('candles', []))
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        pass

    if not candles:
        return None

    df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.drop_duplicates(subset='Timestamp', inplace=True)
    df.sort_values('Timestamp', inplace=True)
    df.set_index('Timestamp', inplace=True)
    return df


def fetch_live_market_data(access_token):
    df = None
    live_price = None

    # Step 1: Try fetching data from Upstox API (real, live, broker feed)
    if access_token:
        live_price = _fetch_upstox_ltp(access_token)
        df = _fetch_upstox_candles(access_token)

    # Step 2: Fallback to Yahoo Finance if Upstox fails or returns empty
    market_source = "UPSTOX_LIVE" if (df is not None and not df.empty and live_price is not None) else "UNAVAILABLE"

    if df is None or df.empty:
        # Dashboard-only fallback. It is explicitly marked non-tradable so no
        # strategy/AI decision can mistake Yahoo data for the authenticated
        # Upstox feed requested by the trading engine.
        df = yf.download("^NSEI", period="5d", interval="5m", progress=False)
        market_source = "YAHOO_FALLBACK_NON_TRADABLE"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Adj Close' in df.columns:
            df.drop(columns=['Adj Close'], inplace=True)
        if not df.empty:
            live_price = float(df['Close'].iloc[-1])

    if df is None or df.empty:
        return None, None, [], None

    # Step 3: Base columns & Technical Indicators calculation
    # Never fabricate volume. NIFTY spot can legitimately have zero volume;
    # app.py may replace it with the real near-month NIFTY futures volume.
    df['Volume'] = df.get('Volume', 0.0)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0.0)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # ATR Calculation
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean().fillna(df['Close'] * 0.01)

    # Safely calculate additional indicators from indicators.py module
    try:
        df['RSI'] = indicators.calculate_rsi(df, period=14).fillna(50)
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        df['RSI'] = 50

    try:
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = indicators.calculate_macd(df)
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = 0, 0, 0

    try:
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = indicators.calculate_bollinger_bands(df, period=20)
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = df['Close'], df['Close'], df['Close']

    # -------------------------------------------------------------
    # 🚀 ADVANCED MEMORY FEATURES (Giving model past candles context)
    # -------------------------------------------------------------
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['MACD_Lag1'] = df['MACD'].shift(1)
    # -------------------------------------------------------------

    # Clean up missing values WITHOUT using future candles.
    # bfill() is forbidden here because it can copy a future candle's
    # indicator value backwards into an earlier observation (look-ahead bias).
    # Forward-fill is causal; the remaining warm-up NaNs are neutralized only
    # after the causal fill.
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    # Step 4: Safe Live Price Injection using .iloc (No KeyError)
    if live_price is not None and not df.empty:
        try:
            df.iloc[-1, df.columns.get_loc('Close')] = live_price
        except Exception:
            logger.exception("Broad exception caught; fallback path executed")
            pass

    # Step 5: Advanced Machine Learning Setup (XGBoost + Memory)
    # The model is not retrained every 30-second refresh. It only changes when
    # the candle history changes, which keeps the live dashboard responsive.
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    model_df = df.iloc[:-1].copy()
    feature_cols = [
        'EMA_20', 'EMA_50', 'ATR', 'RSI', 'MACD',
        'BB_Upper', 'BB_Lower', 'Close_Lag1', 'Close_Lag2',
        'RSI_Lag1', 'MACD_Lag1'
    ]
    feature_cols = [col for col in feature_cols if col in model_df.columns]
    if not feature_cols:
        model_df['Dummy_Feature'] = model_df['Close'].pct_change().fillna(0)
        feature_cols = ['Dummy_Feature']

    try:
        _fp = (len(model_df), str(model_df.index[0]), str(model_df.index[-1]),
               round(float(model_df['Close'].iloc[-1]), 4))
    except Exception:
        _fp = (len(model_df),)

    if _MODEL_CACHE['fingerprint'] == _fp and _MODEL_CACHE['model'] is not None:
        model = _MODEL_CACHE['model']
        feature_cols = list(_MODEL_CACHE['feature_cols'])
    else:
        X = model_df[feature_cols]
        y = model_df['Target']
        model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='logloss'
        )
        model.fit(X, y)
        _MODEL_CACHE.update({"fingerprint": _fp, "model": model, "feature_cols": list(feature_cols)})
    # Persist source metadata on the dataframe without changing the public
    # four-value return contract used throughout the dashboard.
    df.attrs['market_source'] = market_source
    df.attrs['live_upstox'] = market_source == 'UPSTOX_LIVE'
    df.attrs['live_price_source'] = 'UPSTOX_LTP' if market_source == 'UPSTOX_LIVE' else market_source
    return df, model, feature_cols, live_price


_DAILY_CACHE = {"ts": 0.0, "df": None}
DAILY_CACHE_SECONDS = 1800   # daily candles barely change intraday; refresh every 30 min (saves API quota)


def fetch_daily_candles(access_token=None, days_back=180):
    """Daily NIFTY candles (~6 months) for REAL support/resistance:
    previous-day / previous-week / previous-month H/L and daily swing points.
    The 5-minute frame only holds ~5 days, which is not enough for those.

    Tries Upstox first, then Yahoo (^NSEI). Returns a DataFrame indexed by date
    with Open/High/Low/Close, or None if nothing is available -- never fabricated.
    Cached for DAILY_CACHE_SECONDS."""
    import time as _t
    now = _t.time()
    if _DAILY_CACHE["df"] is not None and now - _DAILY_CACHE["ts"] < DAILY_CACHE_SECONDS:
        return _DAILY_CACHE["df"]
    df = None
    if access_token:
        try:
            df = _fetch_upstox_candles(access_token, unit="days", interval="1", days_back=days_back)
        except Exception:
            logger.exception("Upstox daily candles failed")
            df = None
        if df is not None and not df.empty:
            try:
                df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df = df[~df.index.duplicated(keep="last")]
            except Exception:
                logger.exception("Upstox daily candle cleanup failed")
                df = None
    if df is None or df.empty or len(df) < 20:
        try:
            y = yf.download("^NSEI", period="6mo", interval="1d", progress=False)
            if isinstance(y.columns, pd.MultiIndex):
                y.columns = y.columns.get_level_values(0)
            y = y[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
            y.index = pd.to_datetime(y.index).tz_localize(None).normalize()
            df = y if not y.empty else None
        except Exception:
            logger.exception("Yahoo daily candles failed")
            df = None
    if df is not None and not df.empty:
        _DAILY_CACHE["ts"] = now
        _DAILY_CACHE["df"] = df
    return df
