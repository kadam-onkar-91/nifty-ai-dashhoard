import pandas as pd
import requests
import urllib.parse
import yfinance as yf
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import indicators  # Custom indicators module

NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"


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
        pass

    # Today's live/intraday candles
    try:
        intraday_url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/{unit}/{interval}"
        res = requests.get(intraday_url, headers=headers, timeout=8)
        res_json = res.json()
        if res_json.get('status') == 'success':
            candles.extend(res_json.get('data', {}).get('candles', []))
    except Exception:
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
    if df is None or df.empty:
        df = yf.download("^NSEI", period="5d", interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Adj Close' in df.columns:
            df.drop(columns=['Adj Close'], inplace=True)
        if not df.empty:
            live_price = float(df['Close'].iloc[-1])

    if df is None or df.empty:
        return None, None, [], None

    # Step 3: Base columns & Technical Indicators calculation
    df['Volume'] = df.get('Volume', 100000)
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
        df['RSI'] = 50

    try:
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = indicators.calculate_macd(df)
    except Exception:
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = 0, 0, 0

    try:
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = indicators.calculate_bollinger_bands(df, period=20)
    except Exception:
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = df['Close'], df['Close'], df['Close']

    # -------------------------------------------------------------
    # 🚀 ADVANCED MEMORY FEATURES (Giving model past candles context)
    # -------------------------------------------------------------
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['MACD_Lag1'] = df['MACD'].shift(1)
    # -------------------------------------------------------------

    # Clean up any missing values
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    # Step 4: Safe Live Price Injection using .iloc (No KeyError)
    if live_price is not None and not df.empty:
        try:
            df.iloc[-1, df.columns.get_loc('Close')] = live_price
        except Exception:
            pass

    # Step 5: Advanced Machine Learning Setup (XGBoost + Memory)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    model_df = df.iloc[:-1].copy()

    # Expanded feature list including memory lag columns
    feature_cols = [
        'EMA_20', 'EMA_50', 'ATR', 'RSI', 'MACD', 
        'BB_Upper', 'BB_Lower', 'Close_Lag1', 'Close_Lag2', 
        'RSI_Lag1', 'MACD_Lag1'
    ]
    feature_cols = [col for col in feature_cols if col in model_df.columns]
    
    if not feature_cols:
        model_df['Dummy_Feature'] = model_df['Close'].pct_change().fillna(0)
        feature_cols = ['Dummy_Feature']

    X = model_df[feature_cols]
    y = model_df['Target']

    # Train Advanced XGBoost Model with Regularization & Memory
    model = XGBClassifier(
        n_estimators=150,        # More trees for deeper pattern matching
        max_depth=4,             # Optimized depth to prevent fake noise/overfitting
        learning_rate=0.03,      # Gradual learning for high accuracy
        subsample=0.8,           # Random sampling for robustness
        colsample_bytree=0.8,    # Feature diversity
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X, y)
    
    return df, model, feature_cols, live_price
