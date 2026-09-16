import pandas as pd

def calculate_rsi(df, period=14):
    if df is None or len(df) < period:
        return pd.Series([50] * len(df), index=df.index if df is not None else None)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(df, fast=12, slow=26, signal=9):
    if df is None or len(df) < slow:
        zeros = pd.Series([0] * len(df), index=df.index if df is not None else None)
        return zeros, zeros, zeros
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(df, period=20, std_dev=2):
    if df is None or len(df) < period:
        zeros = pd.Series([0] * len(df), index=df.index if df is not None else None)
        return zeros, zeros, zeros
    middle_band = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    return upper_band.fillna(middle_band), middle_band.fillna(middle_band), lower_band.fillna(middle_band)


# -------------------------------------------------------------------------
# PHASE 1 ADDITIONS — additive only, none of the above functions touched.
# -------------------------------------------------------------------------

def _true_range(df):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def calculate_adx(df, period=14):
    """Average Directional Index + +DI/-DI -- trend STRENGTH (not direction).
    Used by the Market Regime Engine to tell TRENDING apart from RANGE."""
    if df is None or len(df) < period * 2:
        z = pd.Series([0.0] * (len(df) if df is not None else 0), index=df.index if df is not None else None)
        return z, z, z
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, 1e-9)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def calculate_supertrend(df, period=10, multiplier=3.0):
    """Returns (supertrend_line, direction) where direction=1 (bullish,
    price above the line) or -1 (bearish, price below)."""
    if df is None or len(df) < period:
        z = pd.Series([0.0] * (len(df) if df is not None else 0), index=df.index if df is not None else None)
        return z, z
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    hl2 = (df['High'] + df['Low']) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    direction = pd.Series(1, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        if upper_band.iloc[i] < final_upper.iloc[i - 1] or df['Close'].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        if lower_band.iloc[i] > final_lower.iloc[i - 1] or df['Close'].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if df['Close'].iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df['Close'].iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return supertrend, direction


def calculate_stochastic(df, k_period=14, d_period=3):
    if df is None or len(df) < k_period:
        z = pd.Series([50.0] * (len(df) if df is not None else 0), index=df.index if df is not None else None)
        return z, z
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, 1e-9)
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)


def calculate_roc(df, period=12):
    """Rate of Change (%) -- pure momentum, used as a confirmation factor."""
    if df is None or len(df) < period:
        return pd.Series([0.0] * (len(df) if df is not None else 0), index=df.index if df is not None else None)
    return (df['Close'].pct_change(periods=period) * 100).fillna(0)


def calculate_obv(df):
    """On-Balance Volume -- cumulative volume flow direction."""
    if df is None or df.empty or 'Volume' not in df.columns:
        return pd.Series([0.0] * (len(df) if df is not None else 0), index=df.index if df is not None else None)
    direction = pd.Series(0, index=df.index)
    direction[df['Close'] > df['Close'].shift(1)] = 1
    direction[df['Close'] < df['Close'].shift(1)] = -1
    return (direction * df['Volume']).cumsum().fillna(0)


def calculate_mfi(df, period=14):
    """Money Flow Index -- volume-weighted RSI."""
    if df is None or len(df) < period + 1 or 'Volume' not in df.columns:
        return pd.Series([50.0] * (len(df) if df is not None else 0), index=df.index if df is not None else None)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_sum = positive_flow.rolling(window=period).sum()
    neg_sum = negative_flow.rolling(window=period).sum().replace(0, 1e-9)
    mfr = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + mfr))
    return mfi.fillna(50)


def calculate_cci(df, period=20):
    """Commodity Channel Index -- overbought/oversold relative to a moving mean."""
    if df is None or len(df) < period:
        return pd.Series([0.0] * (len(df) if df is not None else 0), index=df.index if df is not None else None)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)
    cci = (typical_price - sma) / (0.015 * mean_dev.replace(0, 1e-9))
    return cci.fillna(0)
