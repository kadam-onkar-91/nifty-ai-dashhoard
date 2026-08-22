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
    
