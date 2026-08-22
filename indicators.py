import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """
    Computes technical indicators such as RSI, EMAs, VWAP, MACD, ATR, and Bollinger Bands.
    """
    if df is None or df.empty:
        return df
        
    df = df.copy()
    
    if 'Close' not in df.columns:
        return df
        
    close = df['Close']
    
    # Exponential Moving Averages
    df['EMA_9'] = close.ewm(span=9, adjust=False).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    
    # Relative Strength Index (RSI 14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / max(1e-9, loss)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Average True Range (ATR)
    if 'High' in df.columns and 'Low' in df.columns:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - close.shift())
        low_close = np.abs(df['Low'] - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
    else:
        df['ATR'] = 15.0

    # Bollinger Bands (20, 2)
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    
    # Volume Weighted Average Price (VWAP)
    if 'Volume' in df.columns and 'High' in df.columns and 'Low' in df.columns:
        typical_price = (df['High'] + df['Low'] + close) / 3
        df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    else:
        df['VWAP'] = close.rolling(window=14).mean()
        
    return df
