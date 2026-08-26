import pandas as pd

def detect_smc_zones(df):
    """
    Calculates Fair Value Gaps, Order Blocks, and Liquidity Sweeps with clean formatting.
    """
    if df is None or len(df) < 5:
        return [], [], []

    fvgs = []
    obs = []
    sweeps = []

    # 1. Fair Value Gaps (FVG)
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i-2]:
            fvgs.append({
                "Type": "Bullish FVG 🟢", 
                "Price": f"{float(df['Low'].iloc[i]):,.2f}"
            })
        elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
            fvgs.append({
                "Type": "Bearish FVG 🔴", 
                "Price": f"{float(df['High'].iloc[i]):,.2f}"
            })

    # 2. Liquidity Sweeps
    for i in range(1, len(df)):
        if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Close'].iloc[i] > df['Low'].iloc[i-1]:
            sweeps.append({
                "Type": "Bullish Sweep 🟢", 
                "Price": f"{float(df['Low'].iloc[i]):,.2f}"
            })
        elif df['High'].iloc[i] > df['High'].iloc[i-1] and df['Close'].iloc[i] < df['High'].iloc[i-1]:
            sweeps.append({
                "Type": "Bearish Sweep 🔴", 
                "Price": f"{float(df['High'].iloc[i]):,.2f}"
            })

    # 3. Order Blocks
    df_copy = df.copy()
    df_copy['Body'] = abs(df_copy['Close'] - df_copy['Open'])
    avg_body = df_copy['Body'].rolling(20).mean().iloc[-1]
    
    for i in range(1, len(df_copy)):
        if df_copy['Body'].iloc[i] > (1.5 * avg_body):
            if df_copy['Close'].iloc[i] > df_copy['Open'].iloc[i]:
                obs.append({
                    "Type": "Bullish OB 🟢", 
                    "Price": f"{float(df_copy['Low'].iloc[i]):,.2f}"
                })
            else:
                obs.append({
                    "Type": "Bearish OB 🔴", 
                    "Price": f"{float(df_copy['High'].iloc[i]):,.2f}"
                })

    return fvgs[-3:], obs[-3:], sweeps[-3:]

def detect_market_structure(df):
    """
    Detects Break of Structure (BOS) and Change of Character (CHoCH).
    """
    structures = []
    if df is None or len(df) < 2:
        return [{"Market Event": "Waiting for live data 🟡", "Trigger Level": "0.00"}]
    
    lookback = min(10, len(df) - 1)
    recent_high = df['High'].iloc[-lookback:-1].max() if lookback > 0 else df['High'].iloc[-1]
    recent_low = df['Low'].iloc[-lookback:-1].min() if lookback > 0 else df['Low'].iloc[-1]
    
    current_close = float(df['Close'].iloc[-1])
    current_high = float(df['High'].iloc[-1])
    current_low = float(df['Low'].iloc[-1])
    
    if current_high > recent_high:
        structures.append({
            "Market Event": "Break of Structure (BOS - Bullish Continuation) 🟢", 
            "Trigger Level": f"{float(recent_high):,.2f}"
        })
    elif current_low < recent_low:
        structures.append({
            "Market Event": "Change of Character (CHoCH - Bearish Reversal) 🔴", 
            "Trigger Level": f"{float(recent_low):,.2f}"
        })
    else:
        structures.append({
            "Market Event": "Consolidation / Range Bound 🟡", 
            "Trigger Level": f"{current_close:,.2f}"
        })
        
    return structures
