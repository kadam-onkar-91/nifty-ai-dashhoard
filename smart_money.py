import pandas as pd


def detect_candlestick_pattern(df):
    """
    Classic candlestick price-action patterns on the most recent candle(s):
    Engulfing, Hammer/Pin Bar, Shooting Star, Doji, Marubozu. This is real
    pattern recognition on actual OHLC shape/geometry, used to confirm (or
    flag a conflict with) a level reaction -- FVG/OB/CPR tell you WHERE a
    reaction might happen, this tells you whether the candles are actually
    SHOWING one forming right now.
    """
    if df is None or len(df) < 3:
        return {"pattern": "NONE", "bias": "NEUTRAL", "strength": 0.0}

    try:
        c0, c1 = df.iloc[-1], df.iloc[-2]
        o0, h0, l0, cl0 = float(c0['Open']), float(c0['High']), float(c0['Low']), float(c0['Close'])
        o1, h1, l1, cl1 = float(c1['Open']), float(c1['High']), float(c1['Low']), float(c1['Close'])

        range0 = (h0 - l0) or 0.0001
        body0 = abs(cl0 - o0)
        upper_wick0 = h0 - max(o0, cl0)
        lower_wick0 = min(o0, cl0) - l0
        body_pct0 = body0 / range0
        body1 = abs(cl1 - o1)

        pattern, bias, strength = "None (no clear pattern on latest candle)", "NEUTRAL", 0.0

        # 1) Engulfing -- current candle's body fully swallows the prior one
        if cl0 > o0 and cl1 < o1 and cl0 >= o1 and o0 <= cl1 and body0 > body1 * 1.1:
            pattern, bias, strength = "Bullish Engulfing", "BULLISH", 0.8
        elif cl0 < o0 and cl1 > o1 and cl0 <= o1 and o0 >= cl1 and body0 > body1 * 1.1:
            pattern, bias, strength = "Bearish Engulfing", "BEARISH", 0.8

        # 2) Hammer/Bullish Pin Bar vs Shooting Star/Bearish Pin Bar --
        # small body, one long wick showing clear rejection of that side
        elif body_pct0 < 0.35 and lower_wick0 > body0 * 2 and upper_wick0 < body0 * 0.5:
            pattern, bias, strength = "Hammer / Bullish Pin Bar (long lower wick -- buyers rejected the downside)", "BULLISH", 0.65
        elif body_pct0 < 0.35 and upper_wick0 > body0 * 2 and lower_wick0 < body0 * 0.5:
            pattern, bias, strength = "Shooting Star / Bearish Pin Bar (long upper wick -- sellers rejected the upside)", "BEARISH", 0.65

        # 3) Doji -- open ≈ close, pure indecision
        elif body_pct0 < 0.1:
            pattern, bias, strength = "Doji (indecision -- no clear side in control)", "NEUTRAL", 0.2

        # 4) Marubozu -- near-zero wicks, full-body momentum candle
        elif body_pct0 > 0.85:
            if cl0 > o0:
                pattern, bias, strength = "Bullish Marubozu (strong one-directional momentum)", "BULLISH", 0.7
            else:
                pattern, bias, strength = "Bearish Marubozu (strong one-directional momentum)", "BEARISH", 0.7

        return {"pattern": pattern, "bias": bias, "strength": strength}
    except Exception:
        return {"pattern": "NONE", "bias": "NEUTRAL", "strength": 0.0}


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
