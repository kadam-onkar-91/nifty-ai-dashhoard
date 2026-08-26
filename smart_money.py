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


def calculate_premium_discount_zone(df, live_price):
    """
    Core ICT (Inner Circle Trader) concept: split the recent trading range
    into Premium (top ~30%, price is "expensive" -- institutional smart
    money favors SELLING here, expecting a pull back toward Discount) and
    Discount (bottom ~30%, price is "cheap" -- smart money favors BUYING
    here, expecting a push toward Premium). Equilibrium (middle 40%) is
    neutral. This directly answers "which side is price more likely to
    go" independent of the break-vs-bounce statistical estimate above --
    when both agree, that's real added confidence; when they conflict,
    that's a useful warning too.
    """
    if df is None or len(df) < 10 or live_price is None:
        return {"zone": "UNKNOWN", "equilibrium": None, "range_high": None,
                "range_low": None, "position_pct": None}

    lookback = min(30, len(df))
    range_high = float(df['High'].tail(lookback).max())
    range_low = float(df['Low'].tail(lookback).min())
    if range_high == range_low:
        return {"zone": "UNKNOWN", "equilibrium": None, "range_high": range_high,
                "range_low": range_low, "position_pct": None}

    equilibrium = (range_high + range_low) / 2
    position_pct = ((live_price - range_low) / (range_high - range_low)) * 100

    if position_pct >= 70:
        zone = "PREMIUM"
    elif position_pct <= 30:
        zone = "DISCOUNT"
    else:
        zone = "EQUILIBRIUM"

    return {
        "zone": zone, "equilibrium": round(equilibrium, 2), "range_high": round(range_high, 2),
        "range_low": round(range_low, 2), "position_pct": round(position_pct, 1)
    }


def detect_liquidity_pools(df, tolerance_pct=0.05):
    """
    ICT "liquidity pool" concept: clusters of nearby equal highs (buy-side
    liquidity -- stop-losses of short sellers rest just above) or equal
    lows (sell-side liquidity -- stop-losses of long holders rest just
    below). Price is statistically drawn to sweep these pools (grab the
    stops) before making its real move -- a classic ICT stop-hunt pattern.
    If a support/resistance level coincides with one of these, expect a
    brief wick through it before the real reversal, not necessarily a
    genuine breakout.
    """
    if df is None or len(df) < 10:
        return {"equal_highs": [], "equal_lows": []}

    lookback = df.tail(30)
    highs = lookback['High'].values
    lows = lookback['Low'].values

    equal_highs = set()
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if highs[i] and abs(highs[i] - highs[j]) / highs[i] * 100 <= tolerance_pct:
                equal_highs.add(round((highs[i] + highs[j]) / 2, 2))

    equal_lows = set()
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            if lows[i] and abs(lows[i] - lows[j]) / lows[i] * 100 <= tolerance_pct:
                equal_lows.add(round((lows[i] + lows[j]) / 2, 2))

    return {
        "equal_highs": sorted(equal_highs)[-3:],
        "equal_lows": sorted(equal_lows)[:3],
    }


def calculate_ict_bias(df, live_price, proximity_info):
    """
    Combines Premium/Discount zone with liquidity-pool proximity into a
    single ICT confluence score (-1 bearish to +1 bullish) and a plain
    description, for direct use as another vote in the confluence engine.
    """
    pd_zone = calculate_premium_discount_zone(df, live_price)
    pools = detect_liquidity_pools(df)

    score = 0.0
    desc = "Price is in Equilibrium — no strong ICT premium/discount bias right now."

    if pd_zone["zone"] == "PREMIUM":
        score = -0.6
        desc = (f"Price is in the PREMIUM zone ({pd_zone['position_pct']}% of recent range) — "
                f"ICT bias favors SELLING here, expecting a pull back toward "
                f"equilibrium ₹{pd_zone['equilibrium']} / discount ₹{pd_zone['range_low']}.")
    elif pd_zone["zone"] == "DISCOUNT":
        score = 0.6
        desc = (f"Price is in the DISCOUNT zone ({pd_zone['position_pct']}% of recent range) — "
                f"ICT bias favors BUYING here, expecting a push toward "
                f"equilibrium ₹{pd_zone['equilibrium']} / premium ₹{pd_zone['range_high']}.")

    # If the level currently being approached also lines up with a
    # liquidity pool, flag the stop-hunt risk explicitly.
    pool_note = None
    if proximity_info:
        status = proximity_info.get("status")
        tol = None
        if status == "APPROACHING_SUPPORT" and proximity_info.get("support"):
            lvl = proximity_info["support"]
            tol = lvl * 0.0015
            near_pool = [p for p in pools["equal_lows"] if abs(p - lvl) <= tol]
            if near_pool:
                pool_note = (f"⚠️ This support lines up with an equal-lows liquidity pool (₹{near_pool[0]}) — "
                             f"classic ICT stop-hunt zone: expect a possible wick BELOW it to grab sell-side "
                             f"liquidity before any real reversal.")
        elif status == "APPROACHING_RESISTANCE" and proximity_info.get("resistance"):
            lvl = proximity_info["resistance"]
            tol = lvl * 0.0015
            near_pool = [p for p in pools["equal_highs"] if abs(p - lvl) <= tol]
            if near_pool:
                pool_note = (f"⚠️ This resistance lines up with an equal-highs liquidity pool (₹{near_pool[0]}) — "
                             f"classic ICT stop-hunt zone: expect a possible wick ABOVE it to grab buy-side "
                             f"liquidity before any real reversal.")

    return {
        "score": max(-1.0, min(1.0, score)),
        "description": desc,
        "premium_discount": pd_zone,
        "liquidity_pools": pools,
        "pool_warning": pool_note,
    }


def calculate_directional_bias(df):
    """
    Which way is price currently leaning -- toward resistance or toward
    support -- using short-term momentum (rate of change, EMA slope, MACD
    histogram trend). This is the "before it even gets near a level"
    directional read.
    """
    if df is None or len(df) < 6:
        return {"direction": "NEUTRAL", "confidence_pct": 50.0, "momentum_score": 0.0}

    closes = df['Close'].tail(6).values
    roc = ((closes[-1] - closes[0]) / closes[0]) * 100 if closes[0] else 0.0

    if 'EMA_20' in df.columns:
        ema20 = df['EMA_20'].tail(6).values
        ema_slope = ((ema20[-1] - ema20[0]) / ema20[0]) * 100 if ema20[0] else 0.0
    else:
        ema_slope = roc

    if 'MACD_Hist' in df.columns:
        macd_hist = df['MACD_Hist'].tail(4).values
        macd_trend = float(macd_hist[-1] - macd_hist[0]) if len(macd_hist) >= 2 else 0.0
    else:
        macd_trend = 0.0

    score = 0.0
    score += max(-1.0, min(1.0, roc / 0.3)) * 0.45
    score += max(-1.0, min(1.0, ema_slope / 0.2)) * 0.35
    score += max(-1.0, min(1.0, macd_trend / 3.0)) * 0.20
    score = max(-1.0, min(1.0, score))

    if score > 0.1:
        direction = "TOWARD_RESISTANCE"
        confidence_pct = round(50 + score * 45, 1)
    elif score < -0.1:
        direction = "TOWARD_SUPPORT"
        confidence_pct = round(50 + abs(score) * 45, 1)
    else:
        direction = "NEUTRAL"
        confidence_pct = round(50 - abs(score) * 10, 1)

    return {"direction": direction, "confidence_pct": confidence_pct, "momentum_score": round(score, 2)}


def calculate_break_vs_bounce(df, live_price, proximity_info):
    """
    When price is approaching/at a support or resistance: estimates the
    probability it BREAKS THROUGH (continues) vs BOUNCES OFF (reverses),
    using three real signals --
      1) Momentum direction going into the level (from calculate_directional_bias)
      2) RSI exhaustion at the level (extreme RSI favors a reversal/bounce)
      3) How many times this exact level has already been tested recently
         (a level that's been tested repeatedly is statistically weaker)
    This is a heuristic estimate, not a trained/backtested model -- it is
    intentionally kept within a 15-85% band so it never claims false
    certainty either way.
    """
    if df is None or proximity_info is None:
        return None
    status = proximity_info.get("status")
    if status not in ("APPROACHING_SUPPORT", "APPROACHING_RESISTANCE"):
        return None

    is_support = status == "APPROACHING_SUPPORT"
    level = proximity_info.get("support") if is_support else proximity_info.get("resistance")
    if not level:
        return None

    dirbias = calculate_directional_bias(df)
    momentum_score = dirbias.get("momentum_score", 0.0)
    # Momentum continuing IN the breaking direction raises break odds.
    momentum_break_push = max(0.0, -momentum_score) if is_support else max(0.0, momentum_score)

    rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and len(df) > 0 else None
    exhaustion_bounce_push = 0.0
    if rsi is not None:
        if is_support and rsi < 35:
            exhaustion_bounce_push = min(1.0, (35 - rsi) / 20)
        elif not is_support and rsi > 65:
            exhaustion_bounce_push = min(1.0, (rsi - 65) / 20)

    tolerance = level * 0.001
    lookback = df.tail(20)
    if is_support:
        touches = int(((lookback['Low'] - level).abs() <= tolerance).sum())
    else:
        touches = int(((lookback['High'] - level).abs() <= tolerance).sum())
    touch_break_push = min(1.0, max(0, touches - 1) * 0.20)

    break_pct = 50.0
    break_pct += momentum_break_push * 25
    break_pct -= exhaustion_bounce_push * 20
    break_pct += touch_break_push * 15
    break_pct = max(15.0, min(85.0, break_pct))
    bounce_pct = round(100 - break_pct, 1)
    break_pct = round(break_pct, 1)

    return {
        "level_type": "Support" if is_support else "Resistance",
        "level": level,
        "break_probability_pct": break_pct,
        "bounce_probability_pct": bounce_pct,
        "touches_tested": touches,
        "momentum_score": round(momentum_score, 2),
        "rsi": round(rsi, 1) if rsi is not None else None,
    }


def detect_approaching_zone(df, live_price):
    """
    BOS/CHoCH above only fire AFTER price has already broken the recent
    high/low -- by definition that's a lagging confirmation. This function
    is the anticipatory counterpart: it flags when price is CLOSING IN ON
    a recent support/resistance level BEFORE it's touched, so a reaction
    (bounce off support / rejection at resistance) can be anticipated
    instead of only reacting after the fact.

    Returns a dict: status, resistance, support, distance_to_resistance_pct,
    distance_to_support_pct.
    """
    if df is None or len(df) < 5 or live_price is None:
        return {"status": "UNKNOWN", "resistance": None, "support": None,
                "distance_to_resistance_pct": None, "distance_to_support_pct": None}

    lookback = min(20, len(df) - 1)
    recent_high = float(df['High'].iloc[-lookback:].max())
    recent_low = float(df['Low'].iloc[-lookback:].min())

    dist_res = recent_high - live_price
    dist_sup = live_price - recent_low
    dist_res_pct = (dist_res / live_price) * 100 if live_price else None
    dist_sup_pct = (dist_sup / live_price) * 100 if live_price else None

    # "Close to a level" threshold -- ~0.12% of spot (roughly 25-30 points
    # on Nifty around the 24,000 mark). Tune if it feels too tight/wide.
    threshold_pct = 0.12

    status = "MID_RANGE"
    if dist_res_pct is not None and 0 <= dist_res_pct <= threshold_pct:
        status = "APPROACHING_RESISTANCE"
    elif dist_sup_pct is not None and 0 <= dist_sup_pct <= threshold_pct:
        status = "APPROACHING_SUPPORT"
    elif dist_res < 0:
        status = "ABOVE_RECENT_RANGE"
    elif dist_sup < 0:
        status = "BELOW_RECENT_RANGE"

    return {
        "status": status,
        "resistance": round(recent_high, 2),
        "support": round(recent_low, 2),
        "distance_to_resistance_pct": round(dist_res_pct, 3) if dist_res_pct is not None else None,
        "distance_to_support_pct": round(dist_sup_pct, 3) if dist_sup_pct is not None else None,
    }
