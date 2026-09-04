"""
pre_move_detector.py — NEW, purely additive module.

Everything else in this tool answers "what to do" AFTER price is already
near a level (Early Warning) or a full checklist has aligned (AI Trade
Decision Engine). This module answers a different, earlier question:
"is the market coiling right now, and which way is it leaning?"

app.py already computes `is_choppy` from Bollinger Band width contraction
-- but today that flag only does one thing: it tells the decision engine
to go silent ("staying flat"). That's the right call for a precise
strike/SL/target trade -- reliability genuinely is lower in a coil -- but
it means the tool says NOTHING while price is compressing right before a
big directional candle, which is exactly the gap that was reported: a
large move happened and the dashboard had shown no heads-up beforehand.

This module reads the SAME compression, but instead of going silent it
scores the lean building up inside the coil using price-vs-band position,
EMA slope, RSI/MACD tilt, up-volume vs down-volume inside the coil, the
latest candlestick pattern, and recent FVG/Order-Block imbalance -- then
reports a direction + the exact trigger prices (upper/lower band) to
watch for the breakout to actually confirm.

Explicitly NOT a trade signal: no strike, no entry, no SL, no target.
It answers "which way should I be watching, before the candle prints" --
the AI Trade Decision Engine still requires an actual confirmed break
before it will call a real setup, and this module doesn't loosen that.
"""

import pandas as pd


def detect_pre_move_squeeze(df, fvg=None, ob=None, candle_pattern=None,
                             baseline_window=50, min_squeeze_candles=3, squeeze_ratio=0.75):
    """
    Returns:
      {
        "squeeze_active": bool,
        "squeeze_candles": int,        # consecutive recent candles inside the compression band
        "lean": "BULLISH" | "BEARISH" | "NEUTRAL",
        "lean_score_pct": float,       # 0-100, distance from 50 = conviction strength
        "trigger_up": float | None,    # price that would confirm a bullish breakout right now
        "trigger_down": float | None,  # price that would confirm a bearish breakout right now
        "factors": [str, ...],
      }
    "squeeze_active" is False whenever there isn't a real, sustained
    compression -- this stays quiet on ordinary candles, it doesn't fire
    on every refresh.

    Uses the SAME baseline definition as app.py's own `is_choppy` flag
    (current BB width vs a slower rolling AVERAGE, default 50 candles) --
    deliberately NOT a rolling min/mean of the same short recent window,
    which would compare a shrinking value against itself and rarely fire
    cleanly. Comparing against a longer, slower baseline is what actually
    tells you the coil is TIGHT relative to normal, not just quiet.
    """
    empty = {"squeeze_active": False, "squeeze_candles": 0, "lean": "NEUTRAL",
              "lean_score_pct": 50.0, "trigger_up": None, "trigger_down": None, "factors": []}

    needed_cols = {'Close', 'High', 'Low', 'Open'}
    if df is None or len(df) < max(baseline_window, 25) or not needed_cols.issubset(df.columns):
        return empty

    d = df.copy()
    if 'BB_Width' not in d.columns or 'BB_Mid' not in d.columns or 'BB_Upper' not in d.columns:
        mid = d['Close'].rolling(20).mean()
        std = d['Close'].rolling(20).std()
        d['BB_Mid'] = mid
        d['BB_Upper'] = mid + 2 * std
        d['BB_Lower'] = mid - 2 * std
        d['BB_Width'] = (d['BB_Upper'] - d['BB_Lower']) / mid

    width = d['BB_Width']
    baseline_avg = width.rolling(baseline_window).mean()
    if pd.isna(width.iloc[-1]) or pd.isna(baseline_avg.iloc[-1]) or baseline_avg.iloc[-1] <= 0:
        return empty

    threshold = baseline_avg.iloc[-1] * squeeze_ratio

    # Count consecutive most-recent candles that qualify as "inside the coil"
    # relative to that same slow baseline (baseline itself barely moves
    # candle-to-candle over just a few bars, so this isn't self-referential).
    squeeze_candles = 0
    for w in width.iloc[::-1]:
        if pd.notna(w) and w <= threshold:
            squeeze_candles += 1
        else:
            break

    squeeze_active = bool(squeeze_candles >= min_squeeze_candles)
    if not squeeze_active:
        return {**empty, "squeeze_candles": squeeze_candles}

    # ---- Directional lean while coiled: score starts neutral at 50 ----
    score = 50.0
    factors = []
    window = d.iloc[-squeeze_candles:]

    close = float(d['Close'].iloc[-1])
    bb_mid = float(d['BB_Mid'].iloc[-1]) if pd.notna(d['BB_Mid'].iloc[-1]) else None
    if bb_mid is not None:
        if close > bb_mid:
            score += 8
            factors.append(f"Price (₹{close:,.2f}) coiling ABOVE the band midline (₹{bb_mid:,.2f})")
        elif close < bb_mid:
            score -= 8
            factors.append(f"Price (₹{close:,.2f}) coiling BELOW the band midline (₹{bb_mid:,.2f})")

    if 'EMA_20' in d.columns and len(d) > 5 and pd.notna(d['EMA_20'].iloc[-5]):
        ema_slope = float(d['EMA_20'].iloc[-1] - d['EMA_20'].iloc[-5])
        if ema_slope > 0:
            score += 6
            factors.append("EMA 20 mildly rising through the coil")
        elif ema_slope < 0:
            score -= 6
            factors.append("EMA 20 mildly falling through the coil")

    if 'RSI' in d.columns and pd.notna(d['RSI'].iloc[-1]):
        rsi = float(d['RSI'].iloc[-1])
        if rsi > 55:
            score += 6
            factors.append(f"RSI {rsi:.0f} leaning bullish inside the coil")
        elif rsi < 45:
            score -= 6
            factors.append(f"RSI {rsi:.0f} leaning bearish inside the coil")

    if 'MACD_Hist' in d.columns and pd.notna(d['MACD_Hist'].iloc[-1]):
        hist = float(d['MACD_Hist'].iloc[-1])
        if hist > 0:
            score += 6
            factors.append("MACD histogram positive -- mild bullish momentum building")
        elif hist < 0:
            score -= 6
            factors.append("MACD histogram negative -- mild bearish momentum building")

    if 'Volume' in d.columns and window['Volume'].sum() > 0:
        up_vol = window.loc[window['Close'] >= window['Open'], 'Volume'].sum()
        down_vol = window.loc[window['Close'] < window['Open'], 'Volume'].sum()
        total_vol = up_vol + down_vol
        if total_vol > 0:
            up_share = up_vol / total_vol
            if up_share > 0.58:
                score += 10
                factors.append(f"Volume inside the coil skewed to up-candles ({up_share*100:.0f}%) -- accumulation")
            elif up_share < 0.42:
                score -= 10
                factors.append(f"Volume inside the coil skewed to down-candles ({(1-up_share)*100:.0f}%) -- distribution")

    if candle_pattern and candle_pattern.get('bias') in ('BULLISH', 'BEARISH'):
        delta = 10 * float(candle_pattern.get('strength', 0.5))
        if candle_pattern['bias'] == 'BULLISH':
            score += delta
        else:
            score -= delta
        factors.append(f"Latest candle: {candle_pattern.get('pattern', candle_pattern['bias'])}")

    def _net_bias(zones, label):
        nonlocal score
        if not zones:
            return
        bulls = sum(1 for z in zones if "Bullish" in z.get("Type", ""))
        bears = sum(1 for z in zones if "Bearish" in z.get("Type", ""))
        if bulls > bears:
            score += 5
            factors.append(f"Recent {label}: net bullish ({bulls} bull vs {bears} bear)")
        elif bears > bulls:
            score -= 5
            factors.append(f"Recent {label}: net bearish ({bears} bear vs {bulls} bull)")

    _net_bias(fvg, "Fair Value Gaps")
    _net_bias(ob, "Order Blocks")

    score = max(5.0, min(95.0, score))
    if score >= 60:
        lean = "BULLISH"
    elif score <= 40:
        lean = "BEARISH"
    else:
        lean = "NEUTRAL"

    trigger_up = float(d['BB_Upper'].iloc[-1]) if pd.notna(d['BB_Upper'].iloc[-1]) else None
    trigger_down = float(d['BB_Lower'].iloc[-1]) if pd.notna(d['BB_Lower'].iloc[-1]) else None

    return {
        "squeeze_active": True,
        "squeeze_candles": squeeze_candles,
        "lean": lean,
        "lean_score_pct": round(score, 1),
        "trigger_up": round(trigger_up, 2) if trigger_up is not None else None,
        "trigger_down": round(trigger_down, 2) if trigger_down is not None else None,
        "factors": factors,
    }
