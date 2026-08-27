import pandas as pd
import numpy as np

"""
EARLY-WARNING SUPPORT/RESISTANCE APPROACH PREDICTOR
-----------------------------------------------------
Problem this solves: the existing confluence engine only confirms a
direction AFTER price has already reacted (touched a level and moved).
By then price is often already near the NEXT level, so the signal feels
"late". This module looks at price WHILE it is still approaching a key
support/resistance level (within ~1 ATR of it) and estimates, before the
touch happens, the probability the level BREAKS vs the probability price
REVERSES (bounces) off it -- combining ICT-style confluence (liquidity
sweeps, order blocks, fair value gaps, premium/discount) with classic
momentum/volume/wick exhaustion reads.

This module is fully standalone and does not modify or depend on any
other file's internal logic -- it only reads the OHLC/indicator data
that market_data.py already produces, plus the FVG/OB lists smart_money.py
already produces.
"""

PROXIMITY_ATR_MULT = 1.0      # "approaching zone" = within 1x ATR of a level
SWING_LOOKBACK = 15           # candles each side to confirm a fractal pivot
PDLEVELS_LOOKBACK_DAYS = 3    # how many previous days' High/Low to keep
ENTRY_CONFIDENCE_THRESHOLD = 60.0  # only suggest an actionable early entry above this break/bounce %


def _find_swing_points(df, lookback=SWING_LOOKBACK):
    """Simple fractal-style swing high/low detector using a centered window."""
    highs, lows = [], []
    if df is None or len(df) < (lookback * 2 + 1):
        return highs, lows
    h = df['High'].values
    l = df['Low'].values
    for i in range(lookback, len(df) - lookback):
        window_h = h[i - lookback:i + lookback + 1]
        window_l = l[i - lookback:i + lookback + 1]
        if h[i] == window_h.max():
            highs.append(float(h[i]))
        if l[i] == window_l.min():
            lows.append(float(l[i]))
    return highs, lows


def get_key_levels(df):
    """
    Builds key support/resistance levels from:
    - Previous day(s) High/Low (classic intraday levels every trader watches)
    - Fractal swing highs/lows from recent candle history
    """
    result = {'resistances': [], 'supports': [], 'prev_day_levels': []}
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return result

    try:
        daily = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
        daily = daily.iloc[:-1]  # drop today's still-forming session
        for date_idx, row in daily.tail(PDLEVELS_LOOKBACK_DAYS).iterrows():
            result['prev_day_levels'].append({
                'date': str(date_idx), 'high': float(row['High']),
                'low': float(row['Low']), 'close': float(row['Close'])
            })
            result['resistances'].append(float(row['High']))
            result['supports'].append(float(row['Low']))
    except Exception:
        pass

    try:
        swing_highs, swing_lows = _find_swing_points(df.tail(150))
        result['resistances'].extend(swing_highs)
        result['supports'].extend(swing_lows)
    except Exception:
        pass

    result['resistances'] = sorted(set(round(x, 2) for x in result['resistances']))
    result['supports'] = sorted(set(round(x, 2) for x in result['supports']))
    return result


def _nearest_level(levels, price, direction):
    candidates = [lv for lv in levels if (lv > price if direction == 'above' else lv < price)]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - price))


def _candle_wick_ratios(df, n=5):
    recent = df.tail(n)
    if recent.empty:
        return 0.0, 0.0
    rng = (recent['High'] - recent['Low']).replace(0, 0.01)
    lower_wick = (recent[['Open', 'Close']].min(axis=1) - recent['Low'])
    upper_wick = (recent['High'] - recent[['Open', 'Close']].max(axis=1))
    return float((lower_wick / rng).mean()), float((upper_wick / rng).mean())


def _liquidity_sweep_recent(df, level, kind, n=8):
    """
    ICT-style liquidity sweep: did price wick THROUGH the level recently
    and then close back on the defended side? Classic stop-hunt pattern
    that front-runs a reversal ("Turtle Soup" / Judas Swing).
    """
    recent = df.tail(n)
    if recent.empty or level is None:
        return False
    if kind == 'support':
        return bool(((recent['Low'] < level) & (recent['Close'] > level)).any())
    return bool(((recent['High'] > level) & (recent['Close'] < level)).any())


def _premium_discount_position(df, lookback=100):
    """ICT premium/discount: equilibrium (50%) of the recent swing range."""
    recent = df.tail(lookback)
    if recent.empty:
        return 'equilibrium', 0.5
    swing_high, swing_low = recent['High'].max(), recent['Low'].min()
    if swing_high == swing_low:
        return 'equilibrium', 0.5
    pos = (recent['Close'].iloc[-1] - swing_low) / (swing_high - swing_low)
    zone = 'discount' if pos < 0.5 else 'premium'
    return zone, round(float(pos), 2)


def _volume_trend(df, n=5):
    """Positive = volume expanding into the move, negative = contracting."""
    if 'Volume' not in df.columns or len(df) < n + 3:
        return 0.0
    recent_vol = df['Volume'].tail(n).mean()
    prior_vol = df['Volume'].tail(n * 2).head(n).mean()
    if prior_vol <= 0:
        return 0.0
    return float(np.clip((recent_vol - prior_vol) / prior_vol, -1, 1))


def _rsi_divergence(df, kind, n=10):
    """
    Rough divergence check over the last n candles.
    support: bullish divergence (price lower-low, RSI higher-low) -> reversal risk.
    resistance: bearish divergence (price higher-high, RSI lower-high) -> reversal risk.
    Returns negative (favors reversal/bounce) or slightly positive (favors continuation).
    """
    if 'RSI' not in df.columns or len(df) < n:
        return 0.0
    recent = df.tail(n)
    try:
        if kind == 'support':
            idx = recent['Low'].idxmin()
            half = recent.loc[:idx]
            if len(half) < 3:
                return 0.0
            f, s = half.iloc[:len(half) // 2], half.iloc[len(half) // 2:]
            if f['Low'].min() >= s['Low'].min() and f['RSI'].min() < s['RSI'].min():
                return -0.3
            return 0.1
        else:
            idx = recent['High'].idxmax()
            half = recent.loc[:idx]
            if len(half) < 3:
                return 0.0
            f, s = half.iloc[:len(half) // 2], half.iloc[len(half) // 2:]
            if f['High'].max() <= s['High'].max() and f['RSI'].max() > s['RSI'].max():
                return -0.3
            return 0.1
    except Exception:
        return 0.0


def _parse_price(s):
    try:
        return float(str(s).replace(',', ''))
    except Exception:
        return None


def _zone_confluence(zone_list, level_price, atr):
    """Net bullish-vs-bearish OB/FVG confluence sitting at/near this level."""
    if not zone_list or level_price is None or not atr:
        return 0
    tol = 0.3 * atr
    hits = 0
    for z in zone_list:
        p = _parse_price(z.get('Price'))
        if p is None or abs(p - level_price) > tol:
            continue
        t = str(z.get('Type', '')).lower()
        if 'bullish' in t:
            hits += 1
        elif 'bearish' in t:
            hits -= 1
    return hits


def _detect_price_action_pattern(df):
    """
    NEW — Classic price-action candlestick pattern recognition on the most
    recent candle(s) forming right into the level: Engulfing, Pin Bar /
    Hammer / Shooting Star, Marubozu (momentum candle), Doji, Inside Bar.
    These are the rawest, earliest price-action tells -- often visible
    before RSI/volume/anything else reacts, which is exactly the kind of
    extra detail useful right as price approaches a level.

    Returns (pattern_name, net_bias) where net_bias is in [-1, 1]:
    positive = bullish pattern, negative = bearish pattern, 0 = neutral/
    indecisive pattern (Doji/Inside Bar). Returns (None, 0.0) if nothing
    recognizable formed on the latest candle.
    """
    if df is None or len(df) < 2:
        return None, 0.0
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        last_range = max(float(last['High'] - last['Low']), 0.01)
        last_body = abs(float(last['Close'] - last['Open']))
        body_ratio = last_body / last_range
        upper_wick = float(last['High']) - max(float(last['Open']), float(last['Close']))
        lower_wick = min(float(last['Open']), float(last['Close'])) - float(last['Low'])

        prev_bullish = prev['Close'] > prev['Open']
        last_bullish = last['Close'] > last['Open']

        # 1. Engulfing — strongest reversal pattern, checked first
        if (not prev_bullish and last_bullish and
                last['Close'] >= prev['Open'] and last['Open'] <= prev['Close']):
            return "Bullish Engulfing", 0.8
        if (prev_bullish and not last_bullish and
                last['Open'] >= prev['Close'] and last['Close'] <= prev['Open']):
            return "Bearish Engulfing", -0.8

        # 2. Pin Bar / Hammer / Shooting Star — small body, long single wick
        if body_ratio < 0.35:
            if lower_wick > 2 * last_body and lower_wick > upper_wick:
                return "Bullish Pin Bar / Hammer", 0.6
            if upper_wick > 2 * last_body and upper_wick > lower_wick:
                return "Bearish Pin Bar / Shooting Star", -0.6

        # 3. Marubozu — strong momentum candle, body dominates the range
        if body_ratio > 0.85:
            return (("Bullish Marubozu (strong momentum candle)", 0.5) if last_bullish else
                    ("Bearish Marubozu (strong momentum candle)", -0.5))

        # 4. Doji — indecision, tiny body
        if body_ratio < 0.1:
            return "Doji (indecision candle)", 0.0

        # 5. Inside Bar — compression, current candle fully inside previous
        if float(last['High']) <= float(prev['High']) and float(last['Low']) >= float(prev['Low']):
            return "Inside Bar (compression, breakout pending)", 0.0

        return None, 0.0
    except Exception:
        return None, 0.0


def _oi_pcr_confluence(df_option_chain, level_price, band=50):
    """
    NEW — Options OI/PCR confluence (strongest institutional footprint we
    have). Looks at Call OI vs Put OI in a strike band around the level:
    heavy Put OI writing near a support = writers are defending it (they
    profit if it holds) = bounce favored. Heavy Call OI writing near a
    resistance = writers defending it = rejection favored. This is real
    money positioned at that exact price, not just a chart pattern.
    Returns pcr_bias in [-1, 1]: positive = put-heavy (bullish), negative
    = call-heavy (bearish). Returns None if no usable option chain data.
    """
    try:
        if df_option_chain is None or df_option_chain.empty or level_price is None:
            return None
        if 'Strike' not in df_option_chain.columns:
            return None
        near = df_option_chain[(df_option_chain['Strike'] >= level_price - band) &
                                (df_option_chain['Strike'] <= level_price + band)]
        if near.empty:
            return None
        call_oi = float(near['Call OI'].sum()) if 'Call OI' in near.columns else 0.0
        put_oi = float(near['Put OI'].sum()) if 'Put OI' in near.columns else 0.0
        total = call_oi + put_oi
        if total <= 0:
            return None
        return float(np.clip((put_oi - call_oi) / total, -1, 1)), call_oi, put_oi
    except Exception:
        return None


def _vwap_bias(df, atr, n=6):
    """
    NEW — Where is price relative to VWAP, and which way is VWAP itself
    sloping, as price approaches the level? Price on the strong side of a
    rising/falling VWAP is the classic "smart money is still in control"
    tell used alongside ICT concepts. Returns bias in [-1, 1]: positive =
    bullish VWAP positioning, negative = bearish.
    """
    if 'VWAP' not in df.columns or not atr:
        return None
    vwap_series = df['VWAP'].tail(n)
    if len(vwap_series) < 2:
        return None
    try:
        live = float(df['Close'].iloc[-1])
        vwap_now = float(vwap_series.iloc[-1])
        dist = (live - vwap_now) / atr
        slope = (float(vwap_series.iloc[-1]) - float(vwap_series.iloc[0])) / atr
        return float(np.clip((dist + slope) / 2.0, -1, 1))
    except Exception:
        return None


def _htf_structure_bias(df, atr, resample_rule='15min', bars=8):
    """
    NEW — Multi-timeframe confluence: resamples the intraday candles up to
    a higher timeframe (15-min) and checks whether that higher-timeframe
    trend agrees with the move into the level. A lower-timeframe signal
    that also has higher-timeframe structure behind it is far more
    reliable than one that doesn't. Returns bias in [-1, 1].
    """
    if df is None or df.empty or not atr or not isinstance(df.index, pd.DatetimeIndex):
        return None
    try:
        htf_close = df['Close'].resample(resample_rule).last().dropna()
        if len(htf_close) < bars:
            return None
        recent = htf_close.tail(bars)
        slope = float(recent.iloc[-1] - recent.iloc[0])
        return float(np.clip(slope / (atr * 2.0), -1, 1))
    except Exception:
        return None


def _round_number_confluence(level_price):
    """
    NEW — Round numbers (multiples of 100, and to a lesser extent 50) act
    as extra psychological support/resistance -- price tends to hesitate
    or react there even before the "real" technical level is reached.
    This doesn't pick a direction; it just flags that a reaction (of some
    kind) is more likely right at this price, nudging toward a bounce/
    reaction rather than a clean break-through.
    """
    if level_price is None:
        return None
    nearest_100 = round(level_price / 100) * 100
    nearest_50 = round(level_price / 50) * 50
    if abs(level_price - nearest_100) <= 15:
        return 'major', nearest_100
    if abs(level_price - nearest_50) <= 8:
        return 'minor', nearest_50
    return None


def _level_test_count(df, level_price, kind, atr, lookback=100):
    """
    NEW — How many times has price already tested this level recently
    without a clean break? Classic technical-analysis view: a level
    WEAKENS with each additional test (the resting orders defending it
    get consumed), so a level tested 2-3+ times is more likely to finally
    give way than a "fresh" level being tested for the first time.
    """
    if df is None or level_price is None or not atr:
        return 0
    recent = df.tail(lookback)
    if recent.empty:
        return 0
    tol = 0.3 * atr
    try:
        if kind == 'support':
            touches = ((recent['Low'] <= level_price + tol) & (recent['Low'] >= level_price - tol) &
                       (recent['Close'] > level_price - tol)).sum()
        else:
            touches = ((recent['High'] >= level_price - tol) & (recent['High'] <= level_price + tol) &
                       (recent['Close'] < level_price + tol)).sum()
        return int(touches)
    except Exception:
        return 0


def _build_early_entry(target_kind, live_price, atr, break_pct, bounce_pct):
    """
    Turns the break/bounce read into an ACTIONABLE suggestion (not just an
    info %) once conviction crosses ENTRY_CONFIDENCE_THRESHOLD -- this is
    what lets you enter WHILE the move is still forming near the level,
    instead of waiting for the slower main confirmation signal (by which
    point the move is often already over).

    SL/target are ATR-based off the current price (not off the level), so
    the trade stays valid/consistent no matter how close price already is
    to the level.
    """
    conf = max(break_pct, bounce_pct)
    if conf < ENTRY_CONFIDENCE_THRESHOLD:
        return {
            'action': 'WAIT — Conviction Too Low', 'confidence_pct': conf,
            'entry_price': None, 'stop_loss': None, 'target': None
        }

    sl_buffer = max(0.3 * atr, 5.0)

    if target_kind == 'support':
        going_up = bounce_pct > break_pct
        label = 'BUY (Anticipated Bounce)' if going_up else 'SELL (Anticipated Breakdown)'
    else:
        going_up = break_pct > bounce_pct
        label = 'BUY (Anticipated Breakout)' if going_up else 'SELL (Anticipated Rejection)'

    entry = live_price
    if going_up:
        sl = entry - sl_buffer
        target = entry + (sl_buffer * 2.0)
    else:
        sl = entry + sl_buffer
        target = entry - (sl_buffer * 2.0)

    return {
        'action': label, 'confidence_pct': conf,
        'entry_price': round(entry, 2), 'stop_loss': round(sl, 2), 'target': round(target, 2)
    }


def predict_level_reaction(df, live_price, atr, fvg_list=None, ob_list=None, trend_bias=0.0, df_option_chain=None):
    """
    Core "before-the-touch" early-warning function.

    Returns a dict describing whichever key level price is currently
    approaching (support while falling, resistance while rising), with a
    break-probability vs bounce-probability split and the confluence
    factors behind it. If price isn't near any key level right now, it
    just reports the nearest levels with status NO_KEY_LEVEL_NEARBY.

    score convention (internal): negative = favors BOUNCE/REVERSAL,
    positive = favors BREAK/CONTINUATION -- for both support and resistance.
    """
    out = {
        'status': 'NO_KEY_LEVEL_NEARBY', 'approaching': None, 'level_price': None,
        'distance_pts': None, 'break_pct': 50.0, 'bounce_pct': 50.0,
        'factors': [], 'level_type': None, 'directional_bias': None,
        'nearest_support': None, 'nearest_resistance': None, 'early_entry': None
    }
    if df is None or df.empty or live_price is None or not atr or atr <= 0 or len(df) < 10:
        return out

    levels = get_key_levels(df)
    fvg_list, ob_list = fvg_list or [], ob_list or []

    nearest_support = _nearest_level(levels['supports'], live_price, 'below')
    nearest_resistance = _nearest_level(levels['resistances'], live_price, 'above')
    out['nearest_support'] = nearest_support
    out['nearest_resistance'] = nearest_resistance

    proximity = PROXIMITY_ATR_MULT * atr
    dist_support = (live_price - nearest_support) if nearest_support is not None else None
    dist_resistance = (nearest_resistance - live_price) if nearest_resistance is not None else None

    recent_closes = df['Close'].tail(5)
    momentum_down = len(recent_closes) >= 2 and recent_closes.iloc[-1] < recent_closes.iloc[0]
    momentum_up = len(recent_closes) >= 2 and recent_closes.iloc[-1] > recent_closes.iloc[0]

    approaching_support = dist_support is not None and dist_support <= proximity and momentum_down
    approaching_resistance = dist_resistance is not None and dist_resistance <= proximity and momentum_up

    target_kind = None
    if approaching_support and approaching_resistance:
        target_kind = 'support' if dist_support <= dist_resistance else 'resistance'
    elif approaching_support:
        target_kind = 'support'
    elif approaching_resistance:
        target_kind = 'resistance'

    if target_kind is None:
        return out

    level_price = nearest_support if target_kind == 'support' else nearest_resistance
    distance_pts = dist_support if target_kind == 'support' else dist_resistance

    score = 0.0
    factors = []

    # 1. RSI momentum divergence (early exhaustion tell)
    div_score = _rsi_divergence(df, target_kind)
    score += div_score
    if div_score < 0:
        factors.append("RSI momentum divergence detected -> underlying push into this level is weakening (favors reversal)")
    else:
        factors.append("No RSI divergence -> momentum still aligned with the move (favors continuation)")

    # 2. Wick rejection bias
    lower_ratio, upper_ratio = _candle_wick_ratios(df, n=5)
    if target_kind == 'support':
        wick_score = lower_ratio - upper_ratio
        score -= 0.3 * float(np.clip(wick_score, -1, 1))
        if wick_score > 0.15:
            factors.append("Long lower wicks forming into support -> buyers actively defending (favors bounce)")
        elif wick_score < -0.15:
            factors.append("Weak wicks / strong down-closes into support -> sellers in control (favors breakdown)")
        else:
            factors.append("Wick pattern neutral into support -> no clear rejection or exhaustion signal yet")
    else:
        wick_score = upper_ratio - lower_ratio
        score -= 0.3 * float(np.clip(wick_score, -1, 1))
        if wick_score > 0.15:
            factors.append("Long upper wicks forming into resistance -> sellers actively defending (favors rejection)")
        elif wick_score < -0.15:
            factors.append("Weak wicks / strong up-closes into resistance -> buyers in control (favors breakout)")
        else:
            factors.append("Wick pattern neutral into resistance -> no clear rejection or exhaustion signal yet")

    # 3. ICT Liquidity Sweep — already fired = strong early reversal tell
    if _liquidity_sweep_recent(df, level_price, target_kind, n=8):
        score -= 0.3
        factors.append("ICT Liquidity Sweep already detected at this level (stop-hunt + reclaim) -> strong early reversal signal")
    else:
        factors.append("No liquidity sweep detected yet at this level")

    # 4. Order Block / FVG confluence at this exact level
    zone_conf = _zone_confluence(ob_list, level_price, atr) + _zone_confluence(fvg_list, level_price, atr)
    if target_kind == 'support':
        score += -0.15 * float(np.clip(zone_conf, -2, 2))
        if zone_conf > 0:
            factors.append("Bullish Order Block / FVG sitting right at this support -> institutional demand zone (favors bounce)")
        elif zone_conf < 0:
            factors.append("Bearish Order Block / FVG stacked at this support -> weak defense (favors breakdown)")
        else:
            factors.append("No Order Block / FVG confluence right at this support -> no institutional footprint detected here")
    else:
        score += 0.15 * float(np.clip(zone_conf, -2, 2))
        if zone_conf > 0:
            factors.append("Bullish Order Block / FVG stacked at this resistance -> buyers overpowering supply (favors breakout)")
        elif zone_conf < 0:
            factors.append("Bearish Order Block / FVG sitting right at this resistance -> institutional supply zone (favors rejection)")
        else:
            factors.append("No Order Block / FVG confluence right at this resistance -> no institutional footprint detected here")

    # 5. Volume trend into the move
    vol_score = _volume_trend(df, n=5)
    score += 0.2 * vol_score
    if vol_score > 0.15:
        factors.append("Volume expanding into this move -> favors continuation through the level")
    elif vol_score < -0.15:
        factors.append("Volume contracting into this move -> favors exhaustion/reversal at the level")
    else:
        factors.append("Volume trend neutral -> no strong expansion or contraction into this move")

    # 6. Broader trend bias (pass in e.g. the confluence engine's tech_score)
    trend_sign = -1 if target_kind == 'support' else 1
    score += 0.15 * trend_sign * float(np.clip(trend_bias, -1, 1))
    if trend_bias > 0.15:
        factors.append("Broader trend is bullish -> " + ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
    elif trend_bias < -0.15:
        factors.append("Broader trend is bearish -> " + ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
    else:
        factors.append("Broader trend is flat/neutral -> no strong macro tilt either way at this level")

    # 7. ICT Premium / Discount zone
    zone, pos = _premium_discount_position(df)
    pd_score = (0.5 - pos) * 2
    if target_kind == 'support':
        score += -0.1 * pd_score
        factors.append(f"Price is in a {zone} zone (ICT) of the recent range -> " +
                        ("favors demand / bounce at support" if zone == 'discount' else "less support defense expected"))
    else:
        score += 0.1 * pd_score
        factors.append(f"Price is in a {zone} zone (ICT) of the recent range -> " +
                        ("favors supply / rejection at resistance" if zone == 'premium' else "resistance may break easier"))

    # 8. NEW — Options OI/PCR confluence (strongest institutional footprint)
    oi_result = _oi_pcr_confluence(df_option_chain, level_price)
    if oi_result is not None:
        pcr_bias, call_oi, put_oi = oi_result
        if target_kind == 'support':
            score += -0.25 * pcr_bias
        else:
            score += 0.25 * pcr_bias
        if pcr_bias > 0.15:
            factors.append(f"Heavy Put OI writing near this {target_kind} (Put OI {put_oi:,.0f} vs Call OI {call_oi:,.0f}) -> writers defending, favors bounce/breakout")
        elif pcr_bias < -0.15:
            factors.append(f"Heavy Call OI writing near this {target_kind} (Call OI {call_oi:,.0f} vs Put OI {put_oi:,.0f}) -> writers defending against upside, favors rejection/breakdown")
        else:
            factors.append("Options OI near this level is balanced -> no strong writer-side bias")
    else:
        factors.append("No usable option-chain OI data near this level")

    # 9. NEW — VWAP position + slope (institutional average price flow)
    vwap_bias = _vwap_bias(df, atr)
    if vwap_bias is not None:
        score += 0.15 * trend_sign * vwap_bias
        if vwap_bias > 0.15:
            factors.append("Price above a rising VWAP -> bullish institutional flow -> " +
                            ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif vwap_bias < -0.15:
            factors.append("Price below a falling VWAP -> bearish institutional flow -> " +
                            ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            factors.append("Price is hugging VWAP -> no clear institutional flow bias right now")
    else:
        factors.append("VWAP data unavailable for this check")

    # 10. NEW — Multi-timeframe (15-min) structure confluence
    htf_bias = _htf_structure_bias(df, atr)
    if htf_bias is not None:
        score += 0.15 * trend_sign * htf_bias
        if htf_bias > 0.15:
            factors.append("15-min higher-timeframe structure is bullish -> " +
                            ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif htf_bias < -0.15:
            factors.append("15-min higher-timeframe structure is bearish -> " +
                            ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            factors.append("15-min higher-timeframe structure is flat -> no strong multi-timeframe confluence")
    else:
        factors.append("Not enough history yet for a 15-min multi-timeframe read")

    # 11. NEW — Round number / psychological level confluence
    round_hit = _round_number_confluence(level_price)
    if round_hit is not None:
        strength, round_val = round_hit
        nudge = -0.1 if strength == 'major' else -0.05
        score += nudge
        factors.append(f"Level sits right at a {'major' if strength=='major' else 'minor'} round number (₹{round_val:,.0f}) -> "
                        f"psychological level, extra hesitation/reaction likely here")
    else:
        factors.append("Level is not near a round psychological number -> no extra round-number effect")

    # 12. NEW — Level strength / test count (more tests = weaker level)
    touches = _level_test_count(df, level_price, target_kind, atr)
    if touches >= 2:
        weaken = min(touches - 1, 3) * 0.05
        score += weaken
        factors.append(f"This level has already been tested {touches} times recently -> each test consumes defending orders, favors an eventual break")
    else:
        factors.append(f"This is a relatively fresh level (tested {touches}x recently) -> defending orders still largely intact")

    # 13. NEW — Price Action candlestick pattern right at the level
    pa_pattern, pa_bias = _detect_price_action_pattern(df)
    if pa_pattern is not None:
        score += 0.2 * trend_sign * pa_bias
        if pa_bias > 0:
            factors.append(f"Price Action: {pa_pattern} forming -> bullish signal -> " +
                            ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif pa_bias < 0:
            factors.append(f"Price Action: {pa_pattern} forming -> bearish signal -> " +
                            ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            factors.append(f"Price Action: {pa_pattern} forming -> indecision/compression, no clear direction from this candle yet")
    else:
        factors.append("No notable price-action candlestick pattern (Engulfing/Pin Bar/Marubozu/Doji) on the current candle")

    score = float(np.clip(score, -1, 1))
    break_pct = float(np.clip(round(50 + score * 40, 1), 10, 90))
    bounce_pct = round(100 - break_pct, 1)

    if target_kind == 'support':
        directional_bias = "BREAKDOWN LIKELY (bearish) 🔴" if break_pct > bounce_pct else "BOUNCE LIKELY (bullish) 🟢"
    else:
        directional_bias = "BREAKOUT LIKELY (bullish) 🟢" if break_pct > bounce_pct else "REJECTION LIKELY (bearish) 🔴"

    early_entry = _build_early_entry(target_kind, live_price, atr, break_pct, bounce_pct)

    out.update({
        'status': 'APPROACHING_LEVEL', 'approaching': target_kind,
        'level_price': round(level_price, 2), 'distance_pts': round(distance_pts, 2),
        'break_pct': break_pct, 'bounce_pct': bounce_pct, 'factors': factors,
        'level_type': 'Support' if target_kind == 'support' else 'Resistance',
        'directional_bias': directional_bias, 'raw_score': round(score, 3),
        'early_entry': early_entry
    })
    return out


# =====================================================================
# NEW — ROUND-NUMBER LADDER CALCULATOR (every 50 / 100 pt level)
# -----------------------------------------------------------------------
# ADDED ON TOP OF EVERYTHING ABOVE -- nothing above this line was touched.
#
# predict_level_reaction() above only ever scores the SINGLE nearest
# support and the SINGLE nearest resistance (swing/prev-day levels).
# This section adds a full "calculator" that runs the exact same
# break-vs-bounce scoring model against EVERY round-number level (every
# 50 pts, with 100 pts getting extra weight as a "major" level via the
# existing _round_number_confluence() check) both above and below the
# live price -- so you get a break/bounce % for every rung of the
# ladder, not just the one nearest level. Same factors, same OI/PCR
# confluence (which already looks at a +/-50 strike band around each
# level -- exactly the "OI support every 50 level" behaviour), same
# ICT/volume/price-action reads as before. Nothing here changes how
# predict_level_reaction() behaves -- it is a fully separate function.
# =====================================================================

LADDER_STEP = 50                 # a level every 50 points
LADDER_LEVELS_EACH_SIDE = 8       # how many rungs above AND below live price


def _score_break_bounce_at_level(df, live_price, atr, level_price, target_kind,
                                  fvg_list, ob_list, trend_bias, df_option_chain):
    """
    Same break-vs-bounce confluence model used inside predict_level_reaction()
    (RSI divergence, wick rejection, liquidity sweep, OB/FVG zone confluence,
    volume trend, broader trend bias, ICT premium/discount, options OI/PCR,
    VWAP bias, 15-min HTF structure, round-number confluence, level test
    count, price-action pattern) -- factored out here so it can be run
    against ANY level_price/kind pair, not just the single nearest level.
    """
    score = 0.0
    factors = []

    div_score = _rsi_divergence(df, target_kind)
    score += div_score
    if div_score < 0:
        factors.append("RSI momentum divergence detected -> underlying push into this level is weakening (favors reversal)")
    else:
        factors.append("No RSI divergence -> momentum still aligned with the move (favors continuation)")

    lower_ratio, upper_ratio = _candle_wick_ratios(df, n=5)
    if target_kind == 'support':
        wick_score = lower_ratio - upper_ratio
        score -= 0.3 * float(np.clip(wick_score, -1, 1))
        if wick_score > 0.15:
            factors.append("Long lower wicks forming into support -> buyers actively defending (favors bounce)")
        elif wick_score < -0.15:
            factors.append("Weak wicks / strong down-closes into support -> sellers in control (favors breakdown)")
        else:
            factors.append("Wick pattern neutral into support -> no clear rejection or exhaustion signal yet")
    else:
        wick_score = upper_ratio - lower_ratio
        score -= 0.3 * float(np.clip(wick_score, -1, 1))
        if wick_score > 0.15:
            factors.append("Long upper wicks forming into resistance -> sellers actively defending (favors rejection)")
        elif wick_score < -0.15:
            factors.append("Weak wicks / strong up-closes into resistance -> buyers in control (favors breakout)")
        else:
            factors.append("Wick pattern neutral into resistance -> no clear rejection or exhaustion signal yet")

    if _liquidity_sweep_recent(df, level_price, target_kind, n=8):
        score -= 0.3
        factors.append("ICT Liquidity Sweep already detected at this level (stop-hunt + reclaim) -> strong early reversal signal")
    else:
        factors.append("No liquidity sweep detected yet at this level")

    zone_conf = _zone_confluence(ob_list, level_price, atr) + _zone_confluence(fvg_list, level_price, atr)
    if target_kind == 'support':
        score += -0.15 * float(np.clip(zone_conf, -2, 2))
        if zone_conf > 0:
            factors.append("Bullish Order Block / FVG sitting right at this support -> institutional demand zone (favors bounce)")
        elif zone_conf < 0:
            factors.append("Bearish Order Block / FVG stacked at this support -> weak defense (favors breakdown)")
        else:
            factors.append("No Order Block / FVG confluence right at this support -> no institutional footprint detected here")
    else:
        score += 0.15 * float(np.clip(zone_conf, -2, 2))
        if zone_conf > 0:
            factors.append("Bullish Order Block / FVG stacked at this resistance -> buyers overpowering supply (favors breakout)")
        elif zone_conf < 0:
            factors.append("Bearish Order Block / FVG sitting right at this resistance -> institutional supply zone (favors rejection)")
        else:
            factors.append("No Order Block / FVG confluence right at this resistance -> no institutional footprint detected here")

    vol_score = _volume_trend(df, n=5)
    score += 0.2 * vol_score
    if vol_score > 0.15:
        factors.append("Volume expanding into this move -> favors continuation through the level")
    elif vol_score < -0.15:
        factors.append("Volume contracting into this move -> favors exhaustion/reversal at the level")
    else:
        factors.append("Volume trend neutral -> no strong expansion or contraction into this move")

    trend_sign = -1 if target_kind == 'support' else 1
    score += 0.15 * trend_sign * float(np.clip(trend_bias, -1, 1))
    if trend_bias > 0.15:
        factors.append("Broader trend is bullish -> " + ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
    elif trend_bias < -0.15:
        factors.append("Broader trend is bearish -> " + ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
    else:
        factors.append("Broader trend is flat/neutral -> no strong macro tilt either way at this level")

    zone, pos = _premium_discount_position(df)
    pd_score = (0.5 - pos) * 2
    if target_kind == 'support':
        score += -0.1 * pd_score
        factors.append(f"Price is in a {zone} zone (ICT) of the recent range -> " +
                        ("favors demand / bounce at support" if zone == 'discount' else "less support defense expected"))
    else:
        score += 0.1 * pd_score
        factors.append(f"Price is in a {zone} zone (ICT) of the recent range -> " +
                        ("favors supply / rejection at resistance" if zone == 'premium' else "resistance may break easier"))

    oi_result = _oi_pcr_confluence(df_option_chain, level_price)
    if oi_result is not None:
        pcr_bias, call_oi, put_oi = oi_result
        if target_kind == 'support':
            score += -0.25 * pcr_bias
        else:
            score += 0.25 * pcr_bias
        if pcr_bias > 0.15:
            factors.append(f"Heavy Put OI writing near this {target_kind} (Put OI {put_oi:,.0f} vs Call OI {call_oi:,.0f}) -> writers defending, favors bounce/breakout")
        elif pcr_bias < -0.15:
            factors.append(f"Heavy Call OI writing near this {target_kind} (Call OI {call_oi:,.0f} vs Put OI {put_oi:,.0f}) -> writers defending against upside, favors rejection/breakdown")
        else:
            factors.append("Options OI near this level is balanced -> no strong writer-side bias")
    else:
        factors.append("No usable option-chain OI data near this level")

    vwap_bias = _vwap_bias(df, atr)
    if vwap_bias is not None:
        score += 0.15 * trend_sign * vwap_bias
        if vwap_bias > 0.15:
            factors.append("Price above a rising VWAP -> bullish institutional flow -> " +
                            ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif vwap_bias < -0.15:
            factors.append("Price below a falling VWAP -> bearish institutional flow -> " +
                            ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            factors.append("Price is hugging VWAP -> no clear institutional flow bias right now")
    else:
        factors.append("VWAP data unavailable for this check")

    tf_bias = _multi_timeframe_htf_bias(df, atr)
    tf_labels = {'1h': '1-Hour', '15min': '15-Minute', '5min': '5-Minute (base)'}
    tf_available = [v for v in tf_bias.values() if v is not None]
    for tf_key in ('1h', '15min', '5min'):
        b = tf_bias[tf_key]
        label = tf_labels[tf_key]
        if b is None:
            factors.append(f"Not enough history yet for a {label} structure read")
        elif b > 0.15:
            factors.append(f"{label} structure is bullish -> " +
                            ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif b < -0.15:
            factors.append(f"{label} structure is bearish -> " +
                            ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            factors.append(f"{label} structure is flat -> no strong confluence at this timeframe")
    factors.append("1-Minute structure not checked -- base candles here are 5-minute, so 1-min granularity "
                    "isn't available without a separate live fetch")
    if tf_available:
        htf_bias = sum(tf_available) / len(tf_available)
        score += 0.15 * trend_sign * htf_bias

    round_hit = _round_number_confluence(level_price)
    if round_hit is not None:
        strength, round_val = round_hit
        nudge = -0.1 if strength == 'major' else -0.05
        score += nudge
        factors.append(f"Level sits right at a {'major' if strength=='major' else 'minor'} round number (₹{round_val:,.0f}) -> "
                        f"psychological level, extra hesitation/reaction likely here")
    else:
        factors.append("Level is not near a round psychological number -> no extra round-number effect")

    touches = _level_test_count(df, level_price, target_kind, atr)
    if touches >= 2:
        weaken = min(touches - 1, 3) * 0.05
        score += weaken
        factors.append(f"This level has already been tested {touches} times recently -> each test consumes defending orders, favors an eventual break")
    else:
        factors.append(f"This is a relatively fresh level (tested {touches}x recently) -> defending orders still largely intact")

    pa_pattern, pa_bias = _detect_price_action_pattern(df)
    if pa_pattern is not None:
        score += 0.2 * trend_sign * pa_bias
        if pa_bias > 0:
            factors.append(f"Price Action: {pa_pattern} forming -> bullish signal -> " +
                            ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif pa_bias < 0:
            factors.append(f"Price Action: {pa_pattern} forming -> bearish signal -> " +
                            ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            factors.append(f"Price Action: {pa_pattern} forming -> indecision/compression, no clear direction from this candle yet")
    else:
        factors.append("No notable price-action candlestick pattern (Engulfing/Pin Bar/Marubozu/Doji) on the current candle")

    score = float(np.clip(score, -1, 1))
    break_pct = float(np.clip(round(50 + score * 40, 1), 10, 90))
    bounce_pct = round(100 - break_pct, 1)

    if target_kind == 'support':
        directional_bias = "BREAKDOWN LIKELY (bearish) 🔴" if break_pct > bounce_pct else "BOUNCE LIKELY (bullish) 🟢"
    else:
        directional_bias = "BREAKOUT LIKELY (bullish) 🟢" if break_pct > bounce_pct else "REJECTION LIKELY (bearish) 🔴"

    return {
        'break_pct': break_pct, 'bounce_pct': bounce_pct,
        'factors': factors, 'directional_bias': directional_bias,
        'raw_score': round(score, 3)
    }


def _multi_timeframe_htf_bias(df, atr, bars=8):
    """
    NEW — checks higher-timeframe structure at 1-Hour, 15-Minute, AND the
    base 5-Minute candle timeframe (this app's base candles are already
    5-minute -- see market_data.py's interval="5" fetch -- so "5-min HTF"
    is just the base df's own recent trend, no resample needed for that
    one). Returns {'1h': bias_or_None, '15min': bias_or_None, '5min':
    bias_or_None}, each in [-1, 1], same convention as _htf_structure_bias.

    1-Minute is intentionally NOT included here: this app's live candles
    are fetched at 5-minute resolution, so 1-min structure cannot be
    derived from this df -- that granularity simply isn't in the data.
    A real 1-min read would need a SEPARATE live 1-min candle fetch from
    Upstox every refresh cycle (extra API quota use) -- not added here
    without that being an explicit choice, since it costs real quota.
    """
    out = {'1h': _htf_structure_bias(df, atr, resample_rule='1h', bars=bars),
           '15min': _htf_structure_bias(df, atr, resample_rule='15min', bars=bars)}
    if atr:
        recent = df['Close'].tail(bars)
        if len(recent) >= 2:
            slope = float(recent.iloc[-1] - recent.iloc[0])
            out['5min'] = float(np.clip(slope / (atr * 2.0), -1, 1))
        else:
            out['5min'] = None
    else:
        out['5min'] = None
    return out


def predict_round_number_ladder(df, live_price, atr, fvg_list=None, ob_list=None,
                                 trend_bias=0.0, df_option_chain=None,
                                 step=LADDER_STEP, levels_each_side=LADDER_LEVELS_EACH_SIDE):
    """
    NEW — Full round-number ladder calculator.

    Instead of only the single nearest support/resistance, this walks
    EVERY round-number level (every `step` points, default 50 -- so
    100-point levels are included automatically, just marked "major" by
    the existing round-number-confluence check) both above and below the
    live price, and runs the SAME break/bounce confluence model against
    each one. Gives you the small in-between targets ("chhote moksh"),
    each with its own break-vs-bounce % and factor list, using the exact
    same OI/PCR, ICT, volume, price-action inputs as the single-level
    predictor above.

    Returns a dict:
        {
          'live_price': ..., 'step': 50,
          'supports': [ {level_price, distance_pts, break_pct, bounce_pct,
                          directional_bias, factors}, ... ]  # nearest first
          'resistances': [ ... same shape ... ]               # nearest first
        }
    Each list is ordered nearest-to-price first, matching how a trader
    reads a ladder outward from the current price.
    """
    result = {'live_price': live_price, 'step': step, 'supports': [], 'resistances': []}
    if df is None or df.empty or live_price is None or not atr or atr <= 0 or len(df) < 10:
        return result

    fvg_list, ob_list = fvg_list or [], ob_list or []
    base = round(live_price / step) * step

    # Levels BELOW live price -> supports, nearest first
    for i in range(levels_each_side):
        level_price = base - (i * step)
        if level_price >= live_price:
            level_price -= step
        if level_price <= 0:
            continue
        scored = _score_break_bounce_at_level(
            df, live_price, atr, level_price, 'support',
            fvg_list, ob_list, trend_bias, df_option_chain
        )
        result['supports'].append({
            'level_price': round(level_price, 2),
            'distance_pts': round(live_price - level_price, 2),
            **scored
        })

    # Levels ABOVE live price -> resistances, nearest first
    for i in range(levels_each_side):
        level_price = base + (i * step)
        if level_price <= live_price:
            level_price += step
        scored = _score_break_bounce_at_level(
            df, live_price, atr, level_price, 'resistance',
            fvg_list, ob_list, trend_bias, df_option_chain
        )
        result['resistances'].append({
            'level_price': round(level_price, 2),
            'distance_pts': round(level_price - live_price, 2),
            **scored
        })

    # De-duplicate consecutive identical levels that can arise from the
    # base-rounding step above, keep nearest-first ordering intact.
    def _dedupe(levels_list):
        seen = set()
        out_list = []
        for lv in levels_list:
            if lv['level_price'] in seen:
                continue
            seen.add(lv['level_price'])
            out_list.append(lv)
        return out_list

    result['supports'] = sorted(_dedupe(result['supports']), key=lambda x: x['distance_pts'])
    result['resistances'] = sorted(_dedupe(result['resistances']), key=lambda x: x['distance_pts'])
    return result
