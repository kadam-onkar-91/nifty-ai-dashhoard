from app_logging import get_logger
logger = get_logger(__name__)
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
        logger.exception("Broad exception caught; fallback path executed")
        pass

    try:
        swing_highs, swing_lows = _find_swing_points(df.tail(150))
        result['resistances'].extend(swing_highs)
        result['supports'].extend(swing_lows)
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
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
        logger.exception("Broad exception caught; fallback path executed")
        return 0.0


def _parse_price(s):
    try:
        return float(str(s).replace(',', ''))
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
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
        logger.exception("Broad exception caught; fallback path executed")
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
        logger.exception("Broad exception caught; fallback path executed")
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
        logger.exception("Broad exception caught; fallback path executed")
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
        logger.exception("Broad exception caught; fallback path executed")
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
        logger.exception("Broad exception caught; fallback path executed")
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



def _confirmed_level_break(df, level_price, kind, atr):
    """Require a completed-candle boundary break before replacing a locked level.

    A wick through a level is not enough. We require the latest completed close
    to clear the level by a small ATR-adjusted buffer and either a second close
    confirmation or a strong body/volume expansion.
    """
    if df is None or df.empty or level_price is None or not atr or len(df) < 3:
        return False
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last['Close'])
        prev_close = float(prev['Close'])
        rng = max(float(last['High']) - float(last['Low']), 0.01)
        body_ratio = abs(close - float(last['Open'])) / rng
        vol_confirm = False
        if 'Volume' in df.columns and len(df) >= 8:
            recent = float(df['Volume'].tail(3).mean())
            base = float(df['Volume'].tail(8).head(5).mean())
            vol_confirm = base > 0 and recent >= base * 1.20
        buffer = max(0.12 * float(atr), 2.0)
        if kind == 'support':
            clear = close < float(level_price) - buffer
            second = prev_close < float(level_price)
        else:
            clear = close > float(level_price) + buffer
            second = prev_close > float(level_price)
        return bool(clear and (second or body_ratio >= 0.55 or vol_confirm))
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return False


def _level_structural_strength(df, level_price, kind, atr, df_option_chain=None,
                               fvg_list=None, ob_list=None):
    """Score how structurally well-supported a legacy Early-Warning level is.

    This is a *level-strength* score, not a guarantee or a trade win probability.
    It deliberately rewards independent evidence and penalizes repeated tests.
    """
    if level_price is None or not atr:
        return 0.0, ['insufficient data']
    score = 0.0
    reasons = []
    tol = max(2.0, 0.20 * float(atr))

    # Previous completed daily levels are strong anchors.
    try:
        daily = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min'}).iloc[:-1]
        vals = daily.tail(PDLEVELS_LOOKBACK_DAYS)
        col = 'Low' if kind == 'support' else 'High'
        if not vals.empty and any(abs(float(v) - level_price) <= tol for v in vals[col]):
            score += 22.0
            reasons.append('previous-day level')
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        pass

    # Confirmed fractal/swing evidence.
    try:
        sh, sl = _find_swing_points(df.tail(180))
        swings = sl if kind == 'support' else sh
        near = sum(1 for v in swings if abs(float(v) - level_price) <= tol)
        if near:
            score += min(24.0, 12.0 * near)
            reasons.append(f'{near} confirmed swing anchor(s)')
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        near = 0

    # Reaction/touch evidence: fresh is stronger; repeated tests consume liquidity.
    touches = _level_test_count(df, level_price, kind, atr, lookback=120)
    if touches == 0:
        score += 14.0
        reasons.append('fresh level')
    elif touches == 1:
        score += 18.0
        reasons.append('one clean prior reaction')
    elif touches == 2:
        score += 12.0
        reasons.append('two reactions; some defense remains')
    else:
        score += max(2.0, 10.0 - min(touches - 2, 6) * 1.5)
        reasons.append(f'{touches} tests; defense is being consumed')

    # SMC zone confluence.
    zc = _zone_confluence((ob_list or []), level_price, atr) + _zone_confluence((fvg_list or []), level_price, atr)
    if abs(zc) > 0.35:
        score += 12.0
        reasons.append('SMC OB/FVG confluence')

    # Options OI confluence around the level.
    oi = _oi_pcr_confluence(df_option_chain, level_price)
    if oi is not None:
        pcr_bias, call_oi, put_oi = oi
        relevant = pcr_bias > 0.15 if kind == 'support' else pcr_bias < -0.15
        if relevant:
            score += 12.0
            reasons.append('supportive option OI positioning')
        elif abs(pcr_bias) < 0.15:
            score += 3.0
            reasons.append('balanced option OI')

    # Psychological level is only confluence, never the sole anchor.
    if _round_number_confluence(level_price) is not None:
        score += 5.0
        reasons.append('round-number confluence')

    return round(float(np.clip(score, 0.0, 100.0)), 1), reasons


def predict_level_reaction(df, live_price, atr, fvg_list=None, ob_list=None, trend_bias=0.0, df_option_chain=None, locked_support=None, locked_resistance=None):
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

    # Early-Warning levels are session-stable: keep the previously selected
    # support/resistance until a *confirmed* candle break occurs. This prevents
    # the level from jumping on every Streamlit refresh because a new swing
    # candidate appeared. After a confirmed break, promote the next legacy
    # structural level.
    nearest_support = _nearest_level(levels['supports'], live_price, 'below')
    nearest_resistance = _nearest_level(levels['resistances'], live_price, 'above')
    if locked_support is not None:
        if not _confirmed_level_break(df, locked_support, 'support', atr):
            nearest_support = float(locked_support)
    if locked_resistance is not None:
        if not _confirmed_level_break(df, locked_resistance, 'resistance', atr):
            nearest_resistance = float(locked_resistance)
    out['nearest_support'] = nearest_support
    out['nearest_resistance'] = nearest_resistance

    support_strength, support_strength_reasons = _level_structural_strength(
        df, nearest_support, 'support', atr, df_option_chain, fvg_list, ob_list)
    resistance_strength, resistance_strength_reasons = _level_structural_strength(
        df, nearest_resistance, 'resistance', atr, df_option_chain, fvg_list, ob_list)
    out['support_strength_pct'] = support_strength
    out['resistance_strength_pct'] = resistance_strength
    out['support_strength_reasons'] = support_strength_reasons
    out['resistance_strength_reasons'] = resistance_strength_reasons

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
        'level_strength_pct': support_strength if target_kind == 'support' else resistance_strength,
        'level_strength_reasons': support_strength_reasons if target_kind == 'support' else resistance_strength_reasons,
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
                                  fvg_list, ob_list, trend_bias, df_option_chain,
                                  global_avg_change=None, fii_footprint=None,
                                  breadth_advances=None, breadth_declines=None,
                                  banknifty_correlation_note=None,
                                  ml_signal=None, ml_confidence=None, overall_pcr=None):
    """
    Same break-vs-bounce confluence model used inside predict_level_reaction()
    (RSI divergence, wick rejection, liquidity sweep, OB/FVG zone confluence,
    volume trend, broader trend bias, ICT premium/discount, options OI/PCR,
    VWAP bias, 1H/15-min/5-min HTF structure, round-number confluence, level
    test count, price-action pattern) PLUS every other live data source this
    tool has (Global Markets, FII/DII, Market Breadth, Bank Nifty
    correlation, ML ensemble model, overall PCR) -- factored out here so it
    can be run against ANY level_price/kind pair, not just the single
    nearest level.
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

    macro_delta, macro_lines = _global_macro_confluence(
        target_kind, trend_sign, global_avg_change=global_avg_change,
        fii_footprint=fii_footprint, breadth_advances=breadth_advances,
        breadth_declines=breadth_declines, banknifty_correlation_note=banknifty_correlation_note,
        ml_signal=ml_signal, ml_confidence=ml_confidence, overall_pcr=overall_pcr
    )
    score += macro_delta
    factors.extend(macro_lines)

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


def _global_macro_confluence(target_kind, trend_sign, global_avg_change=None,
                              fii_footprint=None, breadth_advances=None,
                              breadth_declines=None, banknifty_correlation_note=None,
                              ml_signal=None, ml_confidence=None, overall_pcr=None):
    """
    NEW — pulls in every OTHER live data source this tool already computes
    elsewhere (Global Markets overnight avg change, FII/DII F&O footprint,
    Nifty internal market breadth, Bank Nifty <-> Nifty correlation, the
    ML ensemble model's own trained prediction, and the overall
    market-wide option-chain PCR) as EXTRA confluence for each level's
    break/bounce score -- on top of the per-level OI-band/ICT/price-action
    factors already used. Each factor gets a modest weight; the combined
    score is still clipped to [-1,1] downstream, so this refines rather
    than overrides the existing read. Same sign convention as every other
    factor here: positive bias = bullish, and multiplying by trend_sign
    automatically flips it correctly for support vs resistance.
    """
    delta = 0.0
    lines = []

    if global_avg_change is not None:
        g = float(np.clip(global_avg_change / 1.0, -1, 1))  # ~1% overnight move = full tilt
        delta += 0.08 * trend_sign * g
        if g > 0.15:
            lines.append(f"Global markets overnight avg +{global_avg_change:.2f}% (risk-on) -> " +
                         ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif g < -0.15:
            lines.append(f"Global markets overnight avg {global_avg_change:.2f}% (risk-off) -> " +
                         ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            lines.append("Global markets overnight roughly flat -> no strong macro tilt")
    else:
        lines.append("Global market data not available for this check")

    if fii_footprint:
        f_up = fii_footprint.upper()
        if "BULLISH" in f_up:
            delta += 0.10 * trend_sign
            lines.append(f"FII/DII F&O footprint: {fii_footprint} -> " +
                         ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif "BEARISH" in f_up:
            delta -= 0.10 * trend_sign
            lines.append(f"FII/DII F&O footprint: {fii_footprint} -> " +
                         ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            lines.append(f"FII/DII F&O footprint: {fii_footprint} -> neutral, no strong bias")
    else:
        lines.append("FII/DII footprint data not available for this check")

    if breadth_advances is not None and breadth_declines is not None and (breadth_advances + breadth_declines) > 0:
        b = (breadth_advances - breadth_declines) / (breadth_advances + breadth_declines)
        delta += 0.08 * trend_sign * b
        if b > 0.15:
            lines.append(f"Market breadth skewed bullish ({breadth_advances} up / {breadth_declines} down) -> " +
                         ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif b < -0.15:
            lines.append(f"Market breadth skewed bearish ({breadth_advances} up / {breadth_declines} down) -> " +
                         ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            lines.append(f"Market breadth roughly balanced ({breadth_advances} up / {breadth_declines} down) -> no strong bias")
    else:
        lines.append("Market breadth data not available for this check")

    if banknifty_correlation_note:
        note_up = banknifty_correlation_note.upper()
        if "DIVERGENCE WARNING" in note_up:
            delta -= 0.05 * trend_sign
            lines.append(f"Bank Nifty divergence flagged ({banknifty_correlation_note}) -> weakens conviction in the current move continuing")
        elif "CONFIRMED" in note_up:
            delta += 0.05 * trend_sign
            lines.append(f"Bank Nifty confirms Nifty's move ({banknifty_correlation_note}) -> adds conviction to the current move")
        else:
            lines.append(f"Bank Nifty <-> Nifty correlation: {banknifty_correlation_note}")
    else:
        lines.append("Bank Nifty correlation data not available for this check")

    if ml_signal is not None and ml_confidence is not None and ml_signal != 0:
        ml_bullish = ml_signal > 0
        conf = float(np.clip(ml_confidence, 0, 1))
        delta += 0.15 * trend_sign * (1.0 if ml_bullish else -1.0) * conf
        direction_word = "BULLISH" if ml_bullish else "BEARISH"
        if ml_bullish:
            read = "favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"
        else:
            read = "favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"
        lines.append(f"ML ensemble model predicts {direction_word} with {conf*100:.0f}% confidence -> {read}")
    else:
        lines.append("ML ensemble model has no confident directional prediction right now")

    if overall_pcr is not None:
        pcr_bias = float(np.clip((overall_pcr - 1.0), -1, 1))
        delta += 0.08 * trend_sign * pcr_bias
        if pcr_bias > 0.15:
            lines.append(f"Overall market-wide Option Chain PCR {overall_pcr:.2f} (put-heavy) -> " +
                         ("favors support holding (bounce)" if target_kind == 'support' else "favors resistance breaking (breakout)"))
        elif pcr_bias < -0.15:
            lines.append(f"Overall market-wide Option Chain PCR {overall_pcr:.2f} (call-heavy) -> " +
                         ("favors support breaking (breakdown)" if target_kind == 'support' else "favors resistance holding (rejection)"))
        else:
            lines.append(f"Overall market-wide Option Chain PCR {overall_pcr:.2f} -> balanced, no strong bias")
    else:
        lines.append("Overall market-wide PCR not available for this check")

    return delta, lines


def predict_round_number_ladder(df, live_price, atr, fvg_list=None, ob_list=None,
                                 trend_bias=0.0, df_option_chain=None,
                                 step=LADDER_STEP, levels_each_side=LADDER_LEVELS_EACH_SIDE,
                                 global_avg_change=None, fii_footprint=None,
                                 breadth_advances=None, breadth_declines=None,
                                 banknifty_correlation_note=None,
                                 ml_signal=None, ml_confidence=None, overall_pcr=None):
    """
    NEW — Full round-number ladder calculator.

    Instead of only the single nearest support/resistance, this walks
    EVERY round-number level (every `step` points, default 50 -- so
    100-point levels are included automatically, just marked "major" by
    the existing round-number-confluence check) both above and below the
    live price, and runs the SAME break/bounce confluence model against
    each one. Gives you the small in-between targets ("chhote moksh"),
    each with its own break-vs-bounce % and factor list, using the same
    OI/PCR, ICT, volume, price-action inputs as the single-level predictor
    above -- PLUS, if passed in, every other live data source this tool
    computes elsewhere: Global Markets overnight change, FII/DII F&O
    footprint, Nifty internal market breadth, Bank Nifty <-> Nifty
    correlation, the ML ensemble model's prediction, and the overall
    market-wide option-chain PCR. All the new params are OPTIONAL --
    if not passed, the ladder works exactly as before, just without that
    extra confluence.

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
    macro_kwargs = dict(
        global_avg_change=global_avg_change, fii_footprint=fii_footprint,
        breadth_advances=breadth_advances, breadth_declines=breadth_declines,
        banknifty_correlation_note=banknifty_correlation_note,
        ml_signal=ml_signal, ml_confidence=ml_confidence, overall_pcr=overall_pcr
    )

    # Levels BELOW live price -> supports, nearest first
    for i in range(levels_each_side):
        level_price = base - (i * step)
        if level_price >= live_price:
            level_price -= step
        if level_price <= 0:
            continue
        scored = _score_break_bounce_at_level(
            df, live_price, atr, level_price, 'support',
            fvg_list, ob_list, trend_bias, df_option_chain, **macro_kwargs
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
            fvg_list, ob_list, trend_bias, df_option_chain, **macro_kwargs
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

# ---------------------------------------------------------------------------
# COMPREHENSIVE S/R ENGINE (production layer)
# ---------------------------------------------------------------------------
# This layer is intentionally separate from the legacy early-warning scorer.
# It builds price zones from independent evidence and exposes a location-first
# gate for the trade decision engine.  It never invents missing option/order-
# flow/volume-profile data.

SR_MAX_ZONES = 12
SR_DEFAULT_TICK = 0.05
SR_MIN_REACTIONS = 2
SR_MAJOR_MIN_SCORE = 2.8
SR_MAJOR_MIN_DISTANCE_ATR = 0.35
SR_ZONE_MAX_WIDTH_ATR = 0.22


def _safe_float(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return None


def _adaptive_tolerance(df, atr):
    atr_v = _safe_float(atr) or 10.0
    # A previous version allowed up to 8 points of tolerance. That could
    # merge several unrelated intraday pivots into fake 5-15 point S/R
    # boxes. Keep the clustering tolerance materially tighter.
    return max(SR_DEFAULT_TICK * 4, min(atr_v * 0.08, 3.0))


def _swing_points_confirmed(df, lookback=5, max_points=80):
    """Confirmed swings only: the last `lookback` candles are not used as future data."""
    highs, lows = [], []
    if df is None or df.empty or len(df) < (2 * lookback + 3):
        return highs, lows
    w = df.tail(500)
    h = pd.to_numeric(w['High'], errors='coerce').to_numpy()
    l = pd.to_numeric(w['Low'], errors='coerce').to_numpy()
    for i in range(lookback, len(w) - lookback):
        if np.isfinite(h[i]) and h[i] >= np.nanmax(h[i-lookback:i+lookback+1]):
            highs.append(float(h[i]))
        if np.isfinite(l[i]) and l[i] <= np.nanmin(l[i-lookback:i+lookback+1]):
            lows.append(float(l[i]))
    return highs[-max_points:], lows[-max_points:]


def _cluster_prices(prices, tolerance):
    vals = sorted([p for p in (_safe_float(x) for x in prices) if p is not None])
    if not vals:
        return []
    clusters = [[vals[0]]]
    for p in vals[1:]:
        if p - np.mean(clusters[-1]) <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.mean(c)) for c in clusters]


def _daily_weekly_monthly_levels(df):
    out = []
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return out
    try:
        for freq, prefix in [('D', 'PD'), ('W', 'PW'), ('ME', 'PM')]:
            g = df.resample(freq).agg({'High':'max','Low':'min','Close':'last'}).dropna()
            if len(g) >= 2:
                prev = g.iloc[-2]
                out += [
                    (float(prev['High']), f'{prefix}H', 'resistance'),
                    (float(prev['Low']), f'{prefix}L', 'support'),
                    (float(prev['Close']), f'{prefix}C', 'neutral'),
                ]
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        pass
    return out


def _today_session_levels(df):
    out = []
    try:
        today = df.index[-1].date()
        d = df[df.index.date == today]
        if not d.empty:
            out += [(float(d['High'].max()), 'Session High', 'resistance'),
                    (float(d['Low'].min()), 'Session Low', 'support')]
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        pass
    return out


def _opening_range_levels(df, minutes=15):
    try:
        today = df.index[-1].date()
        d = df[df.index.date == today]
        if d.empty:
            return []
        start = d.index[0]
        window = d[d.index <= start + pd.Timedelta(minutes=minutes)]
        if window.empty:
            return []
        return [(float(window['High'].max()), 'Opening Range High', 'resistance'),
                (float(window['Low'].min()), 'Opening Range Low', 'support')]
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return []


def _psych_levels(price, step=100):
    if price is None or step <= 0:
        return []
    base = int(round(price / step) * step)
    return [(float(base + i * step), f'Psychological {base + i*step}', 'neutral') for i in range(-3, 4)]


def _extract_volume_profile_levels(df, price, atr):
    out = []
    # Prefer columns already calculated by the dashboard.
    for col, label, typ in [
        ('POC_Level','POC','neutral'), ('VAH','VAH','resistance'), ('VAL','VAL','support'),
        ('HVN','HVN','neutral'), ('LVN','LVN','neutral')]:
        if col in df.columns:
            v = _safe_float(df[col].iloc[-1])
            if v is not None:
                out.append((v, f'Volume Profile {label}', typ))
    # If only OHLCV exists, estimate a volume-by-price profile without fabricating
    # a live feed: distribute each completed bar's volume at its typical price.
    if not out and 'Volume' in df.columns:
        try:
            x = df[['High','Low','Close','Volume']].tail(300).copy()
            x = x.apply(pd.to_numeric, errors='coerce').dropna()
            x = x[x['Volume'] > 0]
            if len(x) >= 30:
                lo, hi = float(x['Low'].min()), float(x['High'].max())
                bins = max(20, min(80, int((hi-lo) / max((atr or 10)*0.15, 1))))
                if hi > lo and bins > 1:
                    idx = np.linspace(lo, hi, bins+1)
                    tp = (x['High'] + x['Low'] + x['Close']) / 3.0
                    hist, edges = np.histogram(tp, bins=idx, weights=x['Volume'])
                    centers = (edges[:-1] + edges[1:]) / 2
                    order = np.argsort(hist)[::-1]
                    poc = float(centers[order[0]])
                    out.append((poc, 'Estimated Volume Profile POC (OHLCV)', 'neutral'))
                    total = hist.sum()
                    if total > 0:
                        cum_order = np.argsort(np.abs(centers-poc))
                        acc = 0.0; chosen=[]
                        for j in cum_order:
                            acc += hist[j]; chosen.append(j)
                            if acc >= total*0.70: break
                        out.append((float(centers[min(chosen)]), 'Estimated Value Area', 'neutral'))
        except Exception:
            logger.exception("Broad exception caught; fallback path executed")
            pass
    return out


def _option_chain_levels(df_option_chain, price, atr):
    """Use verified chain columns only; return OI/OI-change/volume/IV evidence."""
    out = []
    if df_option_chain is None or getattr(df_option_chain, 'empty', True):
        return out, {'status':'UNAVAILABLE'}
    try:
        c = df_option_chain.copy()
        strike_col = next((x for x in ['Strike','strike','strikePrice','strike_price'] if x in c.columns), None)
        if not strike_col:
            return out, {'status':'UNAVAILABLE','reason':'Strike column missing'}
        c['_strike'] = pd.to_numeric(c[strike_col], errors='coerce')
        c = c[c['_strike'].notna()]
        c = c[(c['_strike'] >= price - max(atr*3, 500)) & (c['_strike'] <= price + max(atr*3, 500))]
        if c.empty:
            return out, {'status':'UNAVAILABLE','reason':'No strikes near spot'}
        def col(names):
            return next((n for n in names if n in c.columns), None)
        call_oi = col(['Call OI','call_oi','CE_OI','ce_oi','CallOI'])
        put_oi = col(['Put OI','put_oi','PE_OI','pe_oi','PutOI'])
        call_ch = col(['Call OI Change','call_oi_change','CE_OI_Change','ce_oi_change'])
        put_ch = col(['Put OI Change','put_oi_change','PE_OI_Change','pe_oi_change'])
        vol = col(['Volume','volume'])
        iv = col(['IV','iv','Implied Volatility'])
        for _, r in c.iterrows():
            s = float(r['_strike'])
            co = _safe_float(r[call_oi]) if call_oi else None
            po = _safe_float(r[put_oi]) if put_oi else None
            if co is None and po is None:
                continue
            out.append((s, 'Option Chain OI', 'resistance' if (co or 0) > (po or 0) else 'support'))
        near = c.iloc[(c['_strike']-price).abs().argsort()[:15]]
        max_call = float(pd.to_numeric(near[call_oi], errors='coerce').max()) if call_oi else None
        max_put = float(pd.to_numeric(near[put_oi], errors='coerce').max()) if put_oi else None
        oi_info = {'status':'LIVE_OR_LOADED', 'rows':len(c), 'call_oi_col':call_oi, 'put_oi_col':put_oi,
                   'call_oi_max':max_call, 'put_oi_max':max_put,
                   'oi_change_available': bool(call_ch and put_ch),
                   'volume_available': bool(vol), 'iv_available': bool(iv)}
        return out, oi_info
    except Exception as exc:
        logger.exception("Broad exception caught; fallback path executed")
        return [], {'status':'UNAVAILABLE','reason':f'{type(exc).__name__}: {exc}'}


def _zone_reactions(df, lo, hi):
    try:
        h = pd.to_numeric(df['High'], errors='coerce'); l = pd.to_numeric(df['Low'], errors='coerce')
        touches = int(((h >= lo) & (l <= hi)).sum())
        return touches
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return 0


def _sr_source_family(source):
    """Map raw evidence to an independent evidence family."""
    s = str(source).lower()
    if 'swing high' in s or 'swing low' in s:
        if 'htf' in s:
            return 'htf_swing'
        if 'mtf' in s:
            return 'mtf_swing'
        return 'ltf_swing'
    if s.startswith('pd'):
        return 'daily'
    if s.startswith('pw'):
        return 'weekly'
    if s.startswith('pm'):
        return 'monthly'
    if 'session' in s:
        return 'session'
    if 'opening range' in s:
        return 'opening_range'
    if 'cpr/pivot' in s:
        return 'pivot'
    if 'vwap' in s:
        return 'vwap'
    if 'volume profile' in s:
        return 'volume_profile'
    if 'order block' in s:
        return 'smc_ob'
    if 'fvg' in s:
        return 'smc_fvg'
    if 'equal high' in s or 'equal low' in s or 'liquidity' in s:
        return 'liquidity'
    if 'option chain' in s:
        return 'options'
    return 'other'


def _sr_is_anchor_family(family):
    return family in {
        'htf_swing', 'mtf_swing', 'daily', 'weekly', 'monthly',
        'session', 'opening_range', 'pivot', 'smc_ob', 'liquidity'
    }


def build_comprehensive_sr_context(df, live_price, atr=None, df_option_chain=None,
                                   fvg_list=None, ob_list=None,
                                   liquidity_map=None, pivots=None):
    """Build a conservative, location-first S/R map.

    Important design rule: proximity alone NEVER creates a major support or
    resistance. Micro swings, VWAP, psychological numbers and individual
    option strikes are evidence/confluence, not standalone trade levels.
    A zone becomes actionable only when independent evidence supports it.
    """
    out = {
        'status': 'INSUFFICIENT_DATA', 'live_price': live_price, 'zones': [],
        'nearest_support': None, 'nearest_resistance': None,
        'major_support': None, 'major_resistance': None,
        'location': 'UNKNOWN', 'option_chain': {'status': 'UNAVAILABLE'},
        'sources': {}, 'decision_note': None
    }
    price = _safe_float(live_price)
    atr_v = _safe_float(atr) or 15.0
    if df is None or df.empty or price is None:
        return out

    tol = _adaptive_tolerance(df, atr_v)
    candidates = []

    def add(v, source, typ='neutral', weight=1.0):
        v = _safe_float(v)
        if v is None or v <= 0:
            return
        family = _sr_source_family(source)
        candidates.append({
            'price': v, 'source': str(source), 'type': str(typ),
            'weight': float(weight), 'family': family,
            'anchor': _sr_is_anchor_family(family)
        })

    # 1) Confirmed swings. These are structural evidence, not automatically
    # major levels. LTF swings deliberately carry less weight than HTF swings.
    for rule, mult, w in [(5, 'LTF', 0.8), (10, 'MTF', 1.15), (20, 'HTF', 1.5)]:
        sh, sl = _swing_points_confirmed(df, rule)
        for v in sh:
            add(v, f'{mult} Swing High', 'resistance', w)
        for v in sl:
            add(v, f'{mult} Swing Low', 'support', w)

    # 2) Completed higher-period levels.
    for v, src, typ in _daily_weekly_monthly_levels(df):
        # Completed daily levels are deliberately strong anchors. Previous
        # day high/low and especially the previous completed daily close are
        # not treated like an ordinary intraday swing. They receive extra
        # structural weight and can materially strengthen a nearby zone.
        if src.startswith('PD'):
            base_w = 2.15 if src.endswith('C') else 2.05
        elif src.startswith('PW'):
            base_w = 1.55
        else:
            base_w = 1.45
        add(v, src, typ, base_w)

    # 3) Today's session and opening range. The opening-range high/low is a
    # first-class intraday reference: when independent evidence clusters
    # around it, it should be allowed to form a strong S/R zone rather than
    # being diluted by minor swings.
    for v, src, typ in _today_session_levels(df):
        add(v, src, typ, 1.45)
    for v, src, typ in _opening_range_levels(df):
        add(v, src, typ, 1.75)

    # 4) CPR/pivots are valid structural anchors when the pivot engine is OK.
    if pivots and pivots.get('status') == 'OK':
        for k, typ in [('pivot','neutral'), ('tc','resistance'), ('bc','support'),
                       ('r1','resistance'), ('r2','resistance'), ('r3','resistance'),
                       ('s1','support'), ('s2','support'), ('s3','support')]:
            add(pivots.get(k), f'CPR/Pivot {k.upper()}', typ, 1.15)

    # 5) VWAP and volume-profile are useful confluence but MUST NOT create a
    # standalone major S/R zone.
    if 'VWAP' in df.columns:
        add(_safe_float(df['VWAP'].iloc[-1]), 'Session VWAP', 'neutral', 0.55)
    for v, src, typ in _extract_volume_profile_levels(df, price, atr_v):
        add(v, src, typ, 0.65)

    # 6) SMC zones: OB is stronger than a single FVG. Still require an
    # independent anchor/reaction before calling it a major level.
    for z in (fvg_list or []):
        p = _parse_price(z.get('Price')) if isinstance(z, dict) else None
        if p is not None:
            typ = 'support' if 'bull' in str(z.get('Type','')).lower() else 'resistance'
            add(p, 'SMC FVG', typ, 0.85)
    for z in (ob_list or []):
        p = _parse_price(z.get('Price')) if isinstance(z, dict) else None
        if p is not None:
            typ = 'support' if 'bull' in str(z.get('Type','')).lower() else 'resistance'
            add(p, 'SMC Order Block', typ, 1.25)

    # 7) Liquidity pools are important structural evidence.
    if liquidity_map:
        for x in liquidity_map.get('levels', []):
            add(x.get('price'), x.get('name', 'Liquidity'), x.get('type', 'neutral'), 1.2)
        for x in liquidity_map.get('equal_highs', []):
            add(x.get('level'), 'Equal High Liquidity', 'resistance', 1.45)
        for x in liquidity_map.get('equal_lows', []):
            add(x.get('level'), 'Equal Low Liquidity', 'support', 1.45)

    # 8) Option strikes are context only. Do not turn every 50-point strike
    # into a chart S/R level; OI is consumed as confluence by the decision
    # engine instead.
    opt_levels, opt_info = _option_chain_levels(df_option_chain, price, atr_v)
    out['option_chain'] = opt_info

    # Cluster only actual structural candidates (not option strikes). This is
    # the key fix for the false 23392-23397 / 23397-23402 pair.
    raw_prices = [x['price'] for x in candidates]
    clusters = _cluster_prices(raw_prices, tol)
    zones = []
    for center in clusters:
        members = [x for x in candidates if abs(x['price'] - center) <= tol]
        if not members:
            continue
        lo = max(0.0, min(x['price'] for x in members) - tol)
        hi = max(x['price'] for x in members) + tol
        side = 'support' if center < price else ('resistance' if center > price else 'current')
        families = sorted(set(x['family'] for x in members))
        sources = sorted(set(x['source'] for x in members))
        anchor_families = [f for f in families if _sr_is_anchor_family(f)]
        touches = _zone_reactions(df, lo, hi)
        score = sum(x['weight'] for x in members)
        score += min(touches, 6) * 0.30
        score += min(len(families), 4) * 0.35
        # Extra structural premium for the references the trader explicitly
        # wants treated as strong: completed daily close, previous-day H/L,
        # and the opening-range boundaries. This is only a strength bonus;
        # it never makes proximity alone sufficient for a trade.
        if any(str(x['source']).startswith('PDC') for x in members):
            score += 0.75
        if any('Opening Range' in str(x['source']) for x in members):
            score += 0.50
        if any(str(x['source']).startswith(('PDH', 'PDL')) for x in members):
            score += 0.45

        strong_anchor = bool(anchor_families)
        multi_source = len(families) >= 2
        repeated = touches >= SR_MIN_REACTIONS
        width = hi - lo
        width_ok = width <= max(2.0, atr_v * SR_ZONE_MAX_WIDTH_ATR)
        major_score = score >= SR_MAJOR_MIN_SCORE
        quality_ok = bool(width_ok and major_score and
                          (strong_anchor and (multi_source or repeated) or
                           len(families) >= 3 or touches >= 3))

        distance = abs(price - center)
        too_close = distance < atr_v * SR_MAJOR_MIN_DISTANCE_ATR
        # A very close zone is allowed only when it is independently strong.
        micro = bool(too_close and not (quality_ok and score >= 4.5 and len(families) >= 2))
        actionable = bool(quality_ok and not micro and side in ('support', 'resistance'))

        strength = 'STRONG' if score >= 5.5 else ('MODERATE' if score >= 3.5 else 'WEAK')
        zones.append({
            'price': round(center, 2), 'low': round(lo, 2), 'high': round(hi, 2),
            'side': side, 'distance_pts': round(distance, 2), 'touches': touches,
            'strength': strength, 'score': round(score, 2), 'sources': sources,
            'source_families': families, 'anchor_families': anchor_families,
            'width_pts': round(width, 2), 'quality_ok': quality_ok,
            'micro_zone': micro, 'actionable': actionable
        })

    zones = sorted(zones, key=lambda z: z['distance_pts'])

    # Select ONLY actionable zones for the trade engine. Weak/micro zones stay
    # visible for audit but can never redefine the market location.
    actionable = [z for z in zones if z['actionable']]
    supports = [z for z in actionable if z['side'] == 'support' and z['high'] < price]
    resistances = [z for z in actionable if z['side'] == 'resistance' and z['low'] > price]

    # Prevent overlapping/adjacent fake S/R boxes around the current price.
    ns = supports[0] if supports else None
    nr = resistances[0] if resistances else None
    if ns and nr and ns['high'] >= nr['low']:
        # When two independently generated zones overlap, neither is a clean
        # directional boundary. Keep the stronger one only for audit, but do
        # not declare the current price AT_SUPPORT/AT_RESISTANCE.
        stronger = ns if (ns['score'], len(ns['source_families'])) >= (nr['score'], len(nr['source_families'])) else nr
        if stronger['side'] == 'support':
            nr = None
        else:
            ns = None
        overlap = True
    else:
        overlap = False

    out['zones'] = zones[:SR_MAX_ZONES]
    out['nearest_support'] = ns
    out['nearest_resistance'] = nr
    out['major_support'] = ns
    out['major_resistance'] = nr
    out['overlapping_zone_warning'] = overlap
    out['micro_zones_ignored'] = sum(1 for z in zones if z['micro_zone'])

    # Location is based ONLY on major/actionable zones. If both boundaries
    # are too close/unclear, explicitly say so instead of inventing support.
    if ns and price <= ns['high'] + max(1.0, tol):
        location = 'AT_SUPPORT'
    elif nr and price >= nr['low'] - max(1.0, tol):
        location = 'AT_RESISTANCE'
    elif ns and nr and ns['high'] < price < nr['low']:
        span = max(nr['low'] - ns['high'], 0.01)
        rel = (price - ns['high']) / span
        location = 'MIDRANGE' if 0.25 <= rel <= 0.75 else ('NEAR_SUPPORT' if rel < 0.5 else 'NEAR_RESISTANCE')
    else:
        location = 'BETWEEN_LEVELS'
    out['location'] = location

    # Completed-candle confirmation. A single wick is not enough for a
    # breakout/breakdown. Strong body or volume must support the close.
    recent = df.tail(3)
    last = recent.iloc[-1]
    close = _safe_float(last.get('Close'))
    high = _safe_float(last.get('High'))
    low = _safe_float(last.get('Low'))
    open_ = _safe_float(last.get('Open'))
    body = abs(close - open_) if close is not None and open_ is not None else 0.0
    candle_range = max((high - low) if high is not None and low is not None else 0.0, 0.01)
    body_ratio = body / candle_range
    vol_ok = False
    if 'Volume' in df.columns and len(df) >= 10:
        v = pd.to_numeric(df['Volume'], errors='coerce').tail(20)
        base = _safe_float(v.iloc[:-1].mean())
        last_v = _safe_float(v.iloc[-1])
        vol_ok = bool(last_v and base and last_v > base * 1.15)
    bullish_close = bool(close is not None and open_ is not None and close > open_)
    bearish_close = bool(close is not None and open_ is not None and close < open_)

    breakout = bool(nr and close is not None and close > nr['high'] and
                    (vol_ok or (bullish_close and body_ratio >= 0.55)))
    breakdown = bool(ns and close is not None and close < ns['low'] and
                     (vol_ok or (bearish_close and body_ratio >= 0.55)))
    support_reject = bool(ns and low is not None and low <= ns['high'] and close is not None and
                          close > ns['high'] and bullish_close and body_ratio >= 0.25)
    resistance_reject = bool(nr and high is not None and high >= nr['low'] and close is not None and
                             close < nr['low'] and bearish_close and body_ratio >= 0.25)

    # EARLY MOMENTUM BREAK: detect a decisive completed-candle failure of the
    # immediately preceding move. This is intentionally earlier than waiting
    # for a second retest candle, but much stricter than a single red/green
    # candle. It is used only by the full-context reversal path in the trade
    # engine, which requires additional order-flow/MTF/regime/structure checks.
    bearish_momentum_break = False
    bullish_momentum_break = False
    reversal_break_level = None
    if len(df) >= 4:
        prev2 = df.iloc[-3]
        prev1 = df.iloc[-2]
        p2c = _safe_float(prev2.get('Close'))
        p1c = _safe_float(prev1.get('Close'))
        p1h = _safe_float(prev1.get('High'))
        p1l = _safe_float(prev1.get('Low'))
        if close is not None and open_ is not None and high is not None and low is not None:
            close_pos = (close - low) / max(high - low, 0.01)
            bearish_momentum_break = bool(
                p2c is not None and p1c is not None and p1c > p2c and
                p1h is not None and p1l is not None and close < p1l and
                bearish_close and body_ratio >= 0.45 and close_pos <= 0.35
            )
            bullish_momentum_break = bool(
                p2c is not None and p1c is not None and p1c < p2c and
                p1l is not None and p1h is not None and close > p1h and
                bullish_close and body_ratio >= 0.45 and close_pos >= 0.65
            )
            if bearish_momentum_break:
                reversal_break_level = p1l
            elif bullish_momentum_break:
                reversal_break_level = p1h

    # ENTRY-TIMING / ADVERSE-MOVE RISK: detect when a directional signal is
    # technically still bullish/bearish but the move is already losing quality
    # at the exact proposed entry. This addresses the recurring failure mode
    # where BUY is taken at the top of a rising leg or SELL at the bottom of a
    # falling leg. It is a veto/quality flag, never a trade trigger by itself.
    buy_entry_risk = False
    sell_entry_risk = False
    buy_entry_risk_reasons = []
    sell_entry_risk_reasons = []
    if len(df) >= 5 and all(_safe_float(df.iloc[i].get('Close')) is not None for i in range(-5, 0)):
        closes = [_safe_float(df.iloc[i].get('Close')) for i in range(-5, 0)]
        opens = [_safe_float(df.iloc[i].get('Open')) for i in range(-5, 0)]
        highs = [_safe_float(df.iloc[i].get('High')) for i in range(-5, 0)]
        lows = [_safe_float(df.iloc[i].get('Low')) for i in range(-5, 0)]
        d = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        # A move can remain green/red while momentum deteriorates. Compare the
        # latest impulse with the immediately preceding impulse and inspect
        # candle body/wick quality.
        if d[0] > 0 and d[1] > 0 and d[2] > 0 and d[3] > 0:
            if d[3] < max(0.0, 0.65 * d[2]) and closes[-1] >= closes[-2]:
                buy_entry_risk_reasons.append('uptrend momentum is decelerating')
            last_body = abs(closes[-1] - opens[-1]) if opens[-1] is not None else 0.0
            last_range = max(highs[-1] - lows[-1], 0.01)
            upper_wick = max(0.0, highs[-1] - max(opens[-1], closes[-1]))
            if upper_wick / last_range >= 0.35 and last_body / last_range < 0.45:
                buy_entry_risk_reasons.append('upper-wick rejection/weak close')
        if d[0] < 0 and d[1] < 0 and d[2] < 0 and d[3] < 0:
            if abs(d[3]) < 0.65 * abs(d[2]) and closes[-1] <= closes[-2]:
                sell_entry_risk_reasons.append('downtrend momentum is decelerating')
            last_body = abs(closes[-1] - opens[-1]) if opens[-1] is not None else 0.0
            last_range = max(highs[-1] - lows[-1], 0.01)
            lower_wick = max(0.0, min(opens[-1], closes[-1]) - lows[-1])
            if lower_wick / last_range >= 0.35 and last_body / last_range < 0.45:
                sell_entry_risk_reasons.append('lower-wick rejection/weak close')
        # If price has already travelled unusually far from the recent 5-bar
        # mean, demand stronger confirmation rather than buying/selling the
        # late extension. ATR is used only when available below.
        recent_mean = sum(closes[:-1]) / max(len(closes[:-1]), 1)
        extension = abs(closes[-1] - recent_mean)
        if extension > 0:
            buy_entry_risk = len(buy_entry_risk_reasons) >= 2
            sell_entry_risk = len(sell_entry_risk_reasons) >= 2

    # TREND-TRANSITION GUARD: detect an early turn against the immediately
    # preceding move. This is used only as a safety veto by the trade engine;
    # it does not by itself create a trade.
    bearish_trend_transition = False
    bullish_trend_transition = False
    if len(df) >= 4:
        c2 = _safe_float(df.iloc[-3].get('Close'))
        c1 = _safe_float(df.iloc[-2].get('Close'))
        c0 = close
        if c2 is not None and c1 is not None and c0 is not None:
            move = abs(c1 - c2)
            bearish_trend_transition = bool(
                c1 > c2 and c0 < c1 and bearish_close and body_ratio >= 0.35 and
                c0 <= c1 - max(0.10, 0.10 * move)
            )
            bullish_trend_transition = bool(
                c1 < c2 and c0 > c1 and bullish_close and body_ratio >= 0.35 and
                c0 >= c1 + max(0.10, 0.10 * move)
            )

    out['entry_timing'] = {
        'buy_adverse_move_risk': bool(buy_entry_risk),
        'sell_adverse_move_risk': bool(sell_entry_risk),
        'buy_reasons': buy_entry_risk_reasons,
        'sell_reasons': sell_entry_risk_reasons,
    }

    # A first candle closing through resistance/support can be a false
    # breakout. For a BUY above resistance (or SELL below support), require
    # a separate retest/hold candle: SOME recently-completed candle must
    # have already closed beyond the boundary, while the latest candle
    # retests the old zone and closes back on the accepted side. This
    # prevents the engine from buying the first spike into resistance.
    #
    # Lookback widened from "exactly the immediately previous candle" to the
    # last 8 completed candles. The 1-candle version only ever fired if the
    # retest happened on the very next candle after the breakout -- in real
    # price action the pullback/retest often comes 3-6 candles later, so a
    # genuine breakout-then-hold (e.g. resistance broken, price runs a bit
    # further, then pulls back and holds above the old level) was being
    # rejected purely because of that narrow timing window, not because the
    # setup was actually invalid. The retest/hold itself must still happen
    # on the CURRENT candle -- only the "was there a real breakout recently"
    # check was widened.
    breakout_retest = False
    breakdown_retest = False
    if len(df) >= 3 and nr:
        recent = df.iloc[max(0, len(df) - 9):-1]
        had_recent_breakout = any(
            (c := _safe_float(row.get('Close'))) is not None and c > nr['high']
            for _, row in recent.iterrows()
        )
        cur_low = low
        breakout_retest = bool(
            had_recent_breakout and
            cur_low is not None and cur_low <= nr['high'] + max(1.0, tol) and
            close is not None and close > nr['high'] and bullish_close
        )
    if len(df) >= 3 and ns:
        recent = df.iloc[max(0, len(df) - 9):-1]
        had_recent_breakdown = any(
            (c := _safe_float(row.get('Close'))) is not None and c < ns['low']
            for _, row in recent.iterrows()
        )
        cur_high = high
        breakdown_retest = bool(
            had_recent_breakdown and
            cur_high is not None and cur_high >= ns['low'] - max(1.0, tol) and
            close is not None and close < ns['low'] and bearish_close
        )

    out['confirmation'] = {
        'breakout_confirmed': breakout, 'breakdown_confirmed': breakdown,
        'breakout_retest_confirmed': breakout_retest,
        'breakdown_retest_confirmed': breakdown_retest,
        'support_rejection_confirmed': support_reject,
        'resistance_rejection_confirmed': resistance_reject,
        'bearish_momentum_break_confirmed': bearish_momentum_break,
        'bullish_momentum_break_confirmed': bullish_momentum_break,
        'bearish_trend_transition_confirmed': bearish_trend_transition,
        'bullish_trend_transition_confirmed': bullish_trend_transition,
        'reversal_break_level': round(reversal_break_level, 2) if reversal_break_level is not None else None,
        'volume_expansion': vol_ok, 'latest_close': close,
        'latest_body_ratio': round(body_ratio, 3),
        'confirmation_rule': 'support/resistance reaction uses completed candles; breakouts require a close beyond the zone plus a separate retest/hold before an entry is allowed'
    }

    out['sources'] = {
        'swing': 'CONFIRMED_SWINGS',
        'classic_levels': 'PD/PW/PM + session + opening range',
        'cpr_pivots': 'AVAILABLE' if pivots and pivots.get('status') == 'OK' else 'UNAVAILABLE',
        'vwap': 'CONFLUENCE_ONLY' if 'VWAP' in df.columns else 'UNAVAILABLE',
        'volume_profile': 'CONFLUENCE_ONLY' if _extract_volume_profile_levels(df, price, atr_v) else 'UNAVAILABLE',
        'smc': 'FVG/OB INPUTS',
        'liquidity': 'LIQUIDITY MAP INPUT' if liquidity_map else 'UNAVAILABLE',
        'option_chain': opt_info.get('status', 'UNAVAILABLE'),
        'psychological_levels': 'CONFLUENCE_ONLY_NOT_STANDALONE'
    }
    if not ns and not nr:
        out['decision_note'] = ('No strong multi-source support/resistance zone is validated near price. '
                                'Micro swings, VWAP, option strikes and psychological numbers are being ignored as standalone S/R.')
    elif overlap:
        out['decision_note'] = ('Conflicting/overlapping zones detected around current price. '
                                'Directional S/R location is withheld until a clean boundary is established.')
    else:
        out['decision_note'] = ('Major S/R selected from independent structural evidence; weak nearby micro-levels are excluded.')

    out['status'] = 'OK'
    return out

