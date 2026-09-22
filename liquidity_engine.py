from app_logging import get_logger
logger = get_logger(__name__)
import pandas as pd
import numpy as np

"""
LIQUIDITY ENGINE (Phase 1)
------------------------------------------
A dedicated liquidity map (spec section 13): previous day/week high-low,
session high-low, and equal-highs/equal-lows clustering (resting stop
liquidity). Pure price-action math off the OHLC data this app already
has -- no new data source needed.
"""

EQUAL_LEVEL_TOLERANCE_ATR_MULT = 0.15   # how close two swing points must be to count as "equal"


def _session_bounds(df):
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None, None
    try:
        today = df.index[-1].date()
        today_df = df[df.index.date == today]
        if today_df.empty:
            return None, None
        return float(today_df['High'].max()), float(today_df['Low'].min())
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return None, None


def _prev_period_levels(df, freq):
    """Generic helper: previous COMPLETED period's (day/week) high & low."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None, None
    try:
        grouped = df.resample(freq).agg({'High': 'max', 'Low': 'min'}).dropna()
        if len(grouped) < 2:
            return None, None
        prev = grouped.iloc[-2]  # -1 is the still-forming current period
        return float(prev['High']), float(prev['Low'])
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return None, None


def _cluster_equal_levels(points, tol):
    """Groups nearby swing points into clusters (2+ touches within tol =
    'equal highs/lows' -- classic resting liquidity)."""
    if not points:
        return []
    pts = sorted(points)
    clusters = []
    current = [pts[0]]
    for p in pts[1:]:
        if abs(p - current[-1]) <= tol:
            current.append(p)
        else:
            if len(current) >= 2:
                clusters.append(current)
            current = [p]
    if len(current) >= 2:
        clusters.append(current)
    return [{'level': round(float(np.mean(c)), 2), 'touches': len(c)} for c in clusters]


def _find_swings(df, lookback=15, tail_n=150):
    if df is None or len(df) < (lookback * 2 + 1):
        return [], []
    work = df.tail(tail_n)
    h, l = work['High'].values, work['Low'].values
    highs, lows = [], []
    for i in range(lookback, len(work) - lookback):
        if h[i] == h[i - lookback:i + lookback + 1].max():
            highs.append(float(h[i]))
        if l[i] == l[i - lookback:i + lookback + 1].min():
            lows.append(float(l[i]))
    return highs, lows


def build_liquidity_map(df, live_price, atr):
    """
    Returns the full liquidity map: PDH/PDL, PWH/PWL, session high/low,
    equal-highs/equal-lows clusters, each with distance from current
    price -- sorted nearest-first so the closest liquidity pool is
    obvious at a glance.
    """
    out = {'status': 'INSUFFICIENT DATA', 'levels': [], 'equal_highs': [], 'equal_lows': []}
    if df is None or df.empty or live_price is None:
        return out

    levels = []
    pdh, pdl = _prev_period_levels(df, 'D')
    if pdh is not None:
        levels.append({'name': 'PDH (Prev Day High)', 'price': round(pdh, 2), 'type': 'resistance'})
        levels.append({'name': 'PDL (Prev Day Low)', 'price': round(pdl, 2), 'type': 'support'})

    pwh, pwl = _prev_period_levels(df, 'W')
    if pwh is not None:
        levels.append({'name': 'PWH (Prev Week High)', 'price': round(pwh, 2), 'type': 'resistance'})
        levels.append({'name': 'PWL (Prev Week Low)', 'price': round(pwl, 2), 'type': 'support'})

    sess_high, sess_low = _session_bounds(df)
    if sess_high is not None:
        levels.append({'name': "Today's Session High", 'price': round(sess_high, 2), 'type': 'resistance'})
        levels.append({'name': "Today's Session Low", 'price': round(sess_low, 2), 'type': 'support'})

    for lv in levels:
        lv['distance_pts'] = round(abs(lv['price'] - live_price), 2)
        lv['side'] = 'above' if lv['price'] > live_price else 'below'

    levels.sort(key=lambda x: x['distance_pts'])

    # Equal highs / equal lows -- resting liquidity clusters
    swing_highs, swing_lows = _find_swings(df)
    tol = max(0.15 * (atr or 10), 3.0) if atr else 5.0
    equal_highs = _cluster_equal_levels(swing_highs, tol)
    equal_lows = _cluster_equal_levels(swing_lows, tol)
    for e in equal_highs:
        e['distance_pts'] = round(abs(e['level'] - live_price), 2)
    for e in equal_lows:
        e['distance_pts'] = round(abs(e['level'] - live_price), 2)
    equal_highs.sort(key=lambda x: x['distance_pts'])
    equal_lows.sort(key=lambda x: x['distance_pts'])

    out.update({'status': 'OK', 'levels': levels, 'equal_highs': equal_highs, 'equal_lows': equal_lows})
    return out


def nearest_liquidity_target(liquidity_map, live_price):
    """One-line summary of the single closest resting-liquidity pool --
    the level price is most likely to be drawn toward next (classic ICT
    'liquidity target' concept)."""
    if liquidity_map.get('status') != 'OK':
        return "Not enough data for a liquidity read yet."
    candidates = list(liquidity_map['levels'])
    for e in liquidity_map['equal_highs']:
        candidates.append({'name': f"Equal Highs ({e['touches']}x touched)", 'price': e['level'],
                            'type': 'resistance', 'distance_pts': e['distance_pts'],
                            'side': 'above' if e['level'] > live_price else 'below'})
    for e in liquidity_map['equal_lows']:
        candidates.append({'name': f"Equal Lows ({e['touches']}x touched)", 'price': e['level'],
                            'type': 'support', 'distance_pts': e['distance_pts'],
                            'side': 'above' if e['level'] > live_price else 'below'})
    if not candidates:
        return "No liquidity pools detected nearby."
    nearest = min(candidates, key=lambda x: x['distance_pts'])
    return (f"Nearest liquidity target: {nearest['name']} at ₹{nearest['price']:,.2f} "
            f"({nearest['distance_pts']:.1f} pts {nearest['side']})")
