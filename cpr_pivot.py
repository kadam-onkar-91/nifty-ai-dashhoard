from app_logging import get_logger
logger = get_logger(__name__)
import pandas as pd
import numpy as np

"""
CPR / PIVOT / SESSION ENGINE (Phase 1)
------------------------------------------
Classic floor-trader pivots (Pivot, R1-R3, S1-S3), Central Pivot Range
(CPR: Pivot/TC/BC) with narrow-vs-wide classification, and Opening
Range breakout/breakdown detection (spec section 15). Pure price-action
math off the previous day's OHLC -- no new data source needed.
"""

CPR_WIDTH_LOOKBACK_DAYS = 20   # how many past days' CPR widths to compare against
OPENING_RANGE_MINUTES = 15


def _prev_day_ohlc(df):
    """Previous COMPLETED trading day's High/Low/Close from the intraday df."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None
    try:
        daily = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
        daily = daily.iloc[:-1]  # drop today's still-forming session
        if daily.empty:
            return None
        last_day = daily.iloc[-1]
        return float(last_day['High']), float(last_day['Low']), float(last_day['Close']), daily
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return None


def calculate_pivots(df):
    """
    Classic floor-trader pivot points + CPR (Central Pivot Range) for
    TODAY, computed from the previous day's High/Low/Close.
    Returns a dict, or a dict with 'status': 'INSUFFICIENT DATA' if
    there isn't at least one completed prior session in the data.
    """
    prev = _prev_day_ohlc(df)
    if prev is None:
        return {'status': 'INSUFFICIENT DATA'}
    pdh, pdl, pdc, daily_hist = prev

    pivot = (pdh + pdl + pdc) / 3.0
    bc = (pdh + pdl) / 2.0          # Bottom Central pivot
    tc = (pivot - bc) + pivot        # Top Central pivot
    if tc < bc:
        tc, bc = bc, tc

    r1 = (2 * pivot) - pdl
    s1 = (2 * pivot) - pdh
    r2 = pivot + (pdh - pdl)
    s2 = pivot - (pdh - pdl)
    r3 = pdh + 2 * (pivot - pdl)
    s3 = pdl - 2 * (pdh - pivot)

    cpr_width = abs(tc - bc)

    # Classify this CPR as narrow/wide vs its own recent history
    width_pctl = None
    width_label = "UNKNOWN"
    try:
        if len(daily_hist) >= 5:
            hist = daily_hist.tail(CPR_WIDTH_LOOKBACK_DAYS + 1).copy()
            hist_pivot = (hist['High'] + hist['Low'] + hist['Close']) / 3.0
            hist_bc = (hist['High'] + hist['Low']) / 2.0
            hist_tc = (hist_pivot - hist_bc) + hist_pivot
            hist_width = (hist_tc - hist_bc).abs().iloc[:-1]  # exclude today's own (not in hist anyway)
            if len(hist_width) >= 5:
                width_pctl = float((hist_width < cpr_width).mean() * 100)
                width_label = "NARROW (trend-day potential)" if width_pctl <= 30 else (
                    "WIDE (range-day likely)" if width_pctl >= 70 else "NORMAL")
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        pass

    return {
        'status': 'OK',
        'pivot': round(pivot, 2), 'tc': round(tc, 2), 'bc': round(bc, 2), 'cpr_width': round(cpr_width, 2),
        'cpr_width_percentile': round(width_pctl, 1) if width_pctl is not None else None,
        'cpr_label': width_label,
        'r1': round(r1, 2), 'r2': round(r2, 2), 'r3': round(r3, 2),
        's1': round(s1, 2), 's2': round(s2, 2), 's3': round(s3, 2),
        'pdh': round(pdh, 2), 'pdl': round(pdl, 2), 'pdc': round(pdc, 2),
    }


def cpr_position(live_price, pivots):
    """Where is price relative to today's CPR right now, and has it
    broken out of / rejected from it?"""
    if pivots.get('status') != 'OK' or live_price is None:
        return "INSUFFICIENT DATA"
    tc, bc = pivots['tc'], pivots['bc']
    if live_price > tc:
        return "ABOVE CPR (bullish bias while it holds)"
    if live_price < bc:
        return "BELOW CPR (bearish bias while it holds)"
    return "INSIDE CPR (balanced / wait for a break)"


def opening_range(df, minutes=OPENING_RANGE_MINUTES):
    """
    Today's Opening Range (first N minutes of the session) high/low, and
    whether the current price has broken out above or below it.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return {'status': 'INSUFFICIENT DATA'}
    try:
        today = df.index[-1].date()
        today_df = df[df.index.date == today]
        if today_df.empty:
            return {'status': 'INSUFFICIENT DATA'}
        session_start = today_df.index[0]
        window_end = session_start + pd.Timedelta(minutes=minutes)
        orb_candles = today_df[today_df.index <= window_end]
        if orb_candles.empty:
            return {'status': 'INSUFFICIENT DATA'}
        orb_high = float(orb_candles['High'].max())
        orb_low = float(orb_candles['Low'].min())
        live_price = float(today_df['Close'].iloc[-1])

        if len(today_df) <= len(orb_candles):
            status = "STILL FORMING (opening range not complete yet)"
        elif live_price > orb_high:
            status = "OPENING RANGE BREAKOUT (bullish)"
        elif live_price < orb_low:
            status = "OPENING RANGE BREAKDOWN (bearish)"
        else:
            status = "INSIDE OPENING RANGE"

        return {'status': 'OK', 'orb_high': round(orb_high, 2), 'orb_low': round(orb_low, 2),
                'orb_minutes': minutes, 'read': status}
    except Exception as e:
        logger.exception("Broad exception caught; fallback path executed")
        return {'status': 'INSUFFICIENT DATA', 'error': f"{type(e).__name__}: {e}"}
