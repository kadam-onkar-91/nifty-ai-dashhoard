import pandas as pd
import numpy as np
from datetime import datetime
import indicators

"""
MARKET REGIME DETECTION ENGINE (Phase 1)
------------------------------------------
Classifies the CURRENT market condition so the rest of the system can
adapt its logic instead of using one fixed strategy in every condition
(spec section 9). Built entirely from data this app already has live:
OHLCV candles (for ADX/ATR/structure) and India VIX (already fetched
live in global_markets.py) -- no new data source required.

Honesty rules followed:
- If VIX isn't available, volatility-regime read falls back to an
  ATR-percentile-only method and says so explicitly.
- Never invents an "EVENT/NEWS RISK" flag without a real signal for it
  (that flag is left to the News/Event engine, Phase 2, which has an
  actual calendar -- this module doesn't guess it from price alone).
"""

ADX_TREND_THRESHOLD = 22.0     # ADX above this = market is trending, not ranging
ATR_PCTL_LOOKBACK = 100        # candles used to build the ATR percentile distribution
HIGH_VOL_PCTL = 75             # ATR percentile above this = high volatility regime
LOW_VOL_PCTL = 25              # ATR percentile below this = low volatility regime
VIX_HIGH = 17.0                # India VIX above this = elevated fear/volatility
VIX_LOW = 12.0                 # India VIX below this = complacent/low volatility


def _atr_percentile(df, lookback=ATR_PCTL_LOOKBACK):
    """Where does today's ATR sit vs its own recent history? Returns a
    0-100 percentile, or None if there isn't enough history yet."""
    if 'ATR' not in df.columns or len(df) < 20:
        return None
    recent = df['ATR'].tail(lookback).dropna()
    if len(recent) < 20:
        return None
    current = recent.iloc[-1]
    return float((recent < current).mean() * 100)


def _is_expiry_day(today=None):
    """Nifty weekly expiry is currently Tuesday (NSE changed this more
    than once historically) -- flagged here as a best-effort heuristic,
    not authoritative. Always double-check the actual NSE expiry
    calendar before relying on this for real decisions."""
    today = today or datetime.now()
    return today.weekday() == 1  # Tuesday = 1


def detect_regime(df, live_vix=None):
    """
    Returns a dict:
      - primary: one of TRENDING BULLISH / TRENDING BEARISH / RANGE
      - volatility: HIGH VOLATILITY / LOW VOLATILITY / NORMAL
      - structure: BREAKOUT / BREAKDOWN / EXPANSION / CONTRACTION / NEUTRAL
      - expiry_regime: bool (best-effort, see _is_expiry_day)
      - adx, plus_di, minus_di, atr_percentile, vix
      - notes: list[str] explaining each read
    """
    out = {
        'primary': 'INSUFFICIENT DATA', 'volatility': 'INSUFFICIENT DATA',
        'structure': 'INSUFFICIENT DATA', 'expiry_regime': _is_expiry_day(),
        'adx': None, 'plus_di': None, 'minus_di': None,
        'atr_percentile': None, 'vix': live_vix, 'notes': []
    }
    if df is None or df.empty or len(df) < 30:
        out['notes'].append("Not enough candle history yet for a reliable regime read.")
        return out

    try:
        adx, plus_di, minus_di = indicators.calculate_adx(df)
        adx_now, pdi_now, mdi_now = float(adx.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])
        out.update({'adx': round(adx_now, 1), 'plus_di': round(pdi_now, 1), 'minus_di': round(mdi_now, 1)})

        # --- Primary regime: trending vs range, and which direction ---
        if adx_now >= ADX_TREND_THRESHOLD:
            if pdi_now > mdi_now:
                out['primary'] = 'TRENDING BULLISH'
            else:
                out['primary'] = 'TRENDING BEARISH'
            out['notes'].append(f"ADX {adx_now:.1f} >= {ADX_TREND_THRESHOLD} -> genuine trend, "
                                 f"+DI {pdi_now:.1f} vs -DI {mdi_now:.1f}")
        else:
            out['primary'] = 'RANGE'
            out['notes'].append(f"ADX {adx_now:.1f} < {ADX_TREND_THRESHOLD} -> no directional trend, market is range-bound")

        # --- Volatility regime: ATR percentile, cross-checked with VIX if available ---
        atr_pctl = _atr_percentile(df)
        out['atr_percentile'] = round(atr_pctl, 1) if atr_pctl is not None else None

        vol_votes = []
        if atr_pctl is not None:
            if atr_pctl >= HIGH_VOL_PCTL:
                vol_votes.append('HIGH')
            elif atr_pctl <= LOW_VOL_PCTL:
                vol_votes.append('LOW')
            else:
                vol_votes.append('NORMAL')
            out['notes'].append(f"ATR is at the {atr_pctl:.0f}th percentile of its own last {ATR_PCTL_LOOKBACK} candles")
        if live_vix is not None:
            if live_vix >= VIX_HIGH:
                vol_votes.append('HIGH')
            elif live_vix <= VIX_LOW:
                vol_votes.append('LOW')
            else:
                vol_votes.append('NORMAL')
            out['notes'].append(f"India VIX is {live_vix:.2f}")
        else:
            out['notes'].append("India VIX unavailable right now -- volatility read is ATR-only")

        if not vol_votes:
            out['volatility'] = 'INSUFFICIENT DATA'
        elif 'HIGH' in vol_votes:
            out['volatility'] = 'HIGH VOLATILITY'
        elif 'LOW' in vol_votes:
            out['volatility'] = 'LOW VOLATILITY'
        else:
            out['volatility'] = 'NORMAL'

        # --- Structure regime: expansion/contraction from ATR trend, breakout/breakdown from recent range ---
        atr_series = df['ATR'].tail(20).dropna()
        if len(atr_series) >= 10:
            atr_slope = float(atr_series.iloc[-1] - atr_series.iloc[0])
            expanding = atr_slope > 0
        else:
            expanding = None

        recent_range = df.tail(30)
        range_high, range_low = recent_range['High'].max(), recent_range['Low'].min()
        close_now = float(df['Close'].iloc[-1])
        if close_now >= range_high * 0.999 and out['primary'] == 'TRENDING BULLISH':
            out['structure'] = 'BREAKOUT'
        elif close_now <= range_low * 1.001 and out['primary'] == 'TRENDING BEARISH':
            out['structure'] = 'BREAKDOWN'
        elif expanding is True:
            out['structure'] = 'EXPANSION'
        elif expanding is False:
            out['structure'] = 'CONTRACTION'
        else:
            out['structure'] = 'NEUTRAL'

        if out['expiry_regime']:
            out['notes'].append("Best-effort expiry-day flag is ON (heuristic: Tuesday) -- verify against the actual NSE expiry calendar")

    except Exception as e:
        out['notes'].append(f"Regime detection error: {type(e).__name__}: {e}")

    return out


def regime_adjusted_guidance(regime):
    """
    Spec requirement: 'the strategy engine must adapt according to
    regime, do not use the same logic in every regime'. This returns a
    short, honest text explaining how THIS regime should change how you
    read every other signal on the dashboard -- not a new trade signal.
    """
    if regime['primary'] == 'INSUFFICIENT DATA':
        return "Not enough data yet to give regime-based guidance."

    lines = []
    if regime['primary'] == 'RANGE':
        lines.append("Market is RANGE-BOUND (low ADX): breakout/breakdown signals are more likely to be fake here -- "
                      "mean-reversion at support/resistance is more reliable than trend-following right now.")
    else:
        lines.append(f"Market is {regime['primary']} (ADX {regime['adx']}): trend-following / breakout signals "
                      "carry more weight here than counter-trend mean-reversion calls.")

    if regime['volatility'] == 'HIGH VOLATILITY':
        lines.append("Volatility is HIGH: widen stop-losses, expect bigger whipsaws, reduce position size.")
    elif regime['volatility'] == 'LOW VOLATILITY':
        lines.append("Volatility is LOW: tighter ranges expected, but also watch for a sudden expansion (compression often precedes a big move).")

    if regime['structure'] == 'BREAKOUT':
        lines.append("Price is breaking out of its recent range WITH trend confirmation -- higher-conviction continuation setup.")
    elif regime['structure'] == 'BREAKDOWN':
        lines.append("Price is breaking down out of its recent range WITH trend confirmation -- higher-conviction continuation setup.")
    elif regime['structure'] == 'CONTRACTION':
        lines.append("Range is contracting -- a volatility expansion (in either direction) may be coming soon.")

    if regime['expiry_regime']:
        lines.append("Best-effort expiry-day flag is on -- option positioning/OI can behave unusually into expiry, weight OI-based signals with extra caution today.")

    return " ".join(lines)
