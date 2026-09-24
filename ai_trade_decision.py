"""
ai_trade_decision.py  --  v13 "PLAYBOOK ENGINE"

Why this was rewritten
----------------------
The old engine was a chain of ~15 HARD vetoes (choppy market, lagging main
signal == 0, "context incomplete", "no validated S/R zone", "MIDRANGE / BETWEEN
LEVELS", entry-timing flag, confirmation flags ...).  Any ONE of them returned
"no setup", so on a fast fall (like the 79-pt drop to 23,163) there was often
no validated support below price -> "BETWEEN_LEVELS" -> no trade, and after
the fall the main signal lagged -> no BUY at the bounce either.  There was
also a real bug (`_early_entry_ok` used before assignment).

How it works now
----------------
The engine evaluates FOUR playbooks on BOTH sides, every refresh, independent
of the lagging main signal:

    BOUNCE_BUY      price at/near support after a fall + stabilisation
    REJECTION_SELL  price at/near resistance after a rise + rejection
    BREAKDOWN_SELL  price breaks/under support with bearish momentum
    BREAKOUT_BUY    price breaks/over resistance with bullish momentum

Each candidate gets a 0-100 evidence score (location, price-action trigger,
trend/regime, order-flow, breadth/global, extras) minus soft penalties for
conflicts.  The best candidate above ENTRY_SCORE_MIN becomes the trade; the
existing self-learning layer (trade_learning.compute_confidence) then turns
the factor flags into an honest confidence.  Only truly unsafe conditions
(stale data, no price/ATR, cooldown after a fresh loss) block a trade.

Every return value (trade or no-trade) also carries a `market_view`:
where price can fall to, where it can bounce from, where it can rise to.

This module never reports a confidence above trade_learning.CONFIDENCE_CEILING.
"""
from app_logging import get_logger
logger = get_logger(__name__)

import re
from datetime import datetime

import trade_learning

ENTRY_SCORE_MIN = 46.0      # minimum evidence score for CONTINUATION plays (0-100)
REVERSAL_SCORE_MIN = 44.0   # minimum evidence score for BOUNCE/REJECTION plays (they need level + candle both)
MIN_CONFIDENCE = 50.0       # minimum learned/blended confidence
MIN_RR = 1.2                # minimum reward:risk for a structural target
FALLBACK_RR = 1.5           # reward:risk used when no structural target fits
LOSS_COOLDOWN_MIN = 10      # minutes to wait before re-entering the SAME side after a LOSS

REQUIRED_CONTEXT_SECTIONS = (
    "Market Status", "Data Freshness", "Nifty Spot Price",
    "RAW Option Chain (all loaded strikes)", "RAW Full NIFTY 50 Breadth/Stock Data",
    "RAW Global Market Table", "RAW Global News/Sentiment Table",
    "RAW Multi-Timeframe Structure", "RAW SMC Zones / FVG / OB / Sweeps",
    "RAW Liquidity Map", "RAW S/R Ladder", "RAW Comprehensive S/R Zones", "RAW Regime Engine",
    "RAW CPR/Pivots/Opening Range", "RAW Order Flow", "RAW ML Results",
    "RAW Backtest Report", "RAW Monte Carlo Report", "RAW Model Drift Report",
    "RAW Position Sizing", "RAW Risk Engine", "RAW Sniper Setup", "RAW Hybrid AI Analysis",
)

PLAYBOOK_LABELS = {
    "BOUNCE_BUY": "Support Bounce (BUY)",
    "REJECTION_SELL": "Resistance Rejection (SELL)",
    "BREAKDOWN_SELL": "Breakdown Continuation (SELL)",
    "BREAKOUT_BUY": "Breakout Continuation (BUY)",
}


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def _round_to_strike(price, step=50):
    return int(round(price / step) * step)


def _sf(v, default=None):
    try:
        if v is None:
            return default
        x = float(v)
        if x != x:  # NaN
            return default
        return x
    except Exception:
        return default


def _flatten_context_text(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items()).lower()
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_context_text(x) for x in value).lower()
    return str(value).lower()


def _extract_order_flow(context):
    raw = context.get("RAW Order Flow")
    text = _flatten_context_text(raw).upper()
    pressure = ""
    if "BUYING PRESSURE" in text:
        pressure = "BUYING PRESSURE"
    elif "SELLING PRESSURE" in text:
        pressure = "SELLING PRESSURE"
    return pressure, None


def _extract_regime(context):
    """Returns (bias, structure) e.g. ('BEARISH', 'BREAKDOWN')."""
    text = str(context.get("RAW Regime Engine") or "")
    m = re.search(r"'primary':\s*'([^']+)'", text)
    primary = (m.group(1).upper() if m else "")
    s = re.search(r"'structure':\s*'([^']+)'", text)
    structure = (s.group(1).upper() if s else "")
    if "BULLISH" in primary:
        return "BULLISH", structure
    if "BEARISH" in primary:
        return "BEARISH", structure
    if "RANGE" in primary:
        return "RANGE", structure
    low = text.lower()
    if "bullish" in low and "bearish" not in low:
        return "BULLISH", structure
    if "bearish" in low and "bullish" not in low:
        return "BEARISH", structure
    return ("RANGE" if "range" in low else "UNKNOWN"), structure


def _extract_mtf(context):
    text = _flatten_context_text(context.get("RAW Multi-Timeframe Structure"))
    if not text:
        return "UNKNOWN"
    bull = text.count("bullish")
    bear = text.count("bearish")
    if bull > bear and bull >= 2:
        return "BULLISH"
    if bear > bull and bear >= 2:
        return "BEARISH"
    return "MIXED"


def _find_best_ladder_level(level_ladder, live_price, atr, direction, max_atr_mult=1.6, preferred_side=None):
    """Kept for backward compatibility (older callers). Not used by the v13 engine."""
    if not level_ladder or not atr:
        return None
    candidates = (level_ladder.get('supports') or []) + (level_ladder.get('resistances') or [])
    best = None
    for lvl in candidates:
        if preferred_side and str(lvl.get('approaching', '')).lower() != str(preferred_side).lower():
            continue
        if lvl.get('distance_pts', 1e9) > max_atr_mult * atr:
            continue
        bias_lower = lvl.get('directional_bias', '').lower()
        if not (("bullish" in bias_lower and direction == "BUY") or ("bearish" in bias_lower and direction == "SELL")):
            continue
        if best is None or lvl['distance_pts'] < best['distance_pts']:
            best = lvl
    if best is None:
        return None
    approaching = 'support' if best['level_price'] < live_price else 'resistance'
    return {'status': 'APPROACHING_LEVEL', 'approaching': approaching,
            'level_price': best['level_price'], 'distance_pts': best['distance_pts'],
            'break_pct': best['break_pct'], 'bounce_pct': best['bounce_pct'],
            'directional_bias': best['directional_bias'], 'factors': best.get('factors', [])}


# ---------------------------------------------------------------------
# price-action features (from the live candle dataframe)
# ---------------------------------------------------------------------
def _price_features(df, live, atr):
    f = {"ok": False}
    try:
        if df is None or len(df) < 15:
            return f
        d = df.tail(150)
        c = d['Close'].astype(float).values
        h = d['High'].astype(float).values
        l = d['Low'].astype(float).values
        o = d['Open'].astype(float).values

        f["ret3"] = (c[-1] - c[-4]) / atr
        f["ret6"] = (c[-1] - c[-7]) / atr
        f["ret12"] = (c[-1] - c[-13]) / atr if len(c) >= 13 else f["ret6"]
        f["drop_from_high"] = (h[-12:].max() - live) / atr
        f["rise_from_low"] = (live - l[-12:].min()) / atr
        f["low5"] = float(l[-5:].min())
        f["high5"] = float(h[-5:].max())

        try:
            idx = d.index
            sess = d[idx.normalize() == idx[-1].normalize()]
        except Exception:
            sess = d.tail(60)
        if len(sess) < 3:
            sess = d.tail(60)
        f["sess_low"] = float(sess['Low'].astype(float).min())
        f["sess_high"] = float(sess['High'].astype(float).max())
        f["dist_sess_low"] = (live - f["sess_low"]) / atr
        f["dist_sess_high"] = (f["sess_high"] - live) / atr

        rng = max(h[-1] - l[-1], 1e-9)
        f["bull"] = bool(c[-1] > o[-1])
        f["bear"] = bool(c[-1] < o[-1])
        f["body"] = abs(c[-1] - o[-1]) / rng
        f["close_pos"] = (c[-1] - l[-1]) / rng
        f["lower_wick"] = (min(o[-1], c[-1]) - l[-1]) / rng
        f["upper_wick"] = (h[-1] - max(o[-1], c[-1])) / rng
        rng_p = max(h[-2] - l[-2], 1e-9)
        f["lower_wick_prev"] = (min(o[-2], c[-2]) - l[-2]) / rng_p
        f["upper_wick_prev"] = (h[-2] - max(o[-2], c[-2])) / rng_p

        f["ll6"] = int(sum(l[-i] < l[-i - 1] for i in range(1, 7)))
        f["hh6"] = int(sum(h[-i] > h[-i - 1] for i in range(1, 7)))

        prior_fall = (c[-4] - c[-10]) / atr if len(c) >= 10 else 0.0
        f["stall_after_fall"] = bool(abs(c[-1] - c[-4]) <= 0.45 * atr and prior_fall <= -0.9)
        f["stall_after_rise"] = bool(abs(c[-1] - c[-4]) <= 0.45 * atr and prior_fall >= 0.9)

        last, prev = d.iloc[-1], d.iloc[-2]
        for k in ("EMA_20", "EMA_50", "VWAP", "RSI", "MACD_Hist", "POC_Level"):
            f[k] = _sf(last.get(k))
        f["rsi_prev"] = _sf(prev.get("RSI"))
        f["macd_prev"] = _sf(prev.get("MACD_Hist"))
        f["ok"] = True
    except Exception:
        logger.exception("price feature extraction failed")
        f["ok"] = False
    return f


def _swing_levels(df, atr, live):
    """Recent confirmed swing highs/lows (3 candles each side)."""
    out = []
    try:
        d = df.tail(140)
        h = d['High'].astype(float).values
        l = d['Low'].astype(float).values
        n = len(d)
        for i in range(3, n - 3):
            if l[i] == min(l[i - 3:i + 4]):
                out.append((float(l[i]), "swing low"))
            if h[i] == max(h[i - 3:i + 4]):
                out.append((float(h[i]), "swing high"))
    except Exception:
        pass
    return out


def _collect_levels(live, atr, df, feats, level_prediction, level_ladder, sr_context):
    raw = []

    def add(price, strength, src, dynamic=False):
        p = _sf(price)
        if p is None or p <= 0:
            return
        raw.append({"price": p, "strength": float(strength), "src": src, "dynamic": dynamic})

    # 1) validated multi-source S/R zones (strong evidence)
    if isinstance(sr_context, dict):
        for z in (sr_context.get("zones") or []):
            if not isinstance(z, dict):
                continue
            base = 2.0 if z.get("actionable") else (1.0 if z.get("strength") in ("MODERATE", "STRONG") else 0.6)
            if z.get("strength") == "STRONG":
                base += 0.8
            elif z.get("strength") == "MODERATE":
                base += 0.3
            src = "S/R zone"
            if z.get("sources"):
                src = "S/R: " + ", ".join(str(x) for x in list(z["sources"])[:2])
            add(z.get("price"), min(base, 3.0), src)
        for k in ("major_support", "major_resistance"):
            z = sr_context.get(k)
            if isinstance(z, dict):
                add(z.get("price"), 2.8, "major " + k.split("_")[1])

    # 2) legacy swing-based level predictor
    if isinstance(level_prediction, dict) and level_prediction.get("level_price"):
        add(level_prediction.get("level_price"), 1.6, "key swing level")

    # 3) every-50pt ladder
    if isinstance(level_ladder, dict):
        for lvl in (level_ladder.get("supports") or []) + (level_ladder.get("resistances") or []):
            if not isinstance(lvl, dict):
                continue
            pct = max(_sf(lvl.get("bounce_pct"), 50.0), _sf(lvl.get("break_pct"), 50.0))
            lp = _sf(lvl.get("level_price"))
            st = 0.6 + (0.6 if pct >= 60 else 0.0) + (0.4 if lp is not None and lp % 100 == 0 else 0.0)
            add(lp, st, "round-number level")

    # 4) price-action levels from the candles themselves
    if feats.get("ok"):
        add(feats["sess_low"], 1.6, "today's low")
        add(feats["sess_high"], 1.6, "today's high")
        for k, name, st in (("VWAP", "VWAP", 1.0), ("EMA_20", "EMA20", 0.8),
                            ("EMA_50", "EMA50", 1.0), ("POC_Level", "volume POC", 0.8)):
            add(feats.get(k), st, name, dynamic=True)
    try:
        for price, name in _swing_levels(df, atr, live):
            add(price, 1.0, name)
    except Exception:
        pass

    # cluster nearby levels (within 0.2 ATR) -> stronger single level
    raw.sort(key=lambda x: x["price"])
    merged = []
    tol = 0.2 * atr
    for lv in raw:
        if merged and abs(lv["price"] - merged[-1]["price"]) <= tol:
            m = merged[-1]
            n = m["count"] + 1
            m["price"] = (m["price"] * m["count"] + lv["price"]) / n
            m["strength"] = max(m["strength"], lv["strength"]) + 0.35
            if lv["src"] not in m["src"]:
                m["src"] = (m["src"] + " + " + lv["src"]) if len(m["src"]) < 60 else m["src"]
            m["dynamic"] = m["dynamic"] and lv["dynamic"]
            m["count"] = n
        else:
            merged.append({**lv, "count": 1})
    for m in merged:
        m["strength"] = round(min(m["strength"], 3.5), 2)
        m["price"] = round(m["price"], 2)
        m["side"] = "support" if m["price"] < live else "resistance"
    return merged


# ---------------------------------------------------------------------
# evidence scoring for one candidate
# ---------------------------------------------------------------------
def _score_candidate(play, d, X):
    """
    play: BOUNCE_BUY | REJECTION_SELL | BREAKDOWN_SELL | BREAKOUT_BUY
    d: +1 BUY / -1 SELL
    X: shared context dict
    Returns dict(valid, score, loc, pa, parts, reasons, misses, level) or None if the
    playbook simply does not apply right now.
    """
    live, atr, f = X["live"], X["atr"], X["f"]
    conf = X["conf"]
    levels = X["levels"]
    reasons, misses = [], []
    reversal = play in ("BOUNCE_BUY", "REJECTION_SELL")

    # ---------------- LOCATION ----------------
    loc_pts, lvl = 0.0, None
    if reversal:
        want = "support" if d == 1 else "resistance"
        cands = []
        for lv in levels:
            if lv["dynamic"] and lv["strength"] < 1.0:
                continue
            if d == 1:
                inside = (live - 0.7 * atr) <= lv["price"] <= (live + 0.3 * atr)
            else:
                inside = (live - 0.3 * atr) <= lv["price"] <= (live + 0.7 * atr)
            if inside:
                cands.append((lv["strength"] * 4.0 - abs(lv["price"] - live) / atr * 3.0, lv))
        if not cands:
            return None
        lvl = max(cands, key=lambda t: t[0])[1]
        loc_pts = min(25.0, 8.0 + 5.0 * min(lvl["strength"], 3.0))
        reasons.append(f"{want.capitalize()} {lvl['price']:,.1f} ({lvl['src']}, strength {lvl['strength']:.1f})")
    else:
        if d == -1:
            broken = [lv for lv in levels if not lv["dynamic"] and live <= lv["price"] <= live + 1.3 * atr
                      and lv["price"] >= live - 0.05 * atr]
            at_low = f.get("ok") and f.get("dist_sess_low", 9) <= 0.4
        else:
            broken = [lv for lv in levels if not lv["dynamic"] and live - 1.3 * atr <= lv["price"] <= live
                      and lv["price"] <= live + 0.05 * atr]
            at_low = f.get("ok") and f.get("dist_sess_high", 9) <= 0.4
        if broken:
            lvl = max(broken, key=lambda lv: lv["strength"] * 4.0 - abs(live - lv["price"]) / atr * 3.0)
            loc_pts = min(22.0, 9.0 + 4.0 * min(lvl["strength"], 3.0))
            reasons.append(("Support" if d == -1 else "Resistance") + f" {lvl['price']:,.1f} just broken ({lvl['src']})")
        elif at_low:
            loc_pts = 10.0
            reasons.append("Price pressing today's " + ("low" if d == -1 else "high") + " (fresh extreme)")
        else:
            # trend continuation without a fresh level: allowed but needs strong price action
            loc_pts = 6.0
            reasons.append("Trend continuation (no fresh level nearby)")

    # ---------------- PRICE-ACTION TRIGGER ----------------
    pa, pa_notes = 0.0, []
    rsi, rsi_prev = f.get("RSI"), f.get("rsi_prev")
    macd, macd_prev = f.get("MACD_Hist"), f.get("macd_prev")

    if play == "BOUNCE_BUY":
        fell = f.get("ok") and f.get("drop_from_high", 0) >= 0.9
        if not fell and not conf.get("support_rejection_confirmed"):
            return None
        if conf.get("support_rejection_confirmed"):
            pa += 10; pa_notes.append("support rejection candle confirmed")
        if conf.get("bullish_momentum_break_confirmed"):
            pa += 9; pa_notes.append("bullish momentum break")
        if f.get("bull"):
            pa += 3; pa_notes.append("bullish close")
        if max(f.get("lower_wick", 0), f.get("lower_wick_prev", 0)) >= 0.4:
            pa += 5; pa_notes.append("long lower-wick rejection (buyers absorbing)")
        if f.get("stall_after_fall"):
            pa += 5; pa_notes.append(f"selling stalled after {f.get('drop_from_high', 0):.1f} ATR fall")
        if rsi is not None and rsi < 38 and (rsi_prev is None or rsi >= rsi_prev):
            pa += 4 + (2 if rsi < 30 else 0); pa_notes.append(f"RSI {rsi:.0f} oversold and turning")
        if macd is not None and macd_prev is not None and macd < 0 and macd > macd_prev:
            pa += 3; pa_notes.append("MACD histogram improving")
        # falling-knife guard
        if f.get("bear") and f.get("body", 0) >= 0.6 and f.get("close_pos", 1) < 0.25 and pa < 12:
            pa -= 8; misses.append("last candle is still a strong red (falling knife)")
        if f.get("ll6", 0) >= 5 and not conf.get("support_rejection_confirmed"):
            pa -= 4; misses.append("still making lower lows every candle")
    elif play == "REJECTION_SELL":
        rose = f.get("ok") and f.get("rise_from_low", 0) >= 0.9
        if not rose and not conf.get("resistance_rejection_confirmed"):
            return None
        if conf.get("resistance_rejection_confirmed"):
            pa += 10; pa_notes.append("resistance rejection candle confirmed")
        if conf.get("bearish_momentum_break_confirmed"):
            pa += 9; pa_notes.append("bearish momentum break")
        if f.get("bear"):
            pa += 3; pa_notes.append("bearish close")
        if max(f.get("upper_wick", 0), f.get("upper_wick_prev", 0)) >= 0.4:
            pa += 5; pa_notes.append("long upper-wick rejection (sellers absorbing)")
        if f.get("stall_after_rise"):
            pa += 5; pa_notes.append(f"buying stalled after {f.get('rise_from_low', 0):.1f} ATR rise")
        if rsi is not None and rsi > 62 and (rsi_prev is None or rsi <= rsi_prev):
            pa += 4 + (2 if rsi > 70 else 0); pa_notes.append(f"RSI {rsi:.0f} overbought and turning")
        if macd is not None and macd_prev is not None and macd > 0 and macd < macd_prev:
            pa += 3; pa_notes.append("MACD histogram fading")
        if f.get("bull") and f.get("body", 0) >= 0.6 and f.get("close_pos", 0) > 0.75 and pa < 12:
            pa -= 8; misses.append("last candle is still a strong green (rising knife)")
        if f.get("hh6", 0) >= 5 and not conf.get("resistance_rejection_confirmed"):
            pa -= 4; misses.append("still making higher highs every candle")
    elif play == "BREAKDOWN_SELL":
        if not f.get("ok") or not (f.get("ret6", 0) <= -0.6 or f.get("ret12", 0) <= -1.0):
            return None
        if f["ret6"] <= -0.8:
            pa += 5; pa_notes.append(f"strong down-move ({f['ret6']:.1f} ATR in 6 candles)")
        if f.get("ret12", 0) <= -1.5:
            pa += 3
        if f.get("dist_sess_low", 9) <= 0.4:
            pa += 5; pa_notes.append("at/under today's low")
        if conf.get("breakdown_confirmed") or conf.get("breakdown_retest_confirmed"):
            pa += 8; pa_notes.append("breakdown / retest confirmed")
        if f.get("ll6", 0) >= 4:
            pa += 4; pa_notes.append("consistent lower lows")
        if macd is not None and macd < 0 and (macd_prev is None or macd <= macd_prev):
            pa += 3; pa_notes.append("MACD momentum bearish")
        ema20 = f.get("EMA_20")
        if 0.3 <= f.get("rise_from_low", 0) <= 1.1 and ema20 is not None and live < ema20:
            pa += 4; pa_notes.append("weak pullback below EMA20 (sell-the-rally spot)")
        if f.get("bull") and f.get("body", 0) >= 0.6 and f.get("ret3", 0) > 0.5:
            pa -= 8; misses.append("strong green reversal candle against the sell")
        # In a real trend RSI can sit under 30 for a long time, so only punish
        # truly extreme readings (late chase / snap-back risk), and only mildly.
        if rsi is not None and rsi < 18:
            pa -= 6; misses.append(f"RSI {rsi:.0f} extremely oversold - late to chase the fall")
        elif rsi is not None and rsi < 24:
            pa -= 3; misses.append(f"RSI {rsi:.0f} oversold - some bounce risk")
        if lvl is None and pa < 12:
            return None
    elif play == "BREAKOUT_BUY":
        if not f.get("ok") or not (f.get("ret6", 0) >= 0.6 or f.get("ret12", 0) >= 1.0):
            return None
        if f["ret6"] >= 0.8:
            pa += 5; pa_notes.append(f"strong up-move ({f['ret6']:.1f} ATR in 6 candles)")
        if f.get("ret12", 0) >= 1.5:
            pa += 3
        if f.get("dist_sess_high", 9) <= 0.4:
            pa += 5; pa_notes.append("at/over today's high")
        if conf.get("breakout_confirmed") or conf.get("breakout_retest_confirmed"):
            pa += 8; pa_notes.append("breakout / retest confirmed")
        if f.get("hh6", 0) >= 4:
            pa += 4; pa_notes.append("consistent higher highs")
        if macd is not None and macd > 0 and (macd_prev is None or macd >= macd_prev):
            pa += 3; pa_notes.append("MACD momentum bullish")
        ema20 = f.get("EMA_20")
        if 0.3 <= f.get("drop_from_high", 0) <= 1.1 and ema20 is not None and live > ema20:
            pa += 4; pa_notes.append("shallow pullback above EMA20 (buy-the-dip spot)")
        if f.get("bear") and f.get("body", 0) >= 0.6 and f.get("ret3", 0) < -0.5:
            pa -= 8; misses.append("strong red reversal candle against the buy")
        if rsi is not None and rsi > 82:
            pa -= 6; misses.append(f"RSI {rsi:.0f} extremely overbought - late to chase")
        elif rsi is not None and rsi > 76:
            pa -= 3; misses.append(f"RSI {rsi:.0f} overbought - some pullback risk")
        if lvl is None and pa < 12:
            return None

    pa = max(0.0, min(25.0, pa))
    reasons.extend(pa_notes)
    if pa < 8:
        misses.append("price-action trigger too weak (need candle/momentum confirmation)")

    # ---------------- TREND / REGIME / STRUCTURE (0-15) ----------------
    tr = 0.0
    regime, structure, mtf = X["regime"], X["structure"], X["mtf"]
    vwap, ema20, ema50 = f.get("VWAP"), f.get("EMA_20"), f.get("EMA_50")
    if reversal:
        if regime == "RANGE":
            tr += 4
        if (d == 1 and mtf == "BULLISH") or (d == -1 and mtf == "BEARISH"):
            tr += 4; reasons.append("higher-timeframe stack agrees")
        elif mtf == "MIXED":
            tr += 2
        if ema20 is not None:
            stretch = (ema20 - live) / atr if d == 1 else (live - ema20) / atr
            if stretch >= 2.5:
                tr += 8; reasons.append(f"price stretched {stretch:.1f} ATR from EMA20 (big mean-reversion room)")
            elif stretch >= 1.2:
                tr += 6; reasons.append(f"price stretched {stretch:.1f} ATR from EMA20 (mean-reversion room)")
            elif stretch >= 0.6:
                tr += 2
        against = (d == 1 and regime == "BEARISH") or (d == -1 and regime == "BULLISH")
        if against and structure in ("BREAKDOWN", "BREAKOUT") and pa < 14:
            tr -= 5; misses.append("strong trend/structure still against the reversal")
    else:
        if vwap is not None and ((d == -1 and live < vwap) or (d == 1 and live > vwap)):
            tr += 4
        if ema20 is not None and ((d == -1 and live < ema20) or (d == 1 and live > ema20)):
            tr += 3
        if ema20 is not None and ema50 is not None and ((d == -1 and ema20 < ema50) or (d == 1 and ema20 > ema50)):
            tr += 2
        if (d == -1 and regime == "BEARISH") or (d == 1 and regime == "BULLISH"):
            tr += 3; reasons.append("ADX regime is trending your way")
        if (d == -1 and mtf == "BEARISH") or (d == 1 and mtf == "BULLISH"):
            tr += 3; reasons.append("multi-timeframe stack agrees")
        if X["choppy"]:
            tr -= 8; misses.append("market is compressed/choppy - continuation less reliable")
    tr = max(-10.0, min(15.0, tr))

    # ---------------- ORDER FLOW / MODELS (0-10) ----------------
    fl = 0.0
    press = X["pressure"]
    if (d == 1 and press == "BUYING PRESSURE") or (d == -1 and press == "SELLING PRESSURE"):
        fl += 5; reasons.append("order-book pressure agrees")
    elif (d == 1 and press == "SELLING PRESSURE") or (d == -1 and press == "BUYING PRESSURE"):
        # a reversal happens AFTER the opposite pressure; only a light penalty there
        fl -= 1 if reversal else 5
        misses.append("order-book pressure is still against this side")
    if X["ml_agrees"] and X["signal_code"] == d:
        fl += 2
    if X["signal_code"] == d:
        fl += 3; reasons.append("main institutional signal agrees")
    elif X["signal_code"] == -d:
        if not reversal:
            fl -= 4
            misses.append("main signal points the other way")
    fl = max(-8.0, min(10.0, fl))

    # ---------------- BREADTH / GLOBAL / NEWS (0-10) ----------------
    br = 0.0
    if X["breadth_ok"](d):
        br += 3
    if X["global_ok"](d):
        br += 2
    if X["banknifty_ok"]:
        br += 2
    else:
        br -= 3; misses.append("Bank Nifty divergence warning")
    if X["news_ok"](d):
        br += 1.5
    if X["n50news_ok"](d):
        br += 1
    if X["research_ok"](d):
        br += 1.5
    if X["sniper_ok"](d):
        br += 2
    br = max(-5.0, min(10.0, br))

    # ---------------- EXTRAS (0-10) ----------------
    ex = 0.0
    ftxt = X["level_text"](lvl["price"] if lvl else live)
    dw = "bullish" if d == 1 else "bearish"
    if (d == 1 and "heavy put oi" in ftxt) or (d == -1 and "heavy call oi" in ftxt):
        ex += 3; reasons.append("option-chain OI supports the level")
    if "order block" in ftxt and "no order block" not in ftxt:
        ex += 3
    if "liquidity sweep already detected" in ftxt:
        ex += 2
    if X["max_pain_ok"](d):
        ex += 1
    if lvl and any(k in lvl["src"].upper() for k in ("PDH", "PDL", "PDC", "PREV", "OPENING")):
        ex += 2
    ex = min(10.0, ex)

    # ---------------- PENALTIES ----------------
    pen = 0.0
    timing = X["timing"]
    if (d == 1 and timing.get("buy_adverse_move_risk")) or (d == -1 and timing.get("sell_adverse_move_risk")):
        pen += 6; misses.append("entry-timing: momentum already fading at this entry")
    if (d == 1 and conf.get("bearish_trend_transition_confirmed")) or (d == -1 and conf.get("bullish_trend_transition_confirmed")):
        if not (reversal and pa >= 12):
            pen += 7; misses.append("latest candle already turning against this side")
    if X["overlap"] and not reversal:
        pen += 3
    if X["missing_sections"] > 0:
        pen += min(4.0, 0.5 * X["missing_sections"])

    bonus = 0.0
    if reversal and loc_pts >= 14 and pa >= 16:
        bonus = 6.0
        reasons.append("strong level AND strong reaction candle together")
    score = loc_pts + pa + tr + fl + br + ex + bonus - pen
    score = max(0.0, min(100.0, score))
    return {
        "play": play, "d": d, "score": round(score, 1), "loc": round(loc_pts, 1), "pa": round(pa, 1),
        "parts": {"location": round(loc_pts, 1), "price_action": round(pa, 1), "trend": round(tr, 1),
                  "flow": round(fl, 1), "breadth_global": round(br, 1), "extras": round(ex, 1),
                  "bonus": round(bonus, 1), "penalties": round(-pen, 1)},
        "reasons": reasons, "misses": misses, "level": lvl,
    }


# ---------------------------------------------------------------------
# stop-loss / target from structure
# ---------------------------------------------------------------------
def _build_trade_levels(cand, X):
    live, atr, f, levels = X["live"], X["atr"], X["f"], X["levels"]
    d = cand["d"]
    lvl = cand["level"]
    buf = 0.3 * atr

    if d == 1:
        refs = [f.get("low5", live - atr)]
        if lvl:
            refs.append(lvl["price"])
        sl_ref = min(refs) - buf
        sl_dist = live - sl_ref
    else:
        refs = [f.get("high5", live + atr)]
        if lvl:
            refs.append(lvl["price"])
        sl_ref = max(refs) + buf
        sl_dist = sl_ref - live
    sl_dist = max(0.9 * atr, 10.0, min(sl_dist, 2.0 * atr))
    stop = live - sl_dist if d == 1 else live + sl_dist

    # structural targets: real levels in the trade direction, nearest first
    if d == 1:
        tgts = sorted([lv for lv in levels if lv["price"] > live + 0.01], key=lambda x: x["price"])
    else:
        tgts = sorted([lv for lv in levels if lv["price"] < live - 0.01], key=lambda x: -x["price"])
    target, target2, note = None, None, ""
    picked = []
    for lv in tgts:
        dist = abs(lv["price"] - live)
        if dist > 6.0 * atr:
            break
        if dist >= MIN_RR * sl_dist:
            picked.append(lv)
        if len(picked) == 2:
            break
    if picked:
        target = round(picked[0]["price"], 2)
        note = f"target at {picked[0]['src']}"
        if len(picked) > 1:
            target2 = round(picked[1]["price"], 2)
    else:
        target = round(live + FALLBACK_RR * sl_dist if d == 1 else live - FALLBACK_RR * sl_dist, 2)
        note = f"no structure far enough - ATR-based {FALLBACK_RR}R target"
    rr = abs(target - live) / max(sl_dist, 1e-9)
    return round(stop, 2), target, target2, round(rr, 2), note


# ---------------------------------------------------------------------
# market view: where can it fall / bounce / rise
# ---------------------------------------------------------------------
def _build_market_view(X, best_buy, best_sell):
    live, atr, f, levels = X["live"], X["atr"], X["f"], X["levels"]
    sup = sorted([lv for lv in levels if lv["price"] < live - 0.05 * atr and (lv["strength"] >= 1.0 or not lv["dynamic"])],
                 key=lambda x: -x["price"])[:4]
    res = sorted([lv for lv in levels if lv["price"] > live + 0.05 * atr and (lv["strength"] >= 1.0 or not lv["dynamic"])],
                 key=lambda x: x["price"])[:4]
    bs = best_buy["score"] if best_buy else 0.0
    ss = best_sell["score"] if best_sell else 0.0
    if bs >= ss + 8 and bs >= 35:
        bias = "BULLISH BOUNCE/CONTINUATION"
    elif ss >= bs + 8 and ss >= 35:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL / WAIT"

    def _fmt(lv):
        return {"price": lv["price"], "src": lv["src"], "strength": lv["strength"],
                "away_pts": round(abs(lv["price"] - live), 1)}

    lines = []
    if sup:
        chain = " -> ".join(f"{s['price']:,.0f}" for s in sup[:3])
        lines.append(f"Neeche ke supports (girega to yahan ruk sakta hai): {chain}. "
                     f"Pehla support {sup[0]['price']:,.1f} ({sup[0]['src']}), {abs(live - sup[0]['price']):.0f} pts door.")
    else:
        lines.append("Neeche koi validated support nahi mila - price fresh low par hai, bounce sirf candle confirmation par lo.")
    if res:
        chain = " -> ".join(f"{r['price']:,.0f}" for r in res[:3])
        lines.append(f"Upar ke resistances (bounce yahan tak ja sakta hai): {chain}. "
                     f"Pehla resistance {res[0]['price']:,.1f} ({res[0]['src']}), {abs(res[0]['price'] - live):.0f} pts door.")
    if f.get("ok"):
        lines.append(f"Aaj ki range: low {f['sess_low']:,.1f} / high {f['sess_high']:,.1f} "
                     f"(abhi low se {live - f['sess_low']:.0f} pts upar).")
    lines.append(f"Abhi ka best BUY score {bs:.0f}/100, best SELL score {ss:.0f}/100 (trade ke liye {REVERSAL_SCORE_MIN:.0f}-{ENTRY_SCORE_MIN:.0f}+ chahiye).")
    return {"bias": bias, "supports": [_fmt(s) for s in sup], "resistances": [_fmt(r) for r in res],
            "buy_score": round(bs, 1), "sell_score": round(ss, 1), "lines": lines,
            "session_low": f.get("sess_low"), "session_high": f.get("sess_high")}


def _cooldown_active(direction):
    try:
        for s in trade_learning.get_recent_setups(limit=4):
            if s.get("direction") != direction or s.get("status") != "LOSS":
                continue
            ts = s.get("exit_timestamp") or s.get("timestamp")
            t = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - t).total_seconds() < LOSS_COOLDOWN_MIN * 60:
                return True
            return False  # only the most recent same-direction loss matters
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------
# main entry point (signature kept backward compatible; `df` is new/optional)
# ---------------------------------------------------------------------
def generate_trade_decision(live_price, level_prediction, atr, max_pain=None,
                             signal_code=0, ml_agrees=False, banknifty_correlation_note=None,
                             breadth_advances=None, breadth_declines=None,
                             global_avg_change=None, live_vix=None, india_news_sentiment=None,
                             level_ladder=None, sr_context=None, sniper_bias=None, is_choppy=False,
                             dashboard_context=None, global_research=None,
                             nifty50_news_sentiment=None, nifty50_fundamentals_bias=None,
                             df=None):
    """
    Returns a dict. Always contains: has_setup, context_audit, market_view.
      No trade : {"has_setup": False, "reason": str, ...}
      Trade    : {"has_setup": True, "direction", "strike", "option_type", "underlying_entry",
                  "stop_loss", "target", "confidence_pct", "confidence_note", "factor_flags",
                  "factors_true", "factors_total", "level_price", "level_pct", "playbook",
                  "setup_score", "score_parts", "reasons", "risk_reward", ...}
    """
    dashboard_context = dashboard_context or {}
    missing = [k for k in REQUIRED_CONTEXT_SECTIONS if k not in dashboard_context]
    context_audit = {"sections_received": len(dashboard_context), "required_sections": len(REQUIRED_CONTEXT_SECTIONS),
                     "missing_sections": missing, "complete": not missing}

    def _no(reason, **extra):
        out = {"has_setup": False, "reason": reason, "context_audit": context_audit}
        out.update(extra)
        return out

    live = _sf(live_price)
    atr_v = _sf(atr)
    if live is None or live <= 0 or atr_v is None or atr_v <= 0:
        return _no("Live price / ATR unavailable - cannot size a trade.")

    freshness = str(dashboard_context.get("Data Freshness", "")).upper()
    if "STALE" in freshness or "FROZEN" in freshness:
        return _no("Dashboard data is STALE/FROZEN. No fresh setup is opened until live data is fresh.")

    # ---- shared context -------------------------------------------------
    sr = sr_context if isinstance(sr_context, dict) else {}
    conf = sr.get("confirmation") or {}
    timing = sr.get("entry_timing") or {}
    feats = _price_features(df, live, atr_v)
    levels = _collect_levels(live, atr_v, df, feats, level_prediction, level_ladder, sr)
    pressure, _ = _extract_order_flow(dashboard_context)
    regime, structure = _extract_regime(dashboard_context)
    mtf = _extract_mtf(dashboard_context)

    adv_dec = None
    if breadth_advances is not None and breadth_declines is not None:
        adv_dec = breadth_advances - breadth_declines
    news_up = (india_news_sentiment or "").upper()
    n50 = (nifty50_news_sentiment or "").upper()
    sniper_up = (sniper_bias or "").upper()
    gr_bias = str((global_research or {}).get("directional_bias", "") if isinstance(global_research, dict) else "").upper()
    fund = (nifty50_fundamentals_bias or "").upper()

    def _dir_word_ok(txt, d):
        return ("BULL" in txt and d == 1) or ("BEAR" in txt and d == -1)

    ladder_levels = []
    if isinstance(level_ladder, dict):
        ladder_levels = [x for x in (level_ladder.get("supports") or []) + (level_ladder.get("resistances") or [])
                         if isinstance(x, dict)]
    if isinstance(level_prediction, dict) and level_prediction.get("level_price"):
        ladder_levels = ladder_levels + [level_prediction]

    def level_text(price):
        best, bd = None, 1e9
        for x in ladder_levels:
            p = _sf(x.get("level_price"))
            if p is not None and abs(p - price) < bd:
                best, bd = x, abs(p - price)
        if best is None or bd > 0.6 * atr_v:
            return ""
        return " ".join(str(t) for t in (best.get("factors") or [])).lower()

    X = {
        "live": live, "atr": atr_v, "f": feats, "conf": conf, "levels": levels, "timing": timing,
        "overlap": bool(sr.get("overlapping_zone_warning")), "choppy": bool(is_choppy),
        "pressure": pressure, "regime": regime, "structure": structure, "mtf": mtf,
        "signal_code": signal_code if signal_code in (1, -1) else 0, "ml_agrees": bool(ml_agrees),
        "missing_sections": len(missing),
        "breadth_ok": lambda d: adv_dec is not None and adv_dec * d > 0,
        "global_ok": lambda d: global_avg_change is not None and global_avg_change * d > 0.1,
        "banknifty_ok": not (banknifty_correlation_note and "DIVERGENCE WARNING" in str(banknifty_correlation_note).upper()),
        "news_ok": lambda d: _dir_word_ok(news_up, d),
        "n50news_ok": lambda d: _dir_word_ok(n50, d),
        "research_ok": lambda d: _dir_word_ok(gr_bias, d),
        "sniper_ok": lambda d: _dir_word_ok(sniper_up, d),
        "max_pain_ok": lambda d: max_pain is not None and ((d == 1 and live > max_pain) or (d == -1 and live < max_pain)),
        "level_text": level_text,
    }

    # ---- evaluate all four playbooks -----------------------------------
    cands = []
    for play, d in (("BOUNCE_BUY", 1), ("REJECTION_SELL", -1), ("BREAKDOWN_SELL", -1), ("BREAKOUT_BUY", 1)):
        try:
            c = _score_candidate(play, d, X)
        except Exception:
            logger.exception("playbook %s failed", play)
            c = None
        if c:
            cands.append(c)

    best_buy = max((c for c in cands if c["d"] == 1), key=lambda c: c["score"], default=None)
    best_sell = max((c for c in cands if c["d"] == -1), key=lambda c: c["score"], default=None)
    market_view = _build_market_view(X, best_buy, best_sell)

    if not cands:
        return _no("Abhi koi playbook lagu nahi - price na kisi level ke paas hai, na clear momentum hai. "
                   "Engine level ya momentum banne ka wait kar raha hai.", market_view=market_view)

    ranked = sorted(cands, key=lambda c: c["score"], reverse=True)

    def _summary(c):
        return f"{PLAYBOOK_LABELS[c['play']]} {c['score']:.0f}/100"

    chosen = None
    blocked_reason = None
    for c in ranked:
        is_rev = c["play"] in ("BOUNCE_BUY", "REJECTION_SELL")
        if c["score"] < (REVERSAL_SCORE_MIN if is_rev else ENTRY_SCORE_MIN):
            continue
        if c["pa"] < 12 or c["loc"] < (12 if is_rev else 6):
            blocked_reason = blocked_reason or f"{_summary(c)}: location/price-action trigger not strong enough yet"
            continue
        direction_name = "BUY" if c["d"] == 1 else "SELL"
        if _cooldown_active(direction_name):
            blocked_reason = blocked_reason or f"{direction_name} cooldown: same-side trade just hit stop-loss (<{LOSS_COOLDOWN_MIN} min)"
            continue
        chosen = c
        break

    if chosen is None:
        top = ranked[0]
        why = "; ".join(top["misses"][:3]) if top["misses"] else "evidence not enough yet"
        need = REVERSAL_SCORE_MIN if top["play"] in ("BOUNCE_BUY", "REJECTION_SELL") else ENTRY_SCORE_MIN
        reason = (blocked_reason or
                  f"Best candidate: {_summary(top)} - trade ke liye {need:.0f}+ chahiye. Kami: {why}.")
        return _no(reason, market_view=market_view,
                   candidates=[{"playbook": c["play"], "score": c["score"], "parts": c["parts"], "misses": c["misses"]}
                               for c in ranked])

    # ---- build the trade ------------------------------------------------
    direction = "BUY" if chosen["d"] == 1 else "SELL"
    d = chosen["d"]
    stop, target, target2, rr, target_note = _build_trade_levels(chosen, X)
    lvl = chosen["level"]
    level_price = lvl["price"] if lvl else live

    # factor flags (feed the self-learning confidence; same keys as before)
    ftxt = level_text(level_price)
    lp_pct = None
    if isinstance(level_prediction, dict) and level_prediction.get("status") == "APPROACHING_LEVEL":
        lp_pct = _sf(level_prediction.get("bounce_pct") if str(level_prediction.get("approaching")) == ("support" if d == 1 else "resistance")
                     else level_prediction.get("break_pct"))
        bias_lower = str(level_prediction.get("directional_bias", "")).lower()
        if not (("bullish" in bias_lower and d == 1) or ("bearish" in bias_lower and d == -1)):
            lp_pct = None
    same_side = (level_ladder or {}).get("resistances" if d == 1 else "supports") if isinstance(level_ladder, dict) else []
    agreeing = 0
    for x in (same_side or [])[:3]:
        b = str(x.get("directional_bias", "")).lower()
        if ("bullish" in b and d == 1) or ("bearish" in b and d == -1):
            agreeing += 1
    ema20, vwap = feats.get("EMA_20"), feats.get("VWAP")
    src_up = (lvl["src"].upper() if lvl else "")
    trans_against = bool(conf.get("bearish_trend_transition_confirmed" if d == 1 else "bullish_trend_transition_confirmed"))
    timing_bad = bool(timing.get("buy_adverse_move_risk" if d == 1 else "sell_adverse_move_risk"))
    factor_flags = {
        "main_signal_aligned": X["signal_code"] == d,
        "ml_agrees": bool(ml_agrees),
        "level_pct_ge_65": bool(lp_pct is not None and lp_pct >= 60.0),
        "banknifty_no_divergence": X["banknifty_ok"],
        "breadth_aligned": bool(X["breadth_ok"](d)),
        "global_aligned": bool(X["global_ok"](d)),
        "oi_aligned": (d == 1 and "heavy put oi" in ftxt) or (d == -1 and "heavy call oi" in ftxt),
        "fvg_ob_confluence": "order block" in ftxt and "no order block" not in ftxt,
        "vwap_aligned": bool(vwap is not None and ((d == 1 and live > vwap) or (d == -1 and live < vwap))),
        "htf_1h_aligned": ("1-hour" in ftxt and ("bullish" if d == 1 else "bearish") in ftxt),
        "htf_15min_aligned": ("15-minute" in ftxt and ("bullish" if d == 1 else "bearish") in ftxt),
        "round_number_level": bool(level_price % 50 < 3 or level_price % 50 > 47),
        "liquidity_sweep": "liquidity sweep already detected" in ftxt,
        "low_vix": bool(live_vix is not None and live_vix < 15.0),
        "news_sentiment_aligned": bool(X["news_ok"](d)),
        "ladder_confluence_aligned": agreeing >= 2,
        "sniper_setup_aligned": bool(X["sniper_ok"](d)),
        "away_from_max_pain": bool(X["max_pain_ok"](d)),
        "global_research_aligned": bool(X["research_ok"](d) or "NEUTRAL" in gr_bias),
        "order_flow_aligned": (d == 1 and pressure == "BUYING PRESSURE") or (d == -1 and pressure == "SELLING PRESSURE"),
        "regime_aligned": (d == 1 and regime == "BULLISH") or (d == -1 and regime == "BEARISH"),
        "mtf_stack_aligned": (d == 1 and mtf == "BULLISH") or (d == -1 and mtf == "BEARISH"),
        "data_quality_pass": bool(feats.get("ok") and "STALE" not in freshness),
        "strong_daily_level": any(k in src_up for k in ("PDH", "PDL", "PDC", "PREV DAY", "PREVIOUS DAY")),
        "strong_opening_range_level": "OPENING RANGE" in src_up,
        "entry_timing_safe": not timing_bad,
        "no_opposite_transition": not trans_against,
        "immediate_reversal_risk_low": bool(not timing_bad and not trans_against),
        "nifty50_news_aligned": bool(X["n50news_ok"](d)),
        "nifty50_fundamentals_aligned": bool(_dir_word_ok(fund, d) or "NEUTRAL" in fund),
    }
    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    try:
        confidence_pct, used_learning, learned_count = trade_learning.compute_confidence(factor_flags)
    except Exception:
        # a storage hiccup must never freeze the engine again -- fall back to the rule-based estimate
        logger.exception("compute_confidence failed; using rule-based confidence")
        confidence_pct, used_learning, learned_count = 50.0 + min(18.0, factors_true * 1.8), False, 0
    # nudge by how strong THIS setup's evidence is, then respect the honest ceiling
    confidence_pct += max(-6.0, min(6.0, (chosen["score"] - 60.0) * 0.2))
    confidence_pct = round(max(trade_learning.CONFIDENCE_FLOOR, min(trade_learning.CONFIDENCE_CEILING, confidence_pct)), 1)

    if confidence_pct < MIN_CONFIDENCE:
        return _no(f"{_summary(chosen)} mila, lekin apne resolved trades se seekha hua confidence {confidence_pct:.1f}% "
                   f"(< {MIN_CONFIDENCE:.0f}%) hai - engine is pattern ko abhi skip kar raha hai.",
                   market_view=market_view, factors_true=factors_true, factors_total=factors_total,
                   factor_flags=factor_flags)

    try:
        track = trade_learning.get_overall_track_record()
    except Exception:
        logger.exception("track record unavailable")
        track = {"sample_size": 0, "win_rate": None}
    if used_learning:
        confidence_note = (f"Blended from {learned_count} learned factor(s) with enough history, rest rule-based. "
                           f"Overall track record so far: {track['sample_size']} resolved ({track['win_rate']}% win rate)."
                           if track['sample_size'] > 0 else
                           f"Blended from {learned_count} learned factor(s); no resolved trades yet to show an overall win rate.")
    else:
        confidence_note = ("Pure rule-based estimate -- not enough resolved setups yet for the engine to learn which factors "
                           "actually win in your data. It gets more honest as more trades resolve.")

    return {
        "has_setup": True, "direction": direction, "strike": _round_to_strike(live),
        "option_type": "CE" if d == 1 else "PE",
        "underlying_entry": round(live, 2), "stop_loss": stop, "target": target, "target2": target2,
        "confidence_pct": confidence_pct, "confidence_note": confidence_note,
        "factor_flags": factor_flags, "factors_true": factors_true, "factors_total": factors_total,
        "level_price": level_price, "level_pct": lp_pct if lp_pct is not None else chosen["score"],
        "context_audit": context_audit, "market_view": market_view,
        "score_model": "playbook_engine_v13",
        "playbook": chosen["play"], "playbook_label": PLAYBOOK_LABELS[chosen["play"]],
        "setup_score": chosen["score"], "score_parts": chosen["parts"],
        "reasons": chosen["reasons"], "cautions": chosen["misses"],
        "risk_reward": rr, "target_note": target_note,
        "critical_confirmations": factors_true, "major_conflicts": len(chosen["misses"]),
        "order_flow_pressure": pressure, "regime_bias": regime, "mtf_bias": mtf,
        "analysis_mode": "PLAYBOOK_ALL_FACTORS",
        "level_distance_pts": round(abs(live - level_price), 2),
        "entry_location_quality": "AT_LEVEL" if lvl else "MOMENTUM",
        "raw_direction": "BUY" if X["signal_code"] == 1 else ("SELL" if X["signal_code"] == -1 else "NEUTRAL"),
        "reversal_used": chosen["play"] in ("BOUNCE_BUY", "REJECTION_SELL"),
        "reversal_reason": "", "reversal_confirmation_count": 0,
        "early_entry_validated": True, "early_entry_checks": 0,
    }
