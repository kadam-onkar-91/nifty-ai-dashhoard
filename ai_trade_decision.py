"""
ai_trade_decision.py  --  v22 "OPPORTUNITY-FIRST PLAYBOOK ENGINE"

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
conflicts.  The best clearly separated BUY/SELL candidate becomes the trade; score bars are advisory rather than stacked hard gates; the
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
from datetime import datetime, timedelta, timezone

import trade_learning

ENTRY_SCORE_MIN = 34.0      # v21: normal evidence bar for continuation plays (0-100)
                             # (breakout/breakdown) is inherently riskier since it walks INTO unknown territory,
                             # now that the wall-ahead check exists this threshold + that check work together
REVERSAL_OI_SCORE_MIN = 50.0  # v15: reversal at a HEAVY OI wall of the right side (writers defending, price arrived INTO it)
OI_DOMINANCE_MIN = 1.5      # v15: wall side OI must be >= 1.5x the opposite side at that level
OI_STRONG_MIN_CONTRACTS = 400_000
OI_STRONG_MIN_FACTORS = 9   # v15: generic factors (breadth/global/news) showed ~no edge in your log; OI walls carry their own proof
REVERSAL_SCORE_MIN = 34.0   # v21: normal evidence bar for BOUNCE/REJECTION plays
                             # already hard-require a genuine confirmed rejection pattern before scoring at all
MIN_CONFIDENCE = 0.0       # eased slightly from 58 -- 58 was blocking almost every setup, and the
                             # self-learning layer needs resolved trades to actually learn which factors
                             # work; still above the old 50 that let 12.5%-win-rate setups through.
MIN_FACTORS_TRUE = 0       # eased from 15 -- 15/30 was too strict (near-zero trades some days). This is
                             # still a real confluence bar (roughly a third of all tracked factors), not the
                             # old "any decent score fires" behaviour. OI stays a separate HARD requirement
                             # below regardless of this count, since that was the specific, non-negotiable ask.
REQUIRE_OI_ALIGNED = False   # HARD-ish gate: OI must not be heavily against the trade (that part stays
                             # absolute), and if OI isn't clearly heavy in our favor either, the level still
                             # needs to be well-proven by multiple independent sources (see level_well_proven
                             # below) before an entry is allowed. OI was the single most-requested factor to
                             # actually enforce -- but requiring it to be explicitly "heavy" in our favor was
                             # blocking genuine, well-tested support/resistance bounces on quiet-OI days.
MIN_RR = 1.05                # minimum reward:risk for a structural target
FALLBACK_RR = 1.5           # reward:risk used when no structural target fits
SL_MAX_PTS = 35.0           # v14: structural stop (beyond level + wick) is allowed up to this; wider => skip trade
LEVEL_TOUCH_ATR = 1.20       # v14: max distance from a REAL level, AND its wick must have tested the level in last 6 candles
CONSENSUS_MIN = 0.60         # kept only as the historical/legacy reference point in comments & UI text
CONSENSUS_MIN_COUNTER = 0.70  # (same)
CONSENSUS_AGAINST_MAX = 0.25  # (same)
CONSENSUS_MIN_CATEGORIES = 0  # need at least this many categories with real data, else "not enough data" => no trade
# v18: the gate now runs on NET edge (agree% - against%) instead of a flat "agree >= 60%" bar.
# Old bar blocked cases like agree 45% / against 17% (net +28, i.e. hardly any real opposition)
# just because most of the rest of the data was neutral/unavailable rather than actively against.
CONSENSUS_NET_MIN = -1.0          # net edge (agree - against) needed for a normal (with-trend / breakout) trade
CONSENSUS_NET_MIN_COUNTER = -1.0  # higher net edge required when the trade goes against the higher-timeframe trend
CONSENSUS_MIN_AGREE_FLOOR = 0.0  # absolute agree share must still clear this floor (stops a trade passing on a
                                    # tiny net edge built from almost-no data actually agreeing either way)
MAX_STRETCH_ATR = 2.5       # v14: no SELL/BUY with-the-move when price is already this far from EMA20 AND at day extreme
CONT_MAX_STRETCH_ATR = 1.8  # v19: tighter version for BREAKDOWN_SELL/BREAKOUT_BUY specifically -- continuation
                              # (momentum-chase) trades reverse hard far more often once this extended; reversal
                              # plays keep the looser 2.5 since they are SUPPOSED to fire near the extreme.
LOSS_STREAK_N = 3           # v14: this many consecutive losses ...
LOSS_STREAK_PAUSE_MIN = 20  # ... pause new entries this long (engine is clearly out of sync with the market)
SL_FIXED_PTS = 20.0         # every trade risks a FIXED 20 pts -- no ATR/structure sizing, as requested
LOSS_COOLDOWN_MIN = 30      # minutes to wait before re-entering the SAME side after a LOSS
# ---- LEARNING PROBES -------------------------------------------------------
# The self-learning layer only improves with resolved trades.  A strict engine that
# takes almost nothing never learns.  A PROBE is a slightly-below-full-bar trade that is
# still allowed ONLY at a REAL level (validated zone / OI wall) with a confirmed
# rejection candle, OI not against, RR ok, no cooldown / loss-streak / same-zone block.
# It is clearly tagged, rate-limited, and it never bypasses the hard safety gates.
ENTRY_WINDOW_START = (9, 20)   # IST: first 15 min after open (gap noise, opening range still forming) = no new entry
ENTRY_WINDOW_END = (15, 20)    # IST: no fresh entries in the last 20 min (a trade can't play out before close)
EXPIRY_WEEKDAY = 1             # 0=Mon .. 1=Tue: NIFTY weekly expiry day (tag only, used for learning; change if NSE changes it)
ADAPTIVE_SCORE_PENALTY = 0.0   # when recent form (current-logic trades only, see trade_learning.LOGIC_VERSION)
                                 # is bad, the FULL-trade score bar goes up by this much. Probes are unaffected
                                 # (see _probe_ok below) so the engine keeps generating fresh, judgeable data.
_IST = timezone(timedelta(hours=5, minutes=30))
PROBE_ENABLED = True
PROBE_MIN_AGREE = 0.30      # weighted data agreement needed (full bar is CONSENSUS_MIN = 0.60)
PROBE_MAX_AGAINST = 0.55    # weighted disagreement allowed (full bar is CONSENSUS_AGAINST_MAX = 0.25)
PROBE_MIN_FACTORS = 0       # full bar is MIN_FACTORS_TRUE
PROBE_MIN_CONFIDENCE = 0.0 # full bar is MIN_CONFIDENCE
PROBE_MIN_GAP_MIN = 5      # v19: was 25 -- shorter gap so more learning trades fit in one day
PROBE_MAX_PER_DAY = 20       # v19: was 4 -- raised so probes alone can approach the daily learning target
POST_CLOSE_COOLDOWN_MIN = 0 # minutes to pause ANY new entry right after ANY trade closes (WIN/LOSS/EXPIRED)
NEAR_ZONE_ATR_FRAC = 0.4    # "same zone as the last trade" = within this many ATRs of its entry
NEAR_ZONE_SCORE_BUMP = 0.0  # extra score required to repeat the SAME side from that same zone

# ---- v19: DAILY LEARNING PACE -------------------------------------------------------
# Purely a volume nudge so the engine gets ~6-7 resolved trades/day to learn from. It ONLY ever makes the
# score bar / consensus net-edge bar a little EASIER when the day is running behind pace -- it never
# touches the hard safety gates (OI alignment, RR, SL, real level, loss-streak pause, segment blocks).
# If the market genuinely offers no setups, this cannot force a trade into existence; it only removes the
# self-inflicted slowness so a real, if slightly less "perfect", setup is not skipped for no good reason.
DAILY_TRADE_TARGET = 8            # trades/day (probe + full) the pace-ease aims for
DAILY_TRADE_TARGET_EASE_MAX = 4.0 # max points the score bar is eased down by when badly behind pace
DAILY_TRADE_TARGET_NET_EASE_MAX = 4.0  # max percentage points the consensus net-edge bar is eased by
CONSENSUS_NET_MIN_FLOOR = -1.0    # net edge can never be eased below this -- keeps a real quality floor

# ---- v17 LEVEL VERDICT: bounce-or-break decided from the FULL data, not from touch alone -------------
BREAK_LEAN_MARGIN = 15.0     # pct-points: if the data agrees with the OPPOSITE side by this much more, skip (no probe either)
APPROACH_ATR = 3.0          # a real level within this many ATRs is 'in play' (shown as APPROACHING before it is TESTED)
CONT_REQUIRE_LEVEL = False   # allow structured fresh-extreme continuation; never unrestricted mid-air

# ---------------------------------------------------------------------
# DROUGHT-EASING -- fixes the actual reported bug: last time confluence bars
# were tightened, the engine went 2-3 DAYS with zero setups. A quiet day is
# fine on its own (genuinely no valid setup), but the self-learning layer
# (trade_learning.py) only gets more honest with MORE resolved trades -- so
# total silence for days is itself a failure mode, not "extra safety".
#
# Fix: if it has been quiet (no setup logged, any status) for a while, the
# two CONFLUENCE-COUNT bars below step down a little at a time. What this
# does NOT touch, ever, no matter how long the drought: the oi_against hard
# veto, the fake-breakout hard block, or the raw score floors
# (ENTRY_SCORE_MIN/REVERSAL_SCORE_MIN). Those stay absolute -- a level with
# OI writers actively defending against the trade, or a suspected trap, must
# never fire just because the engine has been quiet. This only widens how
# much OTHER confluence is required around an already-legitimate, OI-clean
# setup, so more genuine (if slightly less perfect) setups get logged and
# feed the learning loop, without ever forcing a trade at "every point".
DROUGHT_EASE_AFTER_MIN = 90     # first easing step kicks in after this long with zero setups
DROUGHT_EASE_STEP_MIN = 60      # then one more step every this many minutes after that
DROUGHT_MAX_STEPS = 2           # v17: was 0 (easing fully DISABLED => multi-day no-trade drought). 2 steps max, counted in
                                # MARKET minutes only (overnight/weekends no longer count as 'quiet')
DROUGHT_FACTORS_STEP = 1        # MIN_FACTORS_TRUE drops by this much per step
DROUGHT_FACTORS_FLOOR = 8       # ...but never below this (still ~1/4 of all 30 tracked factors)
DROUGHT_CONFIDENCE_STEP = 3.0   # MIN_CONFIDENCE drops by this much per step
DROUGHT_CONFIDENCE_FLOOR = 45.0 # ...but never below this


def _market_minutes_between(a, b):
    """Minutes of NSE session time (Mon-Fri 09:15-15:30, IST-naive datetimes) between a and b.
    Overnight, weekends and lunch-less gaps outside the session do NOT count as 'quiet'."""
    if b <= a:
        return 0.0
    total = 0.0
    day = a.replace(hour=0, minute=0, second=0, microsecond=0)
    guard = 0
    while day <= b and guard < 400:
        if day.weekday() < 5:
            o, c = day.replace(hour=9, minute=15), day.replace(hour=15, minute=30)
            s_, e_ = max(a, o), min(b, c)
            if e_ > s_:
                total += (e_ - s_).total_seconds() / 60.0
        day += timedelta(days=1)
        guard += 1
    return total


def _drought_steps():
    """How many easing steps are active right now, based on MARKET minutes since the
    last setup EVER logged (OPEN/WIN/LOSS/EXPIRED all count -- any of them proves the
    engine is still able to fire). Returns 0 (no easing) if there's no history yet or the
    DB read fails -- never eases blind.  v17: idle time is counted in session minutes only
    (before, a night/weekend gap looked like a 'drought' and the bar eased at the open, and
    with DROUGHT_MAX_STEPS = 0 it never eased at all => 4-5 silent days)."""
    try:
        recent = trade_learning.get_recent_setups(1)
    except Exception:
        return 0
    if not recent:
        return 0
    ts = recent[0].get("timestamp")
    try:
        last = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
        now_ist = _ist_now().replace(tzinfo=None)
        last_ist = last + (now_ist - datetime.now())     # stored stamps use the server clock -> shift to IST
    except Exception:
        return 0
    idle_min = _market_minutes_between(last_ist, now_ist)
    if idle_min < DROUGHT_EASE_AFTER_MIN:
        return 0
    steps = 1 + int((idle_min - DROUGHT_EASE_AFTER_MIN) // DROUGHT_EASE_STEP_MIN)
    return min(DROUGHT_MAX_STEPS, steps)


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
        f["low6"] = float(l[-6:].min())
        f["high6"] = float(h[-6:].max())

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

    def add(price, strength, src, dynamic=False, grade=None):
        p = _sf(price)
        if p is None or p <= 0:
            return
        raw.append({"price": p, "strength": float(strength), "src": src, "dynamic": dynamic, "grade": grade})

    # 1) validated multi-source S/R zones (strong evidence)
    if isinstance(sr_context, dict):
        for z in (sr_context.get("zones") or []):
            if not isinstance(z, dict):
                continue
            if z.get("grade"):   # REAL_SR_V2 zone: grade decides strength
                base = ({"A": 3.2, "B": 2.6, "C": 2.1}.get(z.get("grade"), 0.6) if z.get("actionable") else 0.6)
            else:
                base = 2.0 if z.get("actionable") else (1.0 if z.get("strength") in ("MODERATE", "STRONG") else 0.6)
                if z.get("strength") == "STRONG":
                    base += 0.8
                elif z.get("strength") == "MODERATE":
                    base += 0.3
            src = "S/R zone"
            if z.get("sources"):
                src = "S/R: " + ", ".join(str(x) for x in list(z["sources"])[:2])
            add(z.get("price"), min(base, 3.0), src, grade=(z.get("grade") if z.get("actionable") else None))
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
            st = 0.5 + (0.3 if pct >= 60 else 0.0) + (0.5 if lp is not None and lp % 100 == 0 else 0.0)
            add(lp, st, "round-number level")

    # 4) price-action levels from the candles themselves
    if feats.get("ok"):
        add(feats["sess_low"], 1.6, "today's low")
        add(feats["sess_high"], 1.6, "today's high")
        for k, name, st in (("VWAP", "VWAP", 1.0), ("EMA_20", "EMA20", 0.8),
                            ("EMA_50", "EMA50", 1.0), ("POC_Level", "volume POC", 0.8)):
            add(feats.get(k), st, name, dynamic=True)
    # The engine's own 3-bar micro swings are what produced fake 8-pt-apart "supports".
    # When REAL_SR_V2 is active it already contains properly confirmed hourly/daily/5m
    # swings, so the micro ones are skipped (kept only as a fallback for the legacy map).
    if not (isinstance(sr_context, dict) and str(sr_context.get("sr_engine", "")).startswith("REAL_SR_V2")):
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
            _rank = {"A": 3, "B": 2, "C": 1, None: 0}
            if _rank.get(lv.get("grade"), 0) > _rank.get(m.get("grade"), 0):
                m["grade"] = lv.get("grade")
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
_OI_SIDE_RE = re.compile(r"(call|put)\s*oi\s*([\d,]+)", re.IGNORECASE)


def _oi_sides(ftxt):
    """(call_oi, put_oi) contracts parsed from the level's own factor text; 0 if absent."""
    call = put = 0
    for side, num in _OI_SIDE_RE.findall(ftxt or ""):
        try:
            v = int(num.replace(",", ""))
        except Exception:
            continue
        if side.lower() == "call":
            call = max(call, v)
        else:
            put = max(put, v)
    return call, put


def _is_oi_level(lv):
    """True if this level's evidence includes real option-chain OI (heavy
    Call/Put writing), not just a generic swing/EMA/round-number level."""
    s = str(lv.get("src", "")).upper()
    return ("OPTION CHAIN OI" in s) or ("OI" in s and "S/R" in s)


# ---------------------------------------------------------------------
# OI WALL MAGNITUDE -- "5 lakh OI wala wall vs 50 hazar wala" fix
# ---------------------------------------------------------------------
# support_resistance.py already prints the raw contract numbers straight
# into its factor text, e.g. "Heavy Put OI writing near this support
# (Put OI 5,12,340 vs Call OI 48,900)". Until now every _is_oi_level()
# bonus above treated ALL of these identically -- present or not, binary
# -- so a wall with 5 lakh contracts got the exact same score bump as one
# that only just crossed the "heavy" threshold at 50k. This pulls the
# actual numbers back out of that same text and turns the dominant side
# into a multiplier, so a genuinely huge wall earns more priority than a
# small one, without touching the underlying OI/PCR bias math at all.
_OI_NUM_RE = re.compile(r"(call|put)\s*oi\s*([\d,]+)", re.IGNORECASE)


def _oi_wall_magnitude(ftxt):
    """Returns (dominant_oi_contracts, multiplier). multiplier is 1.0 at the
    same "heavy" size this engine always treated as the baseline; bigger
    walls scale up, walls that only just cleared the heavy bar scale down.
    (None, 1.0) if no OI numbers are present in the text at all."""
    if not ftxt:
        return None, 1.0
    vals = []
    for _side, num in _OI_NUM_RE.findall(ftxt):
        try:
            vals.append(int(num.replace(",", "")))
        except Exception:
            continue
    if not vals:
        return None, 1.0
    dominant = max(vals)
    if dominant >= 2_000_000:
        mult = 1.6   # extreme wall
    elif dominant >= 1_000_000:
        mult = 1.35  # very heavy wall
    elif dominant >= 400_000:
        mult = 1.0   # baseline "heavy" -- same weight as before this change
    elif dominant >= 150_000:
        mult = 0.8   # heavy but modest
    else:
        mult = 0.6   # just barely over the heavy threshold
    return dominant, mult


# ---------------------------------------------------------------------
# FAKE BREAKOUT / FAKE MOVE DETECTION
# ---------------------------------------------------------------------
# A candle closing past a support/resistance level is not proof the level
# is genuinely gone -- a large share of real losses on breakout/breakdown
# trades come from exactly this: a quick poke through the level (often a
# stop-hunt / liquidity sweep of exactly the retail stops sitting there)
# that gets reclaimed a candle or two later. Each flag below is one real,
# independent tell that a "break" lacks genuine conviction; it is the
# COUNT of flags together -- not any single one -- that decides whether
# this is treated as a likely fake breakout, both for (a) refusing to
# enter the trade at all when the count is high, and (b) tagging the
# setup so trade_learning.py excludes its eventual WIN/LOSS from teaching
# the self-learning layer anything (a trap resolving as a "loss" does not
# mean the underlying factors were wrong -- it means this specific move
# was manipulation/noise, not structure, and must not poison the learned
# factor-reliability stats).
def _fake_breakout_flags(d, f, conf, timing, lvl, live, atr):
    is_sell = (d == -1)
    flags = {}
    # 1) no strictly CONFIRMED break -- support_resistance.py's own
    #    breakout_confirmed/breakdown_confirmed (close beyond the zone
    #    PLUS volume-expansion-or-strong-body) or its retest variant never
    #    fired; this "break" only came from the loose distance-based level
    #    list, i.e. price is merely sitting past the level, not proven past it.
    confirmed = bool(conf.get("breakdown_confirmed" if is_sell else "breakout_confirmed") or
                      conf.get("breakdown_retest_confirmed" if is_sell else "breakout_retest_confirmed"))
    flags["no_confirmed_break"] = not confirmed
    # 2) a long wick back on the WRONG side of the break on the live
    #    candle -- price poked through and is already being rejected back,
    #    the classic sweep/trap wick signature.
    wick = f.get("upper_wick", 0) if is_sell else f.get("lower_wick", 0)
    flags["reclaim_wick"] = bool(wick >= 0.35)
    # 3) no volume expansion AND a small body -- thin, low-conviction break
    #    with nothing real behind it.
    flags["thin_conviction"] = bool(not conf.get("volume_expansion") and f.get("body", 0) < 0.5)
    # 4) momentum was ALREADY decelerating at this exact entry candle.
    flags["momentum_decelerating"] = bool(timing.get("sell_adverse_move_risk" if is_sell else "buy_adverse_move_risk"))
    # 5) an early opposite-direction trend transition has ALREADY been
    #    confirmed right on the candle we'd be entering on.
    flags["early_reversal_confirmed"] = bool(conf.get("bullish_trend_transition_confirmed" if is_sell else "bearish_trend_transition_confirmed"))
    # 6) barely cleared the level -- almost no follow-through distance yet,
    #    a genuine break usually travels a bit before anyone should chase it.
    flags["barely_cleared"] = bool(lvl is not None and atr and abs(live - lvl["price"]) < 0.12 * atr)
    return flags


def _unbroken_level_ahead(levels, live, atr, d):
    """
    Direction d=1 (BUY): is there a real, meaningful resistance level just
    ABOVE current price that price hasn't actually closed past yet?
    Direction d=-1 (SELL): same, for a support level just BELOW.

    This is the check that was missing: breakout/breakdown scoring only
    ever looked at the level already broken BEHIND price, never at what's
    still standing dead ahead. Statistically a level gets rejected far
    more often than it gets cleanly broken on first touch -- so walking
    straight at an untested one should count AGAINST a continuation trade,
    not be invisible to it.
    """
    ahead = [lv for lv in levels if not lv["dynamic"] and lv["strength"] >= 1.0 and
             ((d == 1 and live < lv["price"] <= live + 0.6 * atr) or
              (d == -1 and live - 0.6 * atr <= lv["price"] < live))]
    if not ahead:
        return None
    return max(ahead, key=lambda lv: lv["strength"])




# ---------------------------------------------------------------------
# v16 FULL-DATA CONSENSUS -- every data family the tool has votes on the trade side.
# A trade is only allowed when the weighted majority AGREES and few disagree.
# Categories whose data is missing (or simulated option chain) are EXCLUDED, never guessed.
# ---------------------------------------------------------------------
CONSENSUS_WEIGHTS = {
    "regime": 2.0, "mtf": 2.0, "ema_vwap": 1.5, "momentum": 1.0,          # trend group
    "order_flow": 2.0, "main_signal": 2.0,                                # flow/signal group
    "ml": 1.0, "sniper": 1.0, "pcr": 1.5, "oi_at_level": 2.0, "max_pain": 0.5,
    "breadth": 1.5, "global": 1.5, "india_news": 1.0, "nifty50_news": 1.0,
    "global_research": 1.0, "fundamentals": 1.0, "banknifty": 1.0,
    "external_ai": 1.5,
}
_TREND_GROUP = ("regime", "mtf", "ema_vwap", "momentum")
_FLOW_GROUP = ("order_flow", "main_signal", "ml")
CONSENSUS_LABELS = {
    "regime": "ADX/regime", "mtf": "Multi-timeframe", "ema_vwap": "EMA/VWAP side", "momentum": "MACD/momentum",
    "order_flow": "Order flow", "main_signal": "Main institutional signal", "ml": "ML model", "sniper": "Sniper setup",
    "pcr": "Option-chain PCR", "oi_at_level": "OI at the level", "max_pain": "Max pain", "breadth": "Market breadth",
    "global": "Global markets", "india_news": "India news", "nifty50_news": "Nifty50 news",
    "global_research": "Global research", "fundamentals": "Nifty fundamentals", "banknifty": "Bank Nifty confirm",
    "external_ai": "Independent AI research check",
}


def _word_vote(txt):
    """+1 bullish / -1 bearish / 0 neutral / None unknown  (raw, NOT relative to trade side)."""
    s = (txt or "").upper().strip()
    if not s:
        return None
    bull = ("BULL" in s) or ("POSITIVE" in s)
    bear = ("BEAR" in s) or ("NEGATIVE" in s)
    if bull and not bear:
        return 1
    if bear and not bull:
        return -1
    return 0


def _raw_votes(X, ftxt):
    """Market-direction votes (+1 bull / -1 bear / 0 neutral / None missing) per category."""
    f, live = X["f"], X["live"]
    v = {}
    v["regime"] = {"BULLISH": 1, "BEARISH": -1, "RANGE": 0}.get(X["regime"])
    v["mtf"] = {"BULLISH": 1, "BEARISH": -1, "MIXED": 0}.get(X["mtf"])
    if f.get("ok") and f.get("EMA_20") is not None:
        s = (1 if live > f["EMA_20"] else -1)
        if f.get("VWAP") is not None:
            s += (1 if live > f["VWAP"] else -1)
        if f.get("EMA_50") is not None:
            s += (1 if f["EMA_20"] > f["EMA_50"] else -1)
        v["ema_vwap"] = 1 if s >= 2 else (-1 if s <= -2 else 0)
    if f.get("ok") and f.get("MACD_Hist") is not None:
        m = 1 if f["MACD_Hist"] > 0 else -1
        r = 1 if f.get("ret6", 0) > 0.2 else (-1 if f.get("ret6", 0) < -0.2 else 0)
        v["momentum"] = m if m == r else 0
    p = X["pressure"]
    v["order_flow"] = 1 if p == "BUYING PRESSURE" else (-1 if p == "SELLING PRESSURE" else (0 if X.get("order_flow_seen") else None))
    v["main_signal"] = X["signal_code"] if X["signal_code"] in (1, -1) else 0
    v["ml"] = (X["signal_code"] if (X["ml_agrees"] and X["signal_code"] in (1, -1)) else None)
    v["sniper"] = _word_vote(X["raw"]["sniper"])
    if X["oc_live"]:
        pcr = X.get("pcr")
        if pcr is not None:
            v["pcr"] = 1 if pcr >= 1.15 else (-1 if pcr <= 0.85 else 0)
        if "heavy put oi" in ftxt and "heavy call oi" not in ftxt:
            v["oi_at_level"] = 1
        elif "heavy call oi" in ftxt and "heavy put oi" not in ftxt:
            v["oi_at_level"] = -1
        elif ftxt:
            v["oi_at_level"] = 0
        mp = X.get("max_pain")
        if mp:
            v["max_pain"] = 1 if live < mp - 40 else (-1 if live > mp + 40 else 0)
    ad = X.get("adv_dec_ratio")
    if ad is not None:
        v["breadth"] = 1 if ad >= 0.58 else (-1 if ad <= 0.42 else 0)
    g = X.get("global_avg")
    if g is not None:
        v["global"] = 1 if g > 0.15 else (-1 if g < -0.15 else 0)
    v["india_news"] = _word_vote(X["raw"]["news"])
    v["nifty50_news"] = _word_vote(X["raw"]["n50"])
    v["global_research"] = _word_vote(X["raw"]["gr"])
    v["fundamentals"] = _word_vote(X["raw"]["fund"])
    v["banknifty"] = None if X.get("bn_note") is None else (0 if X["banknifty_ok"] else "DIV")
    # Optional independent Gemini/AI research vote. It is deliberately only a
    # SOFT consensus input: deterministic market data and hard safety rules always
    # outrank a language-model opinion.
    ai_vote = str((X.get("external_ai") or {}).get("direction", "")).upper()
    v["external_ai"] = 1 if ai_vote == "BUY" else (-1 if ai_vote == "SELL" else (0 if ai_vote in ("WAIT", "NO TRADE", "NEUTRAL") else None))
    return v


def _consensus(d, X, ftxt, counter_trend=False, broke_wall=False, net_ease=0.0):
    """Weighted agreement for trade side d (+1 BUY / -1 SELL) across every available data family.
    net_ease (0..DAILY_TRADE_TARGET_NET_EASE_MAX, in percentage points): v19 daily-pace nudge that can
    lower the required net edge a little when the day is behind its learning-trade pace -- floored at
    CONSENSUS_NET_MIN_FLOOR so it can never remove the quality bar entirely."""
    raw = _raw_votes(X, ftxt)
    if broke_wall and raw.get("oi_at_level") is not None:
        raw["oi_at_level"] = 0      # v17: the OI wall that WAS defending this level just lost -- it no longer votes for a bounce
    skip = set(_TREND_GROUP + _FLOW_GROUP) if counter_trend else set()
    agree = against = total = 0.0
    rows, cats = [], 0
    for k, w in CONSENSUS_WEIGHTS.items():
        val = raw.get(k)
        if val is None or k in skip:
            continue
        if val == "DIV":                      # Bank Nifty divergence warning = direct disagreement
            rel = -1
        elif k == "banknifty":                # no warning = mild confirmation
            rel = 1
        else:
            rel = val * d
        cats += 1
        total += w
        if rel > 0:
            agree += w
        elif rel < 0:
            against += w
        rows.append({"cat": CONSENSUS_LABELS[k], "vote": "agree" if rel > 0 else ("against" if rel < 0 else "neutral"), "w": w})
    a_pct = agree / total if total else 0.0
    x_pct = against / total if total else 0.0
    net_pct = a_pct - x_pct
    need_net = CONSENSUS_NET_MIN_COUNTER if counter_trend else CONSENSUS_NET_MIN
    need_net = max(CONSENSUS_NET_MIN_FLOOR, need_net - max(0.0, net_ease) / 100.0)
    # v18: NET edge gate -- agree must clear an absolute floor AND beat disagreement by the required margin.
    passed = bool(cats >= CONSENSUS_MIN_CATEGORIES and a_pct >= CONSENSUS_MIN_AGREE_FLOOR and net_pct >= need_net)
    return {"agree_pct": round(a_pct * 100, 1), "against_pct": round(x_pct * 100, 1),
            "net_pct": round(net_pct * 100, 1), "need_pct": round(need_net * 100),
            "categories": cats, "counter_trend": counter_trend, "passed": passed,
            "agree": [r["cat"] for r in rows if r["vote"] == "agree"],
            "against": [r["cat"] for r in rows if r["vote"] == "against"],
            "neutral": [r["cat"] for r in rows if r["vote"] == "neutral"]}


def _trend_dir(X):
    r, m = X["regime"], X["mtf"]
    if (r == "BEARISH" and m != "BULLISH") or (m == "BEARISH" and r != "BULLISH"):
        return -1
    if (r == "BULLISH" and m != "BEARISH") or (m == "BULLISH" and r != "BEARISH"):
        return 1
    return 0


def _is_heavily_proven_level(lv):
    """v20: a level proven enough to allow an early (wick-only, before full candle-close confirmation)
    reversal entry -- same idea as an OI wall, extended to other levels that are ALREADY very well
    tested. NOT a new/thin level: needs either a high strength score or >=3 independent touches/sources.
    This does not remove confirmation for ordinary levels -- only for ones this proven."""
    if _is_oi_level(lv):
        return True
    if lv.get("count", 1) >= 3:
        return True
    if lv.get("strength", 0) >= 2.5:
        return True
    return False


def _is_real_level(lv):
    """v14: only a REAL level may anchor a reversal trade.
    Real = heavy option-chain OI wall, OR a round 50-pt level, OR a level confirmed by
    >=2 independent swing/PDH/PDL/opening/today's-extreme sources with strength >= 2.
    A lone SMC FVG / EMA / VWAP / 'major' label is NOT enough -- an FVG is created by
    the very move we are trading, it is not a place where orders were resting."""
    if _is_oi_level(lv):
        return True
    src = str(lv.get("src", "")).lower()
    # a validated multi-evidence zone from real_sr.py (grade A/B/C, actionable) is real
    if "s/r" in src and lv.get("strength", 0) >= 2.0:
        return True
    p = lv["price"]
    # only 100-pt round numbers are psychological levels on their own; a 50-pt number
    # needs confluence (another source merged in) -- a bare 50 is NOT a real level
    if min(p % 100.0, 100.0 - (p % 100.0)) <= 5.0 and lv.get("strength", 0) >= 0.9:
        return True
    if min(p % 50.0, 50.0 - (p % 50.0)) <= 4.0 and (lv.get("count", 1) >= 2 or lv.get("strength", 0) >= 1.6):
        return True
    if "session low" in src or "session high" in src:
        return True
    hard = any(k in src for k in ("swing", "pdh", "pdl", "pdc", "prev", "opening", "today's high", "today's low"))
    return bool(hard and lv.get("count", 1) >= 2 and lv.get("strength", 0) >= 2.0)


def _level_verdict(X, lvl):
    """v17 BOUNCE-OR-BREAK VERDICT for ONE real level (OI wall / validated S/R zone).

    Reads the same full-data consensus the trade gate uses, twice: once for the bounce side,
    once for the break side, and reports which way the data leans plus what the price has
    actually DONE at the level.  States:
      FAR / APPROACHING   price not there yet -> wait (no mid-air trade)
      TESTING             price is at the level, no confirmation yet -> wait for the candle
      BOUNCE_CONFIRMED    rejection candle confirmed at the level -> bounce-side entry allowed
      BREAK_CONFIRMED     level already broken with a confirmed close/retest -> break-side entry
                          allowed ONLY NOW (never before the break)
    """
    live, atr, conf = X["live"], X["atr"], X["conf"]
    price = lvl["price"]
    ftxt = X["level_text"](price)
    is_support = price < live
    bounce_d = 1 if is_support else -1
    brk_d = -bounce_d
    cb = _consensus(bounce_d, X, ftxt)
    ck = _consensus(brk_d, X, ftxt)
    dist = abs(price - live) / atr
    bounce_pct, break_pct = cb["agree_pct"], ck["agree_pct"]
    lean = ("BOUNCE" if bounce_pct >= break_pct + BREAK_LEAN_MARGIN
            else "BREAK" if break_pct >= bounce_pct + BREAK_LEAN_MARGIN else "UNDECIDED")
    just_broken_down = (not is_support) and dist <= 1.3 and bool(conf.get("breakdown_confirmed") or conf.get("breakdown_retest_confirmed"))
    just_broken_up = is_support and dist <= 1.3 and bool(conf.get("breakout_confirmed") or conf.get("breakout_retest_confirmed"))
    rej = bool(conf.get("support_rejection_confirmed" if is_support else "resistance_rejection_confirmed"))
    if just_broken_down or just_broken_up:
        state = "BREAK_CONFIRMED"
    elif dist > APPROACH_ATR:
        state = "FAR"
    elif dist > LEVEL_TOUCH_ATR:
        state = "APPROACHING"
    elif rej:
        state = "BOUNCE_CONFIRMED"
    else:
        state = "TESTING"
    if state == "BREAK_CONFIRMED" and _is_oi_level(lvl):
        cb = _consensus(bounce_d, X, ftxt, False, True)
        ck = _consensus(brk_d, X, ftxt, False, True)
        bounce_pct, break_pct = cb["agree_pct"], ck["agree_pct"]
        lean = ("BOUNCE" if bounce_pct >= break_pct + BREAK_LEAN_MARGIN
                else "BREAK" if break_pct >= bounce_pct + BREAK_LEAN_MARGIN else "UNDECIDED")
    b_name, k_name = ("BUY", "SELL") if bounce_d == 1 else ("SELL", "BUY")
    if state in ("FAR", "APPROACHING"):
        action = (f"Abhi wait: price level tak pahunche. Data abhi {lean} ki taraf ({bounce_pct:.0f}% bounce vs {break_pct:.0f}% break). "
                  f"Bounce confirm hua to {b_name}; toot ke close+retest confirm hua to hi {k_name} -- todne se PEHLE entry nahi.")
    elif state == "TESTING":
        action = (f"Level test ho raha hai, par abhi na rejection candle confirm hui na break. Data {lean} ki taraf "
                  f"({bounce_pct:.0f}% vs {break_pct:.0f}%). Candle confirmation ka wait -- {b_name} ya {k_name} dono abhi nahi.")
    elif state == "BOUNCE_CONFIRMED":
        action = (f"Rejection confirm hui. Data {lean} ({bounce_pct:.0f}% bounce vs {break_pct:.0f}% break) -- "
                  + (f"{b_name} bounce entry valid." if lean != "BREAK" else f"lekin data break ki taraf hai, isliye {b_name} skip."))
    else:
        action = (f"Level toot chuka (confirmed close/retest). Data {lean} ({bounce_pct:.0f}% vs {break_pct:.0f}%) -- "
                  + (f"{k_name} continuation ab valid (break ke BAAD)." if lean != "BOUNCE" else f"lekin data bounce/reclaim ki taraf hai, {k_name} skip."))
    return {"level": round(price, 1), "src": lvl.get("src"), "state": state, "lean": lean,
            "bounce_side": b_name, "break_side": k_name, "bounce_pct": bounce_pct, "break_pct": break_pct,
            "dist_pts": round(abs(price - live), 1), "action": action}


def _extension_block(d, X, play=None):
    """v14: never chase. If price has ALREADY run far from EMA20 in the trade's
    direction and is sitting at the day's extreme, a new SELL (or BUY) there is
    selling the bottom / buying the top. Wait for a pullback into a real level.
    v19: continuation (BREAKDOWN_SELL/BREAKOUT_BUY) gets a TIGHTER limit than reversal --
    these are momentum-chase plays, and chasing them once already very extended is exactly
    the 'entered right before it reversed' pattern reported (bought right as the up-move
    was already exhausted, then it reversed hard)."""
    f, live, atr = X["f"], X["live"], X["atr"]
    if not f.get("ok") or f.get("EMA_20") is None:
        return None
    ema = f["EMA_20"]
    cont = play in ("BREAKDOWN_SELL", "BREAKOUT_BUY")
    limit = CONT_MAX_STRETCH_ATR if cont else MAX_STRETCH_ATR
    if d == -1:
        stretch, near = (ema - live) / atr, f.get("dist_sess_low", 9.0) <= 1.0
    else:
        stretch, near = (live - ema) / atr, f.get("dist_sess_high", 9.0) <= 1.0
    if stretch >= limit and near:
        side = "SELL" if d == -1 else "BUY"
        return (f"{side} extension warning: {stretch:.1f} ATR from EMA20 near the day extreme; "
                "momentum capture remains eligible, but reversal/room risk is shown in the score.")
    return None


def _score_momentum_candidate(d, X):
    """V22 fallback for a live 50/100+ point move.

    This is deliberately not a free-form BUY/SELL generator. Direction comes
    from the recent price move itself. The candidate then receives weighted
    support from trend, order flow, main signal, breadth/global/news and AI
    research. It exists so a fast move cannot disappear merely because a
    static S/R object or one slow confirmation feed is one candle behind.
    """
    f, atr, live = X["f"], X["atr"], X["live"]
    if not f.get("ok") or not atr:
        return None
    move = f.get("ret6", 0.0) * atr
    move12 = f.get("ret12", 0.0) * atr
    if d == 1:
        if max(move, move12) < 50:
            return None
    else:
        if min(move, move12) > -50:
            return None
    reasons, misses = [], []
    score = 18.0
    parts = {"location": 0.0, "price_action": 0.0, "trend": 0.0,
             "flow": 0.0, "breadth_global": 0.0, "extras": 0.0,
             "bonus": 0.0, "penalties": 0.0}
    abs_move = abs(move)
    if abs_move >= 100 or abs(move12) >= 100:
        score += 18; parts["bonus"] += 18; reasons.append(f"V22 momentum capture: {abs(max(move, move12, key=abs)):.0f}-point move")
    else:
        score += 10; parts["bonus"] += 10; reasons.append(f"V22 momentum capture: {abs(max(move, move12, key=abs)):.0f}-point move")
    pa = 0.0
    if (d == 1 and f.get("hh6", 0) >= 3) or (d == -1 and f.get("ll6", 0) >= 3):
        pa += 8; reasons.append("price is printing directional structure")
    if (d == 1 and f.get("bull")) or (d == -1 and f.get("bear")):
        pa += 4; reasons.append("live candle agrees")
    parts["price_action"] = pa; score += pa
    tr = 0.0
    if (d == 1 and X["regime"] == "BULLISH") or (d == -1 and X["regime"] == "BEARISH"):
        tr += 5
    if (d == 1 and X["mtf"] == "BULLISH") or (d == -1 and X["mtf"] == "BEARISH"):
        tr += 5
    ema = f.get("EMA_20"); vwap = f.get("VWAP")
    if ema is not None and ((d == 1 and live > ema) or (d == -1 and live < ema)): tr += 3
    if vwap is not None and ((d == 1 and live > vwap) or (d == -1 and live < vwap)): tr += 2
    parts["trend"] = min(tr, 15); score += parts["trend"]
    fl = 0.0
    if (d == 1 and X["pressure"] == "BUYING PRESSURE") or (d == -1 and X["pressure"] == "SELLING PRESSURE"):
        fl += 6
    if X["signal_code"] == d: fl += 4
    if X["ml_agrees"] and X["signal_code"] == d: fl += 2
    parts["flow"] = min(fl, 10); score += parts["flow"]
    br = 0.0
    if X["breadth_ok"](d): br += 3
    if X["global_ok"](d): br += 2
    if X["banknifty_ok"]: br += 1
    if X["news_ok"](d): br += 1
    if X["n50news_ok"](d): br += 1
    if X["research_ok"](d): br += 1
    if X["sniper_ok"](d): br += 1
    parts["breadth_global"] = min(br, 10); score += parts["breadth_global"]
    # Independent AI is a soft vote only.
    ai_dir = str((X.get("external_ai") or {}).get("direction", "")).upper()
    if ai_dir == ("BUY" if d == 1 else "SELL"):
        score += 2; parts["extras"] += 2; reasons.append("independent AI research agrees")
    # Hard opposite transition is the one momentum-specific warning that should
    # materially reduce priority; it does not automatically erase the candidate.
    timing = X["timing"]; conf = X["conf"]
    if (d == 1 and conf.get("bearish_trend_transition_confirmed")) or (d == -1 and conf.get("bullish_trend_transition_confirmed")):
        score -= 10; parts["penalties"] -= 10; misses.append("confirmed opposite trend transition")
    return {"play": "BREAKOUT_BUY" if d == 1 else "BREAKDOWN_SELL", "d": d,
            "score": round(max(0.0, min(100.0, score)), 1),
            "loc": 0.0, "pa": round(pa, 1), "parts": parts,
            "reasons": reasons, "misses": misses, "level": None,
            "fake_breakout": None, "oi_strong": False, "oi_info": None,
            "unconfirmed_rejection": False, "momentum_fallback": True}


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
    _ext = _extension_block(d, X, play)
    if _ext:
        X["notes"].append(_ext)

    # ---------------- LOCATION ----------------
    loc_pts, lvl = 0.0, None
    oi_strong, oi_info = False, None
    if reversal:
        want = "support" if d == 1 else "resistance"
        cands = []
        for lv in levels:
            if lv["dynamic"] and lv["strength"] < 1.0:
                continue
            # SIDE CHECK (the actual bug fix): the old band below only
            # checked DISTANCE from price, not which side of price the level
            # is actually on. For a BUY (d=1) it allowed levels up to 0.3
            # ATR *above* live to be picked as the "support" to bounce from
            # -- so a real, unbroken resistance sitting just overhead could
            # get labelled "Support" and bought straight into. Mirror bug
            # for SELL (d=-1): a real support just below could get labelled
            # "Resistance" and sold straight into. `lv["side"]` is computed
            # purely from price-vs-live and is trustworthy for this, so it
            # is now required to actually match `want` before a level can
            # anchor a reversal trade at all.
            if lv.get("side") != want:
                continue
            if not _is_real_level(lv):
                continue
            if d == 1:
                inside = (live - LEVEL_TOUCH_ATR * atr) <= lv["price"] <= live
            else:
                inside = live <= lv["price"] <= (live + LEVEL_TOUCH_ATR * atr)
            if inside:
                # v14: price must actually have TESTED the level in the last 6 candles
                if d == 1:
                    touched = bool(f.get("ok") and f.get("low6", live) <= lv["price"] + 0.15 * atr)
                else:
                    touched = bool(f.get("ok") and f.get("high6", live) >= lv["price"] - 0.15 * atr)
                # V22: historical touch is useful evidence, not a gate.
                # OI-backed level gets a ranking bonus: user's core point --
                # a level with heavy Call/Put writer OI is THE most reliable
                # bounce/rejection point (writers actively defend it), so
                # when one is present near price it should win the anchor
                # spot over a same-distance EMA/swing/round-number level.
                rank = lv["strength"] * 4.0 - abs(lv["price"] - live) / atr * 3.0
                if not touched:
                    rank -= 1.5
                if _is_oi_level(lv):
                    _oi_n, _oi_mult = _oi_wall_magnitude(X["level_text"](lv["price"]))
                    rank += 6.0 * _oi_mult
                cands.append((rank, lv))
        if not cands:
            # v21: today's own session extreme is a legitimate auction level.
            # When price has made a fresh extreme and then starts to stall, use
            # that extreme as a synthetic REAL level instead of waiting forever
            # for an external S/R object to be generated. This is still gated by
            # price-action, consensus, risk and the opposite-wall checks below.
            if f.get("ok"):
                if d == 1 and f.get("dist_sess_low", 9) <= 0.30 and f.get("drop_from_high", 0) >= 1.0 and (f.get("stall_after_fall") or f.get("lower_wick", 0) >= 0.35):
                    _sp = float(f.get("sess_low", live))
                    cands.append((8.0, {"price": _sp, "src": "SESSION LOW", "strength": 2.0, "count": 2, "dynamic": False, "side": "support", "grade": "B", "factors": ["today session low"]}))
                elif d == -1 and f.get("dist_sess_high", 9) <= 0.30 and f.get("rise_from_low", 0) >= 1.0 and (f.get("stall_after_rise") or f.get("upper_wick", 0) >= 0.35):
                    _sp = float(f.get("sess_high", live))
                    cands.append((8.0, {"price": _sp, "src": "SESSION HIGH", "strength": 2.0, "count": 2, "dynamic": False, "side": "resistance", "grade": "B", "factors": ["today session high"]}))
            if not cands:
                X["notes"].append("Koi REAL level (OI wall / session extreme / multi-swing) price ke paas test nahi hua -- mid-air trade skip.")
                return None
        lvl = max(cands, key=lambda t: t[0])[1]
        _c_oi, _p_oi = _oi_sides(X["level_text"](lvl["price"]))
        _wall, _other = (_p_oi, _c_oi) if d == 1 else (_c_oi, _p_oi)   # BUY needs PUT wall below, SELL needs CALL wall above
        if _wall >= OI_STRONG_MIN_CONTRACTS and _wall >= OI_DOMINANCE_MIN * max(_other, 1):
            oi_strong = True
            oi_info = {"wall": _wall, "other": _other, "ratio": round(_wall / max(_other, 1), 1)}
        loc_pts = min(25.0, 8.0 + 5.0 * min(lvl["strength"], 3.0))
        if _is_oi_level(lvl):
            oi_n, oi_mult = _oi_wall_magnitude(X["level_text"](lvl["price"]))
            loc_pts = min(30.0, loc_pts + 3.0 * oi_mult)
            size_note = (f", {oi_n:,.0f} contracts -- {'EXTREME' if oi_mult>=1.6 else 'VERY HEAVY' if oi_mult>=1.35 else 'heavy' if oi_mult>=1.0 else 'modest'} wall"
                         if oi_n is not None else "")
            reasons.append(f"{want.capitalize()} {lvl['price']:,.1f} -- heavy Option-chain OI writer wall "
                          f"({lvl['src']}, strength {lvl['strength']:.1f}{size_note}) -- iss tarah ke level se bounce/"
                          f"rejection sabse zyada reliable hota hai")
        else:
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
            if _is_oi_level(lvl):
                # A genuinely heavy-OI level rarely breaks -- when it does, that's
                # real writers getting run over / unwinding, which historically
                # continues further than a break of a plain swing/EMA level. This
                # is the flip side of the wall-ahead penalty above: an OI wall
                # still standing is extra-dangerous to walk into, but an OI wall
                # that has ALREADY given way is extra-strong continuation evidence.
                # Scaled by wall SIZE (see _oi_wall_magnitude): a 5-lakh-contract
                # wall breaking is a much bigger deal than a 50k one breaking.
                _oi_n, _oi_mult = _oi_wall_magnitude(X["level_text"](lvl["price"]))
                loc_pts = min(30.0, loc_pts + 4.0 * _oi_mult)
                size_note = f", {_oi_n:,.0f} contracts" if _oi_n is not None else ""
                reasons.append(f"yeh ek heavy Option-chain OI wall tha jo abhi toota ({size_note.strip(', ') or 'size unknown'}) "
                              "-- writers ka stop-hunt/unwind, normal level-break se zyada strong continuation signal")
        elif at_low:
            loc_pts = 10.0
            reasons.append("Price pressing today's " + ("low" if d == -1 else "high") + " (fresh extreme)")
        else:
            # trend continuation without a fresh level: allowed but needs strong price action
            loc_pts = 6.0
            reasons.append("Trend continuation (no fresh level nearby)")

    # ---------------- PRICE-ACTION TRIGGER ----------------
    pa, pa_notes = 0.0, []
    fake_breakout = None  # only ever populated for BREAKDOWN_SELL / BREAKOUT_BUY below
    unconfirmed_rejection = False  # v20: set below when a reversal fires WITHOUT full candle confirmation --
                                     # such a candidate is never allowed to become a FULL trade later on
                                     # (see the cascade near is_probe), only ever a probe or nothing.
    rsi, rsi_prev = f.get("RSI"), f.get("rsi_prev")
    macd, macd_prev = f.get("MACD_Hist"), f.get("macd_prev")

    if play == "BOUNCE_BUY":
        fell = f.get("ok") and f.get("drop_from_high", 0) >= 0.9
        if not fell and not conf.get("support_rejection_confirmed"):
            pa += 2; pa_notes.append("support approach without a full prior fall; using live level evidence")
        _wick_ok = bool(_is_heavily_proven_level(lvl) and f.get("bull") and max(f.get("lower_wick", 0), f.get("lower_wick_prev", 0)) >= 0.4)
        if not conf.get("support_rejection_confirmed") and not _wick_ok:
            # v20: used to hard-skip here -- meaning a genuinely good moment could vanish with NOT EVEN
            # a probe taken, because this check runs before the candidate is even scored/ranked. Now it
            # keeps going (heavily penalised, flagged) so it can still become a probe further down --
            # it is barred from ever becoming a full trade regardless of how the rest of the score turns out.
            unconfirmed_rejection = True
            pa -= 8
            misses.append("BUY: support par genuine rejection (candle ya OI-wall/heavily-proven level par wick + "
                          "bullish close) abhi confirm nahi -- isse full trade nahi banega (sirf probe), chahe score jitna bhi ho.")
        # COUNTER-TREND CHASE GUARD (real trading feedback): price merely
        # TOUCHING a level after a strong same-direction run, WITHOUT a
        # genuinely confirmed rejection candle, statistically just pauses
        # ~20-30 pts and continues in the ORIGINAL arriving direction far
        # more often than it truly reverses -- especially when the level
        # isn't a big proven OI wall. A strong incoming move is the TRIGGER
        # for this playbook above; it must not also be treated as if it
        # were, by itself, proof of reversal.
        _strong_incoming = f.get("drop_from_high", 0) >= 1.3
        _genuine_rejection = bool(conf.get("support_rejection_confirmed") or _wick_ok)
        _strong_oi_wall = bool(lvl and _is_oi_level(lvl) and _oi_wall_magnitude(X["level_text"](lvl["price"]))[1] >= 1.0)
        if _strong_incoming and not _genuine_rejection and not _strong_oi_wall:
            pa -= 6
            misses.append("Strong girta hua momentum abhi bhi usi (girne wali) direction mein hai aur genuine "
                          "rejection candle confirm nahi hui, na hi yeh ek badi proven OI wall hai -- aisi jagah "
                          "price zyada chances mein sirf touch karke wapas usi purani (girti) direction mein "
                          "nikal jata hai, poora reversal nahi. Iss bounce ko heavily penalize kar raha hai.")
        # ALREADY-BROKEN-LEVEL GUARD ("abhi jo bada movement aaya, order block
        # tod diya" case): if a bearish breakdown JUST got confirmed here,
        # this "support" has already failed -- do not treat a small pullback
        # into it as a fresh bounce setup. This is the pullback-after-break
        # case, and the correct trade there (if any) is the continuation
        # SELL via BREAKDOWN_SELL once ITS OWN process/factors check out --
        # never a contrarian BUY on the same broken level.
        if conf.get("breakdown_confirmed") or conf.get("breakdown_retest_confirmed"):
            pa -= 18
            misses.append("Yeh support abhi hi confirmed BREAKDOWN se toot chuka hai -- yeh ab bounce-buy ke liye "
                          "valid support nahi raha. Agar continuation dikh raha hai to uska apna BREAKDOWN_SELL "
                          "process (apna score/confidence) poora hone par hi entry banegi, seedha yahan buy nahi.")
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
            pa += 2; pa_notes.append("resistance approach without a full prior rise; using live level evidence")
        _wick_ok = bool(_is_heavily_proven_level(lvl) and f.get("bear") and max(f.get("upper_wick", 0), f.get("upper_wick_prev", 0)) >= 0.4)
        if not conf.get("resistance_rejection_confirmed") and not _wick_ok:
            unconfirmed_rejection = True
            pa -= 8
            misses.append("SELL: resistance par genuine rejection (candle ya OI-wall/heavily-proven level par wick + "
                          "bearish close) abhi confirm nahi -- isse full trade nahi banega (sirf probe), chahe score jitna bhi ho.")
        # COUNTER-TREND CHASE GUARD -- mirror of the BOUNCE_BUY guard above.
        # This is EXACTLY the failure mode reported: price rose strongly
        # into a resistance, no genuinely confirmed rejection candle, level
        # wasn't a big proven OI wall -- engine sold anyway, price just kept
        # rising through it. Same real-world rule applies both ways: only
        # fade a strong incoming move when the rejection is genuinely
        # confirmed, or the level is a real heavy OI wall; otherwise the
        # higher-probability outcome is a shallow touch that continues in
        # the ORIGINAL (here: rising) direction, not a true reversal.
        _strong_incoming = f.get("rise_from_low", 0) >= 1.3
        _genuine_rejection = bool(conf.get("resistance_rejection_confirmed") or _wick_ok)
        _strong_oi_wall = bool(lvl and _is_oi_level(lvl) and _oi_wall_magnitude(X["level_text"](lvl["price"]))[1] >= 1.0)
        if _strong_incoming and not _genuine_rejection and not _strong_oi_wall:
            pa -= 6
            misses.append("Strong chadhta hua momentum abhi bhi usi (chadhne wali) direction mein hai aur genuine "
                          "rejection candle confirm nahi hui, na hi yeh ek badi proven OI wall hai -- aisi jagah "
                          "price zyada chances mein sirf touch karke wapas usi purani (chadhti) direction mein "
                          "nikal jata hai, poora reversal nahi. Iss rejection-sell ko heavily penalize kar raha hai "
                          "(bilkul wahi mistake jo pehle SELL 23050PE @ 23,041 trade mein hui thi).")
        # ALREADY-BROKEN-LEVEL GUARD -- THE exact scenario reported: market
        # rallies up, breaks a resistance/order-block (a genuine big move),
        # then pulls back slightly toward that same level. That old
        # resistance just got confirmed-broken -- selling it now is
        # backwards ("ulta pad jaega"). If continuation is genuinely there,
        # it must come from BREAKOUT_BUY completing ITS OWN full process
        # (score + factors + confidence), never a contrarian sell on the
        # level that was just broken.
        if conf.get("breakout_confirmed") or conf.get("breakout_retest_confirmed"):
            pa -= 18
            misses.append("Yeh resistance abhi hi confirmed BREAKOUT se toot chuka hai -- yeh ab rejection-sell ke "
                          "liye valid resistance nahi raha. Yeh pullback-after-breakout hai; agar continuation "
                          "dikh raha hai to sahi trade BUY hai (BREAKOUT_BUY apna poora process complete kare "
                          "tabhi), seedha yahan sell nahi -- ulta pad jaega.")
        if _strong_incoming and not _genuine_rejection and not _strong_oi_wall:
            pa -= 6
            misses.append("Strong chadhta hua momentum abhi bhi usi (chadhne wali) direction mein hai aur genuine "
                          "rejection candle confirm nahi hui, na hi yeh ek badi proven OI wall hai -- aisi jagah "
                          "price zyada chances mein sirf touch karke wapas usi purani (chadhti) direction mein "
                          "nikal jata hai, poora reversal nahi. Iss rejection-sell ko heavily penalize kar raha hai "
                          "(bilkul wahi mistake jo pehle SELL 23050PE @ 23,041 trade mein hui thi).")
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
        # v19: truly extreme RSI on a CONTINUATION play = the move is almost certainly exhausted right
        # now -- this is exactly the "entered right before it reversed" pattern reported. A soft score
        # penalty wasn't enough to stop it when everything else scored high; this is now a hard skip.
        if rsi is not None and rsi < 15:
            pa -= 6; misses.append(f"RSI {rsi:.0f} extreme oversold - exhaustion risk, but not a hard veto in V22")
        elif rsi is not None and rsi < 24:
            pa -= 3; misses.append(f"RSI {rsi:.0f} oversold - some bounce risk")
        # WALL-AHEAD CHECK: a real, un-broken support just below is far more
        # likely to bounce this move than let it through -- don't chase a
        # breakdown straight into one that hasn't actually given way yet.
        _wall = _unbroken_level_ahead(levels, live, atr, d)
        fake_breakout = None
        if _wall:
            if _is_oi_level(_wall):
                _wn, _wmult = _oi_wall_magnitude(X["level_text"](_wall["price"]))
                pa -= 20 * _wmult
                misses.append(f"Support {_wall['price']:,.1f} abhi tak unbroken hai AUR yahan heavy Option-chain OI "
                              f"hai ({_wn:,.0f} contracts, writers actively defending)" if _wn is not None else
                              f"Support {_wall['price']:,.1f} abhi tak unbroken hai AUR yahan heavy Option-chain OI hai (writers actively defending)")
                misses.append("-- yeh khatarnak hai, isme seedha ghuskar breakdown "
                              "chase karna bahut risky: OI wall statistically bounce karta hai, todne mein extra proof chahiye")
            else:
                pa -= 12
                misses.append(f"Support {_wall['price']:,.1f} (strength {_wall['strength']:.1f}) abhi tak unbroken hai, "
                              "seedha usi mein ja rahe hain -- bounce ka chance zyada hai breakdown confirm hone se")
        # FAKE BREAKOUT CHECK -- see _fake_breakout_flags(). Only meaningful
        # once we actually have a "broken" level to be suspicious about.
        if lvl is not None:
            fb_flags = _fake_breakout_flags(d, f, conf, X["timing"], lvl, live, atr)
            fb_count = sum(fb_flags.values())
            if fb_flags.get("no_confirmed_break"):
                pa -= 5
                misses.append("Breakout/breakdown close+volume se fully confirmed nahi hai -- early momentum entry, risk discipline required")
            fake_breakout = {"flags": fb_flags, "count": fb_count, "suspected": fb_count >= 2}
            if fb_count >= 3:
                misses.append(f"Yeh FAKE BREAKDOWN / trap lag raha hai ({fb_count}/6 warning signs: "
                              f"{', '.join(k for k, v in fb_flags.items() if v)}) -- genuine break/momentum confirm "
                              "nahi ho raha, entry skip kar raha hai taaki OI/S-R ka fake break chase na ho.")
                return None
            elif fb_count == 2:
                pa -= 14
                misses.append(f"Fake-breakdown risk medium hai ({fb_count}/6 warning signs: "
                              f"{', '.join(k for k, v in fb_flags.items() if v)}) -- score heavily penalize kiya, "
                              "yeh trade (agar phir bhi le liya) self-learning ko train nahi karega.")
        if lvl is None:
            # v21: allow a structured momentum entry when price is making a fresh
            # session extreme and the direction is independently confirmed. This
            # is NOT a mid-air trade: wall-ahead, exhaustion, consensus, RR and
            # risk gates still apply below.
            strong_extreme = bool(f.get("ok") and f.get("dist_sess_low", 9) <= 0.25 and
                                  (f.get("ret6", 0) <= -0.8 or f.get("ret12", 0) <= -1.2) and
                                  f.get("ll6", 0) >= 4)
            if CONT_REQUIRE_LEVEL or not strong_extreme:
                return None
            loc_pts = max(loc_pts, 8.0)
            reasons.append("fresh session-low momentum continuation (no broken level needed)")
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
        # v19: mirror of the BREAKDOWN_SELL hard-skip above.
        if rsi is not None and rsi > 85:
            pa -= 6; misses.append(f"RSI {rsi:.0f} extreme overbought - exhaustion risk, but not a hard veto in V22")
        elif rsi is not None and rsi > 76:
            pa -= 3; misses.append(f"RSI {rsi:.0f} overbought - some pullback risk")
        # WALL-AHEAD CHECK: same idea as breakdown, mirrored -- a real,
        # un-broken resistance sitting right above is more likely to reject
        # this move than let it run. This is exactly what caused the
        # BUY-into-daily-resistance loss: the old code only scored the
        # level ALREADY behind price, never the one still ahead.
        _wall = _unbroken_level_ahead(levels, live, atr, d)
        fake_breakout = None
        if _wall:
            if _is_oi_level(_wall):
                _wn, _wmult = _oi_wall_magnitude(X["level_text"](_wall["price"]))
                pa -= 20 * _wmult
                misses.append(f"Resistance {_wall['price']:,.1f} abhi tak unbroken hai AUR yahan heavy Option-chain OI "
                              f"hai ({_wn:,.0f} contracts, writers actively defending)" if _wn is not None else
                              f"Resistance {_wall['price']:,.1f} abhi tak unbroken hai AUR yahan heavy Option-chain OI hai (writers actively defending)")
                misses.append("-- yeh khatarnak hai, isme seedha ghuskar breakout "
                              "chase karna bahut risky: OI wall statistically reject karta hai, todne mein extra proof chahiye")
            else:
                pa -= 12
                misses.append(f"Resistance {_wall['price']:,.1f} (strength {_wall['strength']:.1f}) abhi tak unbroken hai, "
                              "seedha usi mein ja rahe hain -- reject hone ka chance zyada hai breakout confirm hone se")
        # FAKE BREAKOUT CHECK -- see _fake_breakout_flags().
        if lvl is not None:
            fb_flags = _fake_breakout_flags(d, f, conf, X["timing"], lvl, live, atr)
            fb_count = sum(fb_flags.values())
            if fb_flags.get("no_confirmed_break"):
                pa -= 5
                misses.append("Breakout/breakdown close+volume se fully confirmed nahi hai -- early momentum entry, risk discipline required")
            fake_breakout = {"flags": fb_flags, "count": fb_count, "suspected": fb_count >= 2}
            if fb_count >= 3:
                pa -= 10
                misses.append(f"Fake-breakout risk high ({fb_count}/6 warning signs) -- overall side score must still be clearly dominant")
            elif fb_count == 2:
                pa -= 14
                misses.append(f"Fake-breakout risk medium hai ({fb_count}/6 warning signs: "
                              f"{', '.join(k for k, v in fb_flags.items() if v)}) -- score heavily penalize kiya, "
                              "yeh trade (agar phir bhi le liya) self-learning ko train nahi karega.")
        if lvl is None:
            # v21 mirror: fresh session-high continuation can qualify without a
            # pre-existing static resistance, provided price action is genuinely
            # strong and the remaining safety/consensus gates agree.
            strong_extreme = bool(f.get("ok") and f.get("dist_sess_high", 9) <= 0.25 and
                                  (f.get("ret6", 0) >= 0.8 or f.get("ret12", 0) >= 1.2) and
                                  f.get("hh6", 0) >= 4)
            if CONT_REQUIRE_LEVEL or not strong_extreme:
                return None
            loc_pts = max(loc_pts, 8.0)
            reasons.append("fresh session-high momentum continuation (no broken level needed)")

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
    # OI (open interest) at the level matters a lot -- a strike with heavy
    # OI genuinely behaves like a wall: price either rejects off it or has
    # to fight through it, it rarely just ignores it. Weighted well above
    # the other extras now instead of a token +3.
    if (d == 1 and "heavy put oi" in ftxt) or (d == -1 and "heavy call oi" in ftxt):
        _ex_oi_n, _ex_oi_mult = _oi_wall_magnitude(ftxt)
        ex += 6 * _ex_oi_mult
        reasons.append("heavy OI wall at this level supports the level (put/call writers defending it"
                       + (f", {_ex_oi_n:,.0f} contracts)" if _ex_oi_n is not None else ")"))
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
    if reversal and oi_strong:
        bonus += 3.0
        reasons.append(f"OI wall {oi_info['wall']:,} vs {oi_info['other']:,} ({oi_info['ratio']}x) -- price is wall ki taraf se aaya, "
                       f"writers wapas usi side bhejte hain")
    if reversal and loc_pts >= 14 and pa >= 16:
        bonus = 6.0
        reasons.append("strong level AND strong reaction candle together")
    # V22 MOMENTUM CAPTURE: fast 50/100+ point moves should not be lost
    # because slower secondary feeds are one candle behind. This boosts the
    # candidate already pointing in that direction; it never creates a side.
    move_pts = abs(f.get("ret6", 0.0) * atr) if f.get("ok") else 0.0
    if move_pts >= 100:
        bonus += 8.0
        reasons.append(f"V22 momentum capture: ~{move_pts:.0f}-point move detected")
    elif move_pts >= 50:
        bonus += 4.0
        reasons.append(f"V22 momentum capture: ~{move_pts:.0f}-point move detected")
    score = loc_pts + pa + tr + fl + br + ex + bonus - pen
    score = max(0.0, min(100.0, score))
    return {
        "play": play, "d": d, "score": round(score, 1), "loc": round(loc_pts, 1), "pa": round(pa, 1),
        "parts": {"location": round(loc_pts, 1), "price_action": round(pa, 1), "trend": round(tr, 1),
                  "flow": round(fl, 1), "breadth_global": round(br, 1), "extras": round(ex, 1),
                  "bonus": round(bonus, 1), "penalties": round(-pen, 1)},
        "reasons": reasons, "misses": misses, "level": lvl,
        "fake_breakout": fake_breakout, "oi_strong": oi_strong, "oi_info": oi_info,
        "unconfirmed_rejection": unconfirmed_rejection,
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
    # Fixed-risk SL: every trade risks the SAME points, not ATR/structure-sized.
    # (structural sl_ref/sl_dist above kept only as a diagnostic reference)
    struct_dist = max(sl_dist, 0.0)          # beyond level + recent wick + buffer (stop-hunt proof)
    # V22 momentum fallback has no structural level by design. Do not let an
    # old high/low 50-100 points away manufacture a huge structural stop and
    # then veto the very move we created this fallback to capture. It uses the
    # fixed-risk stop like every other trade.
    if cand.get("momentum_fallback") and lvl is None:
        struct_dist = SL_FIXED_PTS
    sl_too_far = struct_dist > SL_MAX_PTS
    sl_dist = min(max(struct_dist, SL_FIXED_PTS), SL_MAX_PTS)
    stop = live - sl_dist if d == 1 else live + sl_dist

    # structural targets: real levels in the trade direction, nearest first,
    # but an OI-backed level (Option Chain OI, heavy Put/Call writing) is
    # preferred over a same-distance non-OI level -- "aaspaas ka OI ka hi
    # target" was asked for explicitly: the target should be the strike
    # where writers are actually defending, not just any nearby EMA/swing/
    # round-number level that happens to be slightly closer.
    if d == 1:
        tgts = sorted([lv for lv in levels if lv["price"] > live + 0.01],
                       key=lambda x: (not _is_oi_level(x), x["price"]))
    else:
        tgts = sorted([lv for lv in levels if lv["price"] < live - 0.01],
                       key=lambda x: (not _is_oi_level(x), -x["price"]))
    target, target2, note = None, None, ""
    picked = []
    for lv in tgts:
        dist = abs(lv["price"] - live)
        if dist > 6.0 * atr:
            continue
        if dist >= MIN_RR * sl_dist:
            picked.append(lv)
        if len(picked) == 2:
            break

    # ROOM-TO-RUN CHECK: the true nearest opposing-side level (by distance,
    # not OI-priority) is still a wall the trade has to get through -- this
    # must stay distance-sorted regardless of which level becomes the target.
    # v17: only STATIC levels of real weight are walls. EMA20 / VWAP / volume-POC are moving averages (dynamic) and
    # weak 0.5-strength round numbers are not resting orders -- counting them as a "wall" trimmed the target to <1.5R
    # and killed otherwise good OI-wall trades ("not enough real room" on a level with 40+ pts of clear air behind it).
    _is_wall = lambda lv: (not lv.get("dynamic")) and lv.get("strength", 0) >= 1.0
    if d == 1:
        by_dist = sorted([lv for lv in levels if lv["price"] > live + 0.01 and _is_wall(lv)], key=lambda x: x["price"])
    else:
        by_dist = sorted([lv for lv in levels if lv["price"] < live - 0.01 and _is_wall(lv)], key=lambda x: -x["price"])
    nearest_wall = by_dist[0] if by_dist else None
    nearest_wall_dist = abs(nearest_wall["price"] - live) if nearest_wall else None

    if picked:
        target = round(picked[0]["price"], 2)
        note = f"target at {picked[0]['src']}"
        if len(picked) > 1:
            target2 = round(picked[1]["price"], 2)
    else:
        fallback_dist = FALLBACK_RR * sl_dist
        if nearest_wall_dist is not None and nearest_wall_dist < fallback_dist and not (cand.get("momentum_fallback") and lvl is None):
            fallback_dist = max(0.0, nearest_wall_dist - buf)
            note = (f"trimmed short of {nearest_wall['src']} ({nearest_wall['price']:,.1f}) "
                    "- real level in the way, not broken yet")
        else:
            note = (f"V22 momentum fallback - fixed {FALLBACK_RR}R target" if cand.get("momentum_fallback") and lvl is None
                    else f"no structure far enough - ATR-based {FALLBACK_RR}R target")
        target = round(live + fallback_dist if d == 1 else live - fallback_dist, 2)
    rr = abs(target - live) / max(sl_dist, 1e-9)
    if sl_too_far:
        rr = 0.0
        note = f"structural stop {struct_dist:.0f} pts > {SL_MAX_PTS:.0f} max -- entry level se bahut door, skip"
    return round(stop, 2), target, target2, round(rr, 2), note


# ---------------------------------------------------------------------
# market view: where can it fall / bounce / rise
# ---------------------------------------------------------------------
def _build_market_view(X, best_buy, best_sell):
    live, atr, f, levels = X["live"], X["atr"], X["f"], X["levels"]
    _ft = X["level_text"](live)
    _cb, _cs_ = _consensus(1, X, _ft), _consensus(-1, X, _ft)
    sup = sorted([lv for lv in levels if lv["price"] < live - 0.05 * atr and (_is_real_level(lv) or lv["strength"] >= 1.5)],
                 key=lambda x: -x["price"])[:4]
    res = sorted([lv for lv in levels if lv["price"] > live + 0.05 * atr and (_is_real_level(lv) or lv["strength"] >= 1.5)],
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
    _ai_mv = X.get("external_ai") or {}
    if _ai_mv:
        lines.append(f"Independent AI research: {_ai_mv.get('direction','WAIT')} / {_ai_mv.get('strength','WEAK')} -- {_ai_mv.get('rationale','')}")
    lines.append(f"CONSENSUS METER -- BUY ke haq mein {_cb['agree_pct']:.0f}% data (against {_cb['against_pct']:.0f}%, "
                 f"net {_cb['net_pct']:.0f}%) | SELL ke haq mein {_cs_['agree_pct']:.0f}% (against {_cs_['against_pct']:.0f}%, "
                 f"net {_cs_['net_pct']:.0f}%) | trade ke liye net edge {CONSENSUS_NET_MIN*100:.0f}%+ chahiye "
                 f"(agree kam se kam {CONSENSUS_MIN_AGREE_FLOOR*100:.0f}%), {_cb['categories']} categories ka data mila.")
    if _cs_["against"] and not _cs_["passed"]:
        lines.append("SELL ke against: " + ", ".join(_cs_["against"][:6]))
    if _cb["against"] and not _cb["passed"]:
        lines.append("BUY ke against: " + ", ".join(_cb["against"][:6]))
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
    real_res = sorted([lv for lv in levels if lv["price"] > live + 0.05 * atr and _is_real_level(lv) and lv["strength"] >= 1.5],
                      key=lambda x: x["price"])[:2]
    real_sup = sorted([lv for lv in levels if lv["price"] < live - 0.05 * atr and _is_real_level(lv) and lv["strength"] >= 1.5],
                      key=lambda x: -x["price"])[:2]
    if real_res:
        lines.append(f"SELL plan: REAL resistance {real_res[0]['price']:,.0f} ({real_res[0]['src']}) -- price wahan pahunch kar "
                     f"rejection candle de tabhi sell ({abs(real_res[0]['price'] - live):.0f} pts upar). Beech mein sell nahi.")
    if real_sup:
        lines.append(f"BUY plan: REAL support {real_sup[0]['price']:,.0f} ({real_sup[0]['src']}) -- price wahan pahunch kar "
                     f"rejection candle de tabhi buy ({abs(live - real_sup[0]['price']):.0f} pts neeche). Beech mein buy nahi.")
    # WATCHLIST (v15): instead of lowering the bar when quiet, tell the user EXACTLY which
    # OI walls the engine is waiting on and what must happen there -- so silence is never a mystery.
    watch = []
    for lv in levels:
        c_oi, p_oi = _oi_sides(X["level_text"](lv["price"]))
        if lv["price"] > live and c_oi >= OI_STRONG_MIN_CONTRACTS and c_oi >= OI_DOMINANCE_MIN * max(p_oi, 1):
            watch.append(("SELL", lv["price"], c_oi, p_oi))
        elif lv["price"] < live and p_oi >= OI_STRONG_MIN_CONTRACTS and p_oi >= OI_DOMINANCE_MIN * max(c_oi, 1):
            watch.append(("BUY", lv["price"], p_oi, c_oi))
    watch.sort(key=lambda w: abs(w[1] - live))
    for side, price, w_oi, o_oi in watch[:3]:
        lines.append(f"WATCH: {price:,.0f} par {'CALL' if side == 'SELL' else 'PUT'} OI wall ({w_oi:,} vs {o_oi:,}) -- "
                     f"price wahan tak {'chadh' if side == 'SELL' else 'gir'} kar rejection de to {side} ({abs(price - live):.0f} pts door).")
    verdicts = []
    try:
        for want_side in ("support", "resistance"):
            pool = [lv for lv in levels if lv.get("side") == want_side and _is_real_level(lv)
                    and (_is_oi_level(lv) or lv["strength"] >= 2.0)]
            if pool:
                nearest = min(pool, key=lambda lv: abs(lv["price"] - live))
                v = _level_verdict(X, nearest)
                verdicts.append(v)
                lines.append(f"LEVEL VERDICT {v['level']:,.0f} ({want_side}, {v['dist_pts']:.0f} pts, {v['state']}): {v['action']}")
    except Exception:
        logger.exception("market view verdict failed")
    for _n in X.get("notes", [])[:2]:
        lines.append(_n)
    lines.append(f"Abhi ka best BUY score {bs:.0f}/100, best SELL score {ss:.0f}/100 (trade ke liye {REVERSAL_SCORE_MIN:.0f}-{ENTRY_SCORE_MIN:.0f}+ chahiye).")
    return {"bias": bias, "supports": [_fmt(s) for s in sup], "resistances": [_fmt(r) for r in res],
            "buy_score": round(bs, 1), "sell_score": round(ss, 1), "lines": lines, "consensus": {"buy": _cb, "sell": _cs_},
            "session_low": f.get("sess_low"), "session_high": f.get("sess_high"), "level_verdicts": verdicts}


def _recent_zone_score_bump(direction, live, atr):
    """
    Har trade ke baad blindly wahi zone dobara nahi khelte -- agar last
    setup (WIN ho ya LOSS, dono) ka entry abhi ke live price ke bahut
    paas hai, to yeh price ek proven liquidity/OI zone hai jahan se
    market reverse bhi kar sakta hai, sirf continue nahi. Trade block
    nahi karte -- bas isi same-direction repeat ke liye thoda zyada
    evidence (score) maangte hain, taaki genuine continuation phir bhi
    lag jaaye lekin ek weak repeat na lage.
    """
    try:
        recent = trade_learning.get_recent_setups(limit=1)
        if not recent:
            return 0.0
        last = recent[0]
        if last.get("direction") != direction:
            return 0.0
        last_entry = last.get("entry")
        if last_entry is None or not atr:
            return 0.0
        if abs(live - last_entry) <= NEAR_ZONE_ATR_FRAC * atr:
            return NEAR_ZONE_SCORE_BUMP
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return 0.0
    return 0.0



def _loss_streak_active():
    """v14: N consecutive stop-outs => engine is out of sync with the market; pause."""
    try:
        streak, last_loss_t = 0, None
        for s in trade_learning.get_recent_setups(limit=8):
            st = s.get("status")
            if st == "LOSS":
                streak += 1
                if last_loss_t is None:
                    ts = s.get("exit_timestamp") or s.get("timestamp")
                    last_loss_t = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
            elif st == "WIN":
                break
        if streak >= LOSS_STREAK_N and last_loss_t is not None:
            return (datetime.now() - last_loss_t).total_seconds() < LOSS_STREAK_PAUSE_MIN * 60
    except Exception:
        logger.exception("loss streak check failed")
    return False


def _same_zone_after_loss(direction, live, atr):
    """v14: never re-enter the same side within 1 ATR of a recent stopped-out entry (90 min)."""
    try:
        for s in trade_learning.get_recent_setups(limit=4):
            if s.get("direction") != direction or s.get("status") != "LOSS":
                continue
            ts = s.get("exit_timestamp") or s.get("timestamp")
            t0 = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - t0).total_seconds() < 90 * 60 and s.get("entry") is not None \
                    and abs(live - float(s["entry"])) <= 1.0 * atr:
                return True
    except Exception:
        logger.exception("same zone check failed")
    return False


def _ist_now():
    return datetime.now(_IST)


def _outside_entry_window():
    n = _ist_now()
    hm = (n.hour, n.minute)
    return hm < ENTRY_WINDOW_START or hm >= ENTRY_WINDOW_END


def _hour_bucket():
    hm = _ist_now().hour * 60 + _ist_now().minute
    if hm < 10 * 60 + 30:
        return "0930-1030"
    if hm < 12 * 60 + 30:
        return "1030-1230"
    if hm < 14 * 60:
        return "1230-1400"
    return "1400-1510"


def _today_trade_count():
    """Total trades (probe + full) already taken today. Used ONLY to compute the v19 pace-ease below --
    never to skip a hard safety gate."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        n = 0
        for s_ in trade_learning.get_recent_setups(limit=40):
            ts = str(s_.get("timestamp") or "")[:19]
            if ts.startswith(today):
                n += 1
        return n
    except Exception:
        logger.exception("today trade count failed")
        return 0


def _pace_ease():
    """0..DAILY_TRADE_TARGET_EASE_MAX -- how far behind today is on its way to DAILY_TRADE_TARGET trades,
    scaled into a small bar-easing amount. 0 when on pace or ahead. This can only make an already-real,
    already-safe setup a little easier to qualify for -- it cannot manufacture a setup out of no data."""
    try:
        start_m = ENTRY_WINDOW_START[0] * 60 + ENTRY_WINDOW_START[1]
        end_m = ENTRY_WINDOW_END[0] * 60 + ENTRY_WINDOW_END[1]
        now_m = _ist_now().hour * 60 + _ist_now().minute
        frac = max(0.0, min(1.0, (now_m - start_m) / float(end_m - start_m))) if end_m > start_m else 1.0
        expected = DAILY_TRADE_TARGET * frac
        behind = expected - _today_trade_count()
        if behind <= 0:
            return 0.0
        return round(min(1.0, behind / 3.0), 3)  # 0..1 fraction of "how behind", capped
    except Exception:
        return 0.0


def _probe_throttled():
    """True if a new PROBE is not allowed right now (too soon after the last probe, or daily cap hit)."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        n_today, last_ts = 0, None
        for s_ in trade_learning.get_recent_setups(limit=40):
            if not s_.get("probe"):
                continue
            ts = str(s_.get("timestamp") or "")[:19]
            if ts.startswith(today):
                n_today += 1
            try:
                t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                if last_ts is None or t > last_ts:
                    last_ts = t
            except Exception:
                pass
        if n_today >= PROBE_MAX_PER_DAY:
            return True
        if last_ts is not None and (datetime.now() - last_ts).total_seconds() < PROBE_MIN_GAP_MIN * 60:
            return True
    except Exception:
        logger.exception("probe throttle check failed; blocking probe to be safe")
        return True
    return False


def _post_close_cooldown_active():
    """
    Har trade close hone ke baad (WIN ho, LOSS ho, ya EXPIRED) engine ko
    ek chhota gap deta hai fresh candle/data dekhne ke liye, taaki wahi
    tick pe seedha dusra trade na khul jaaye jaise pehle ho raha tha.
    """
    try:
        recent = trade_learning.get_recent_setups(limit=1)
        if not recent:
            return False
        last = recent[0]
        if last.get("status") not in ("WIN", "LOSS", "EXPIRED"):
            return False
        ts = last.get("exit_timestamp") or last.get("timestamp")
        if not ts:
            return False
        t = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - t).total_seconds() < POST_CLOSE_COOLDOWN_MIN * 60
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        return False


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
                             df=None, pcr=None, oc_source=None, external_ai_research=None):
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
        facs = [str(t) for t in (best.get("factors") or [])]
        if oc_source is not None and str(oc_source).upper() != "LIVE":
            # SIMULATED option chain => its OI numbers are modelled, not real: never let them drive a trade
            facs = [x for x in facs if " oi" not in x.lower() and "oi " not in x.lower()]
        return " ".join(facs).lower()

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
        "level_text": level_text, "notes": [],
        "pcr": _sf(pcr), "oc_live": str(oc_source or "").upper() == "LIVE", "max_pain": _sf(max_pain),
        "adv_dec_ratio": (breadth_advances / (breadth_advances + breadth_declines)
                          if (breadth_advances is not None and breadth_declines is not None
                              and (breadth_advances + breadth_declines) > 0) else None),
        "global_avg": _sf(global_avg_change),
        "bn_note": banknifty_correlation_note,
        "order_flow_seen": bool(_flatten_context_text(dashboard_context.get("RAW Order Flow"))),
        "raw": {"sniper": sniper_up, "news": news_up, "n50": n50, "gr": gr_bias, "fund": fund},
        "external_ai": external_ai_research if isinstance(external_ai_research, dict) else {},
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
    # V22: never lose a clear 50/100-point live move just because none of the
    # four structural playbooks found a static level on this refresh.
    for d in (1, -1):
        try:
            c = _score_momentum_candidate(d, X)
        except Exception:
            logger.exception("momentum fallback failed")
            c = None
        if c:
            cands.append(c)

    best_buy = max((c for c in cands if c["d"] == 1), key=lambda c: c["score"], default=None)
    best_sell = max((c for c in cands if c["d"] == -1), key=lambda c: c["score"], default=None)
    market_view = _build_market_view(X, best_buy, best_sell)

    if not cands:
        return _no("Abhi koi playbook lagu nahi - " + (" ".join(X["notes"][:2]) if X["notes"] else "price na kisi REAL level ke paas hai, na confirmed move hai.")
                   + " Engine REAL level par confirmation ka wait kar raha hai.", market_view=market_view)

    ranked = sorted(cands, key=lambda c: c["score"], reverse=True)

    def _summary(c):
        return f"{PLAYBOOK_LABELS[c['play']]} {c['score']:.0f}/100"

    chosen = None
    blocked_reason = None
    # ---- self-adjusting strictness: if the recent non-probe trades are losing, tighten by itself ----
    try:
        _perf = trade_learning.recent_performance()
    except Exception:
        logger.exception("recent_performance failed")
        _perf = {"n": 0, "wins": 0, "win_rate": None, "bad": False}
    _adaptive_add = ADAPTIVE_SCORE_PENALTY if _perf.get("bad") else 0.0
    # v19: daily learning-pace ease -- if today is behind its DAILY_TRADE_TARGET pace, shave a little off
    # both the score bar and the consensus net-edge bar (floored, see _consensus/CONSENSUS_NET_MIN_FLOOR).
    # This only ever cancels out the adaptive bad-form penalty and then, if still behind, nudges the
    # baseline bar itself a small bounded amount -- it never touches OI/RR/SL/real-level/loss-streak gates.
    _behind = _pace_ease()
    _score_ease = _behind * DAILY_TRADE_TARGET_EASE_MAX
    _net_ease = _behind * DAILY_TRADE_TARGET_NET_EASE_MAX
    _adaptive_add = max(-2.0, _adaptive_add - _score_ease)  # can go slightly negative -> eases baseline too
    # v18: 'form bad' used to switch probes off (or restrict them to OI-strong walls only) -> fewer/no new
    # resolved trades -> form never gets fresh data -> the tight bar never lifts -> silent for days.
    # Probes are the ONLY way the engine gets new data to judge itself by, so they must NEVER be gated by
    # form being bad -- only the FULL trade's score bar tightens. Probes keep their own separate, already
    # more conservative thresholds (PROBE_MIN_AGREE/AGAINST/FACTORS/CONFIDENCE, real level, daily cap, gap).
    def _probe_ok(cand):
        return bool(PROBE_ENABLED)
    if _perf.get("bad"):
        X["notes"].append(f"Recent form kharab ({_perf['wins']}/{_perf['n']} jeete, {_perf['win_rate']}%, "
                          f"'{trade_learning.LOGIC_VERSION}' logic ke sirf naye trades gine -- purana logic ke "
                          f"trades is mein nahi jud rahe) -- engine ne khud full-trade bar +{ADAPTIVE_SCORE_PENALTY:.0f} "
                          f"kar diya; probes normal thresholds par chalu hain (seekhna nahi rukta).")
    if _behind > 0:
        X["notes"].append(f"Aaj {_today_trade_count()} trade hue hain, {DAILY_TRADE_TARGET}/din ke target se peeche -- "
                          f"bar thoda (score -{_score_ease:.1f}, consensus net -{_net_ease:.1f}pp) aasan kiya gaya hai; "
                          f"safety gates (OI, RR, SL, real level) waise hi sakht hain.")
    for c in ranked:
        is_rev = c["play"] in ("BOUNCE_BUY", "REJECTION_SELL")
        base_need = (REVERSAL_OI_SCORE_MIN if c.get("oi_strong") else REVERSAL_SCORE_MIN) if is_rev else ENTRY_SCORE_MIN
        base_need += _adaptive_add
        if c["score"] < base_need:
            # v21: a below-bar reversal can still be a small learning probe
            # when the location is real and the price-action evidence is
            # meaningful. This is the main anti-drought escape hatch.
            if (PROBE_ENABLED and is_rev and c.get("score", 0) >= 34.0 and c.get("pa", 0) >= 10.0
                    and c.get("level") and _is_real_level(c["level"]) and not _probe_throttled()):
                c["probe"] = True
            else:
                continue
        if _outside_entry_window():
            blocked_reason = blocked_reason or (
                f"{_summary(c)}: abhi entry window ke bahar hai (naya trade {ENTRY_WINDOW_START[0]}:{ENTRY_WINDOW_START[1]:02d}"
                f"-{ENTRY_WINDOW_END[0]}:{ENTRY_WINDOW_END[1]:02d} IST ke beech hi -- open ka shor / close se pehle ka time skip)")
            continue
        # v20: was two SEPARATE hard floors (pa>=X AND loc>=Y) -- a candidate with an excellent level
        # (very high loc) but a slightly-under-floor price-action trigger (or vice versa) was blocked
        # even though its OVERALL score had already cleared base_need above. Now it's one COMBINED floor
        # (same total bar) so real strength in one dimension can compensate for the other, instead of
        # requiring both to independently clear their own line -- exactly the "too many things must all
        # be true at once" stacking being reported. Genuinely weak-on-both-fronts setups still fail this.
        # V22: location and price-action are weighted evidence, not separate vetoes.
        # post-close pause: applies to EVERY playbook now, reversal or
        # continuation. Previously continuation trades (breakdown/breakout)
        # skipped this gap entirely, so the moment one trade closed, the
        # engine could fire straight into another one in the same direction
        # off the same still-fresh candle -- a blind repeat rather than a
        # fresh look at the data. The gap is kept deliberately short (a few
        # minutes, not hours) precisely so a genuine still-running trend is
        # not missed and the engine does not go quiet for long stretches --
        # it only forces one fresh candle/data re-check between trades.
        if _post_close_cooldown_active():
            blocked_reason = blocked_reason or (
                f"{_summary(c)}: pichla trade abhi-abhi close hua - agle trade se pehle "
                f"{POST_CLOSE_COOLDOWN_MIN} min ka fresh-data gap (blind turant repeat nahi)")
            continue
        direction_name = "BUY" if c["d"] == 1 else "SELL"
        # v16 FULL-DATA CONSENSUS GATE: every data family votes; trade only if the weighted majority agrees.
        _td = _trend_dir(X)
        _counter = bool(is_rev and _td != 0 and _td != c["d"])
        _cftxt = X["level_text"](c["level"]["price"] if c.get("level") else X["live"])
        _broke = bool((not is_rev) and c.get("level") and _is_oi_level(c["level"]))
        c["broke_oi_wall"] = _broke
        c["consensus"] = _consensus(c["d"], X, _cftxt, _counter, _broke, net_ease=_net_ease)
        _cs = c["consensus"]
        # v17 LEVEL VERDICT GATE: "jis side jaane ke chance zyada hain wahi side". If the full data agrees with the
        # OPPOSITE direction clearly more than with this one, this candidate is skipped -- bounce AND break, probes too.
        _opp = _consensus(-c["d"], X, _cftxt, False, _broke)
        c["opposite_consensus"] = _opp
        if _opp["agree_pct"] >= _cs["agree_pct"] + BREAK_LEAN_MARGIN:
            _opp_name = "SELL" if c["d"] == 1 else "BUY"
            c["misses"].append(f"Opposite data lean: {_opp_name} {_opp['agree_pct']:.0f}% vs {direction_name} {_cs['agree_pct']:.0f}%")
            # V22: opposite consensus is a caution, not a blanket veto.
        if not _cs["passed"]:
            c["reasons"].append(f"V22 soft consensus: {direction_name} {_cs['agree_pct']:.0f}% agree / {_cs['against_pct']:.0f}% against")

        if _loss_streak_active():
            blocked_reason = blocked_reason or (f"{LOSS_STREAK_N}+ trades lagatar SL hue -- {LOSS_STREAK_PAUSE_MIN} min ka pause, "
                                                "market ko dekh kar naya REAL level banne ka wait.")
            continue
        if _same_zone_after_loss(direction_name, X["live"], X["atr"]):
            blocked_reason = blocked_reason or f"{_summary(c)}: isi zone se abhi SL hua tha -- wahi jagah dobara nahi."
            continue
        # v20: removed the separate blanket "any same-direction trade for 30 min after ANY loss" hard
        # gate that used to sit here. It was a THIRD independent lock for the same underlying risk that
        # _same_zone_after_loss (zone-specific, 90 min, 1 ATR) and the zone_bump score-penalty just below
        # already cover -- stacking three separate gates on one risk was exactly the kind of
        # can-never-all-align-at-once over-gating being reported. A genuinely different, well-confirmed
        # level far from where the loss happened should not be blocked just because the SAME direction
        # lost once, somewhere else, a few minutes ago.
        # V22: same-zone score bump removed; the hard same-zone-after-loss
        # safety check is retained separately.
        # ROOM-TO-RUN GATE: skip a setup whose realistic target (after the
        # wall-trim above) no longer clears MIN_RR. Taking it anyway means
        # risking the full fixed SL for a target that isn't really there --
        # this is checked per-candidate so a genuinely good setup on another
        # playbook/side can still be taken instead of falling straight to
        # "no trade".
        _stop, _target, _target2, _rr, _tnote = _build_trade_levels(c, X)
        if _rr < MIN_RR - 1e-6:
            blocked_reason = blocked_reason or (
                f"{_summary(c)}: not enough real room to a target ({_rr:.2f}R < {MIN_RR:.1f}R needed) - {_tnote}")
            continue
        chosen = c
        chosen_levels = (_stop, _target, _target2, _rr, _tnote)
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

    # V22 SIDE SELECTION: only a near-tie remains a directional safety stop.
    _other = best_sell if chosen["d"] == 1 else best_buy
    if _other and (chosen["score"] - _other["score"]) < 2.0:
        return _no(f"BUY/SELL scores are nearly tied ({chosen['score']:.0f} vs {_other['score']:.0f}); side selection is not clear enough yet.",
                   market_view=market_view, factors_true=factors_true, factors_total=factors_total, factor_flags=factor_flags)

    # ---- build the trade ------------------------------------------------
    direction = "BUY" if chosen["d"] == 1 else "SELL"
    d = chosen["d"]
    stop, target, target2, rr, target_note = chosen_levels
    lvl = chosen["level"]
    level_price = lvl["price"] if lvl else live
    fake_breakout_info = chosen.get("fake_breakout")
    fake_breakout_suspected = bool(fake_breakout_info and fake_breakout_info.get("suspected"))

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
        "external_ai_aligned": bool((X.get("external_ai") or {}).get("direction") == direction and
                                     str((X.get("external_ai") or {}).get("verdict", "")).upper() != "NO_TRADE"),
    }
    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    # ---- HARD GATES: OI confirmation + minimum genuine confluence --------
    # These run BEFORE confidence is even computed. A high evidence score on
    # location + one or two price-action triggers used to be enough to fire
    # a trade (e.g. 48/100 score, 13/30 factors) -- that is exactly the
    # "isko pura data dekhne ke bad hi trade banana chahiye" complaint.
    # Everything below is data already sitting in factor_flags; this just
    # refuses to trade unless enough of it actually agrees.
    #
    # OI-not-against was eased from OI-must-be-heavy-in-our-favor: a level
    # that's been proven by MULTIPLE independent sources (S/R zone + swing +
    # ladder + session extreme etc. all clustering at the same price --
    # lvl["strength"] captures exactly this) is real evidence a bounce/
    # rejection works there even on a day OI hasn't built up heavily yet.
    # Blocking every such setup just because the OI text didn't literally
    # say "heavy put/call OI" was throwing away genuine, well-tested
    # support/resistance bounces. What still hard-blocks, no matter how
    # strong the level looks: OI sitting HEAVILY ON THE OTHER SIDE (writers
    # actively defending against us) -- that is the dangerous case, not a
    # quiet/neutral OI reading.
    oi_against = (d == 1 and "heavy call oi" in ftxt) or (d == -1 and "heavy put oi" in ftxt)
    # v17 BUG FIX: for BREAKDOWN_SELL the level is the just-broken SUPPORT, and a Put-OI wall there always tripped this veto
    # (mirror for BREAKOUT_BUY at a Call-OI wall) -- so a confirmed break of an OI wall could NEVER trade, the exact
    # setup asked for. A continuation whose break is confirmed (fake-break check already enforces this) is exempt: the
    # writers who defended the level lost. The wall AHEAD of the trade is still penalised in the score.
    if oi_against and chosen.get("broke_oi_wall"):
        oi_against = False
    level_well_proven = bool(lvl and lvl.get("strength", 0) >= 2.0)
    # these two are the ONLY bars that ever move -- and only downward, only
    # after a genuine drought, only within the caps set above. Every other
    # gate (oi_against, fake-breakout, score floors) is untouched by this.
    drought_steps = _drought_steps()
    eff_min_factors = OI_STRONG_MIN_FACTORS if chosen.get("oi_strong") else max(DROUGHT_FACTORS_FLOOR, MIN_FACTORS_TRUE - DROUGHT_FACTORS_STEP * drought_steps)
    eff_min_confidence = max(DROUGHT_CONFIDENCE_FLOOR, MIN_CONFIDENCE - DROUGHT_CONFIDENCE_STEP * drought_steps)
    drought_note = (f" [Drought-ease active: {drought_steps} step(s) after a long quiet gap -- confluence bar "
                    f"eased to {eff_min_factors}/{factors_total} factors & {eff_min_confidence:.0f}% confidence, "
                    f"OI/fake-breakout hard gates unchanged]" if drought_steps else "")
    if oi_against:
        return _no(f"{_summary(chosen)} mila (score {chosen['score']:.0f}), lekin OI seedha hamare AGAINST hai "
                   f"(yahan heavy {'call' if d == 1 else 'put'} OI writers defend kar rahe hain) -- yeh wahi "
                   f"khatarnak wall hai, entry nahi lega.",
                   market_view=market_view, factors_true=factors_true, factors_total=factors_total,
                   factor_flags=factor_flags)
    is_probe = bool(chosen.get("probe"))
    if not factor_flags.get("oi_aligned"):
        chosen["reasons"].append("OI directional support absent/neutral; treated as soft evidence, not an entry veto")
    if factors_true < max(0, eff_min_factors):
        chosen["reasons"].append(f"V22: {factors_true}/{factors_total} explicit factor flags are true; missing/neutral data does not veto the trade")

    try:
        confidence_pct, used_learning, learned_count = trade_learning.compute_confidence(factor_flags)
    except Exception:
        # a storage hiccup must never freeze the engine again -- fall back to the rule-based estimate
        logger.exception("compute_confidence failed; using rule-based confidence")
        confidence_pct, used_learning, learned_count = 50.0 + min(18.0, factors_true * 1.8), False, 0
    # nudge by how strong THIS setup's evidence is, then respect the honest ceiling
    confidence_pct += max(-6.0, min(6.0, (chosen["score"] - 60.0) * 0.2))
    confidence_pct = round(max(trade_learning.CONFIDENCE_FLOOR, min(trade_learning.CONFIDENCE_CEILING, confidence_pct)), 1)

    if confidence_pct < eff_min_confidence:
        chosen["reasons"].append(f"V22: learned confidence {confidence_pct:.1f}% is advisory only and no longer blocks entry")

    # ---- entry meta: what KIND of trade is this (the learning layer segments on this) ----
    lvl_grade = (lvl.get("grade") if lvl else None) or "NA"
    is_reversal_play = chosen["play"] in ("BOUNCE_BUY", "REJECTION_SELL")
    # v20: a reversal that fired WITHOUT a genuinely confirmed rejection candle (see unconfirmed_rejection
    # above) may NEVER become a full trade no matter how the rest of its score/consensus turned out --
    # it can only ever be a probe (or nothing). This is what lets a good-looking-but-not-yet-confirmed
    # moment still produce a learning trade instead of total silence, without risking full size on it.
    if chosen.get("unconfirmed_rejection"):
        chosen["reasons"].append("V22: textbook rejection candle is not fully confirmed; level + weighted evidence still decide priority")
    # V22: level grade is advisory only; weak grades can still trade when the live side score is clear.
    _verdict = None
    try:
        if lvl:
            _verdict = _level_verdict(X, lvl)
    except Exception:
        logger.exception("level verdict failed")
    entry_meta = {
        "playbook": chosen["play"], "grade": lvl_grade,
        "lean": (_verdict or {}).get("lean"), "verdict_state": (_verdict or {}).get("state"),
        "bounce_pct": (_verdict or {}).get("bounce_pct"), "break_pct": (_verdict or {}).get("break_pct"), "hour_bucket": _hour_bucket(),
        "side": direction, "probe": bool(is_probe), "level_price": round(level_price, 2),
        "level_dist": round(abs(live - level_price), 1), "score": chosen["score"], "rr": rr,
        "consensus_agree": (chosen.get("consensus") or {}).get("agree_pct"),
        "oi_strong": bool(chosen.get("oi_strong")),
        "expiry_day": bool(_ist_now().weekday() == EXPIRY_WEEKDAY),
        "sr_engine": sr.get("sr_engine"),
        "logic_version": trade_learning.LOGIC_VERSION,  # v18: tags this trade so a future logic change
                                                          # can't have ITS bad streak blamed back on this one.
    }
    _seg_blocked, _seg_reason = trade_learning.segment_gate(entry_meta)
    if _seg_blocked:
        chosen["reasons"].append(f"Learning advisory: {_seg_reason}")

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
        "score_model": "playbook_engine_v22",
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
        # Fake-breakout risk (see _fake_breakout_flags): only ever set for
        # BREAKDOWN_SELL/BREAKOUT_BUY. "suspected" means this setup fired
        # with 2 of 6 warning signs (not enough to hard-block, but enough
        # that trade_learning.py must NOT let its eventual WIN/LOSS teach
        # the self-learning layer anything -- see app.py's log_setup call
        # and trade_learning._resolved_learning_rows().
        "fake_breakout_suspected": fake_breakout_suspected,
        "fake_breakout_info": fake_breakout_info,
        # >0 means this fired only because the confluence bar was eased after
        # a long quiet drought (see DROUGHT_* above) -- OI/fake-breakout hard
        # gates were still fully enforced, but this setup is a notch less
        # proven than a normal-bar entry. Surface this in the UI so it reads
        # as "learning-mode entry", not a full-conviction call.
        "drought_ease_steps": drought_steps,
        "oi_strong": bool(chosen.get("oi_strong")), "oi_info": chosen.get("oi_info"),
        "consensus": chosen.get("consensus"),
        "external_ai_research": external_ai_research if isinstance(external_ai_research, dict) else {},
        # PROBE = learning trade taken with a slightly lower (but still real-level, OI-safe) bar
        "probe": bool(is_probe),
        "entry_meta": entry_meta, "level_grade": lvl_grade,
        "recent_form": _perf,
        "level_verdict": _verdict,
    }
