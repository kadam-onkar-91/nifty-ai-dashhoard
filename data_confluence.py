"""
data_confluence.py -- v53: turns dashboard data that was only DISPLAYED (or sent to Gemini as text) into numeric evidence for the trade engine.

Before v53 these were computed and shown but never changed a decision: sector breadth, volume POC, FII/DII option footprint, IV skew, IV percentile,
model-drift report, Monte-Carlo risk.  Now each becomes one small signed vote for / against the proposed direction.

SAFETY / HONESTY
  * Every vote is tiny and the total effect is CAPPED (-0.06 .. +0.18 logit): this layer can nudge probability, never create a trade, never touch SL/target.
  * Missing data = no vote (never a fake neutral-as-bullish).
  * Each vote is also stored as a boolean factor flag, so the existing Trade Post-Mortem factor board measures whether it REALLY helps
    on this user's own trades ("seekhne ke baad kaam me lena").  A factor that keeps misleading shows up there and can be switched off below.
  * Pure function, no I/O.  `ENABLED = False` switches the whole layer off.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

ENABLED = True
MAX_PENALTY = 0.18
MAX_BONUS = 0.06
OPPOSE_NET_FOR_PENALTY = 1.5      # weighted opposing votes minus supporting votes needed for a penalty
SUPPORT_NET_FOR_BONUS = 1.5
PENALTY_STEP = 0.10
BONUS_STEP = 0.05
DRIFT_PENALTY = 0.06              # engine's own recent record is worse than its baseline -> be pickier

# switch single votes off here if the post-mortem shows one of them misleads
VOTES_ON = {"sector": True, "poc": True, "fii": True, "iv_skew": True}
W = {"sector": 1.0, "poc": 0.8, "fii": 1.0, "iv_skew": 0.5}


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _sign(direction):
    return 1 if str(direction).upper() == "BUY" else -1


def _sector_vote(sector_breadth):
    """+1 BUY-leaning when most sectors are advancing, -1 when most are declining (None if unavailable)."""
    try:
        if sector_breadth is None or len(sector_breadth) < 4:
            return None
        ch = [_num(v) for v in sector_breadth["Avg Change (%)"]]
        ch = [v for v in ch if v is not None]
        if len(ch) < 4:
            return None
        up = sum(v > 0.10 for v in ch); dn = sum(v < -0.10 for v in ch)
        ratio = (up - dn) / len(ch)
        return 1 if ratio >= 0.40 else -1 if ratio <= -0.40 else 0
    except Exception:
        return None


def _poc_vote(live_price, live_poc, atr):
    p, c, a = _num(live_price), _num(live_poc), _num(atr)
    if p is None or c is None or not a or a <= 0:
        return None
    d = (p - c) / a
    return 1 if d >= 0.10 else -1 if d <= -0.10 else 0


def _fii_vote(fii_footprint):
    t = str(fii_footprint or "").upper()
    if not t or t in ("NEUTRAL", "N/A", "NONE"):
        return 0 if t else None
    if t.startswith("BULLISH"):
        return 1
    if t.startswith("BEARISH"):
        return -1
    return 0


def _skew_vote(iv_skew):
    """Inverted skew (upside IV rich) = squeeze / upside speculation -> BUY-leaning.  Very heavy put skew (> 4) = hedging fear -> SELL-leaning."""
    try:
        if not isinstance(iv_skew, dict):
            return None
        s = _num(iv_skew.get("skew"))
        if s is None:
            return None
        return 1 if s <= -1.5 else -1 if s >= 4.0 else 0
    except Exception:
        return None


def assess(direction, live_price=None, atr=None, sector_breadth=None, live_poc=None, fii_footprint=None, iv_skew=None,
           iv_percentile=None, drift_report=None, monte_carlo=None):
    out = {"penalty_logit": 0.0, "net": 0.0, "supports": [], "opposes": [], "flags": {}, "note": "", "available": 0}
    if not ENABLED or str(direction).upper() not in ("BUY", "SELL"):
        return out
    try:
        sg = _sign(direction)
        votes = {}
        if VOTES_ON["sector"]:
            votes["sector"] = _sector_vote(sector_breadth)
        if VOTES_ON["poc"]:
            votes["poc"] = _poc_vote(live_price, live_poc, atr)
        if VOTES_ON["fii"]:
            votes["fii"] = _fii_vote(fii_footprint)
        if VOTES_ON["iv_skew"]:
            votes["iv_skew"] = _skew_vote(iv_skew)
        label = {"sector": "sector breadth", "poc": "price vs volume POC", "fii": "FII option footprint", "iv_skew": "IV skew"}
        net = 0.0
        for k, v in votes.items():
            if v is None:
                continue
            out["available"] += 1
            if v == 0:
                continue
            aligned = (v * sg) > 0
            net += W[k] * (1 if aligned else -1)
            (out["supports"] if aligned else out["opposes"]).append(label[k])
            out["flags"][{"sector": "sector_breadth_aligned", "poc": "poc_aligned", "fii": "fii_footprint_aligned",
                          "iv_skew": "iv_skew_aligned"}[k]] = bool(aligned)
        out["net"] = round(net, 2)
        pen = 0.0
        if net <= -OPPOSE_NET_FOR_PENALTY:
            pen += PENALTY_STEP * min(2.0, -net / OPPOSE_NET_FOR_PENALTY)
        elif net >= SUPPORT_NET_FOR_BONUS:
            pen -= BONUS_STEP * min(2.0, net / SUPPORT_NET_FOR_BONUS)
        drifting = isinstance(drift_report, dict) and "DRIFT DETECTED" in str(drift_report.get("status", "")).upper()
        if drifting:
            pen += DRIFT_PENALTY
        out["flags"]["model_not_drifting"] = not drifting
        ruin = _num((monte_carlo or {}).get("risk_of_ruin_pct_of_sims")) if isinstance(monte_carlo, dict) else None
        if ruin is not None:
            out["flags"]["risk_of_ruin_low"] = ruin < 5.0
        ivp = _num((iv_percentile or {}).get("percentile")) if isinstance(iv_percentile, dict) else None
        if ivp is not None:
            out["flags"]["iv_not_extreme"] = 15.0 <= ivp <= 85.0
        out["penalty_logit"] = round(max(-MAX_BONUS, min(MAX_PENALTY, pen)), 3)
        bits = []
        if out["supports"]:
            bits.append("saath: " + ", ".join(out["supports"]))
        if out["opposes"]:
            bits.append("khilaf: " + ", ".join(out["opposes"]))
        if drifting:
            bits.append("engine ka recent record baseline se kharab (drift)")
        out["note"] = " | ".join(bits)
    except Exception:
        logger.exception("data_confluence failed (ignored)")
        return {"penalty_logit": 0.0, "net": 0.0, "supports": [], "opposes": [], "flags": {}, "note": "", "available": 0}
    return out
