"""
trade_diagnostics.py -- WHY did trades lose, and don't repeat it.

1) risk_tags(...)   : labels describing the CONDITIONS of an entry (extended entry, no confirmation, after a huge day move,
                      late session, expiry day, event day, OI flow against, high VIX ...).  Deterministic: the same function is
                      used when the trade is taken and again on old trades, so history can be grouped by tag.
2) tag_performance(): win rate of every tag over all resolved real + shadow (paper) trades.
3) tag_penalty()    : if a tag has lost often enough (enough samples, smoothed win rate low) a new candidate that carries the
                      same tag gets a small probability penalty.  This is the "it lost for THIS reason before -> be careful" rule.
4) loss_reason_summary(): table for the dashboard: which reasons appear most among LOST real trades.

Small and statistical on purpose: needs MIN_TAG_N samples before it acts, the penalty is capped, and nothing here can open a trade.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIN_TAG_N = 6.0             # weighted samples before a tag may penalise anything
BAD_WR = 0.30               # smoothed win rate below this -> penalty
VERY_BAD_WR = 0.22
TAG_PENALTY = 0.12
TAG_PENALTY_BAD = 0.20
MAX_TAG_PENALTY = 0.35

TAG_TEXT = {
    "extended_entry": "Entry was chasing an extended move",
    "no_confirmation": "No sweep / rejection / pullback confirmation at entry",
    "day_move_exhausted": "Entered after a very big move already happened today",
    "at_day_extreme": "Entered right at the day's low/high (little room left)",
    "late_session": "Entered late in the session (after 14:00)",
    "opening_15min": "Entered in the first 15 minutes (noise)",
    "midday_lull": "Entered in the midday lull (11:45-13:30)",
    "expiry_day": "Expiry day (erratic moves)",
    "event_day": "Event-risk window (RBI / Fed / Budget ...)",
    "oi_flow_against": "OI flow today was against the trade",
    "high_vix": "High VIX (wild swings)",
    "low_confidence": "Win probability was low when taken",
}


def _num(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


def risk_tags(flags, feats, hour=None):
    """flags: dict of factor booleans; feats: dict of numbers (may be None for old trades); hour: decimal hour (14.5 = 14:30)."""
    tags = []
    try:
        flags = flags or {}
        f = feats or {}
        if flags.get("entry_not_extended") is False:
            tags.append("extended_entry")
        if flags.get("entry_confirmed_bounce") is False and "entry_confirmed_bounce" in flags:
            tags.append("no_confirmation")
        dm = _num(f.get("day_move_pct"))
        if dm is not None and dm >= 1.0:
            tags.append("day_move_exhausted")
        dp = _num(f.get("day_pos"))
        if dp is not None and dp <= 0.15 and (dm is None or dm >= 0.6):
            tags.append("at_day_extreme")
        h = _num(hour if hour is not None else f.get("hour"))
        if h is not None:
            if h >= 14.0:
                tags.append("late_session")
            elif h < 9.5:
                tags.append("opening_15min")
            elif 11.75 <= h < 13.5:
                tags.append("midday_lull")
        if _num(f.get("expiry_day")) == 1.0:
            tags.append("expiry_day")
        if _num(f.get("event_caution")) == 1.0:
            tags.append("event_day")
        if _num(f.get("oi_flow")) == -1.0:
            tags.append("oi_flow_against")
        vix = _num(f.get("vix"))
        if vix is not None and vix >= 18.0:
            tags.append("high_vix")
        p = _num(f.get("win_prob"))
        if p is not None and p < 0.38:
            tags.append("low_confidence")
    except Exception:
        logger.exception("risk_tags failed (ignored)")
    return tags


def tag_performance(rows):
    """rows = trade_learning._labelled_rows_full() -> {tag: {"n": weighted n, "wins": weighted wins, "wr": smoothed win rate, "real_n", "real_losses"}}"""
    stats = {}
    for r in rows or []:
        try:
            w = float(r.get("weight", 1.0))
            win = 1.0 if r.get("label") else 0.0
            for t in risk_tags(r.get("flags"), r.get("feats")):
                s = stats.setdefault(t, {"n": 0.0, "wins": 0.0, "real_n": 0, "real_losses": 0})
                s["n"] += w
                s["wins"] += w * win
                if r.get("source") == "real":
                    s["real_n"] += 1
                    s["real_losses"] += int(not win)
        except Exception:
            continue
    for s in stats.values():
        s["wr"] = (s["wins"] + 1.0) / (s["n"] + 2.0)
    return stats


def tag_penalty(tags, rows):
    """-> (penalty_logit >= 0, note).  Only tags that have PROVEN bad with enough samples count."""
    try:
        stats = tag_performance(rows)
        pen, bad = 0.0, []
        for t in tags or []:
            s = stats.get(t)
            if not s or s["n"] < MIN_TAG_N:
                continue
            if s["wr"] < VERY_BAD_WR:
                pen += TAG_PENALTY_BAD; bad.append(f"{TAG_TEXT.get(t, t)} ({100 * s['wr']:.0f}% win over {s['n']:.0f} trades)")
            elif s["wr"] < BAD_WR:
                pen += TAG_PENALTY; bad.append(f"{TAG_TEXT.get(t, t)} ({100 * s['wr']:.0f}% win over {s['n']:.0f} trades)")
        pen = min(MAX_TAG_PENALTY, pen)
        return round(pen, 3), ("repeat-mistake check: " + "; ".join(bad)) if bad else ""
    except Exception:
        logger.exception("tag_penalty failed (ignored)")
        return 0.0, ""


def loss_reason_summary(rows):
    """Real LOST trades only -> [{"tag", "reason", "losses", "trades_with_tag", "win_rate"}] most frequent first."""
    stats = tag_performance(rows)
    out = []
    for t, s in stats.items():
        if s["real_losses"] > 0:
            out.append({"tag": t, "reason": TAG_TEXT.get(t, t), "losses": s["real_losses"], "trades_with_tag": s["real_n"],
                        "win_rate": round(100.0 * (s["real_n"] - s["real_losses"]) / max(s["real_n"], 1), 1)})
    out.sort(key=lambda z: (-z["losses"], z["win_rate"]))
    return out
