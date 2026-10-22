"""
market_adaptive.py -- makes the WHOLE tool improve day by day, safely (v47, additive).

Four jobs, all bounded, all fail-safe, all with a kill switch (ENABLED / ENGINE_ADJUST):

1. STRATEGY SCORECARD  Market Brain records which named strategies were live on every 5-min candle.  Once the 60-min outcome is known
   each strategy "episode" (a signal that appears and stays) is scored: did price touch +25 in its direction BEFORE -25 against it,
   compared with what a random candle does?  The result feeds strategy_engine's existing small ranking bonus/penalty (max +/-8 pts,
   in practice far less because of shrinkage).  (The old hook `trade_learning.get_strategy_performance` did not exist, so strategies
   never learned from live outcomes at all -- this replaces that dead hook.)
2. GEMINI MEMORY       Gemini's final review prompt gets a short "what this system learned" block (its own recent track record, patterns,
   forecaster status, weak/strong strategies, main reversal reasons).  Advisory only -- it can never override the live packet.
3. ENGINE ADJUSTMENT   A tiny probability nudge (max +/-0.15 logit ~ +/-3.5 pct points) from the forecaster and from statistically
   solid patterns.  It switches ITSELF on only when the evidence has been earned (see engine_adjustment) and off again if it stops.
4. DAILY UPGRADE LOG   One metrics row per day (data learned, forecaster quality, live accuracy, engine win rate ...) so you can SEE
   whether the system is really getting better, plus plain-language suggestions (never auto-applied).

Nothing here edits a stored trade, a table the old code uses, or a hard gate.
"""
from __future__ import annotations

import logging
import math
import time

import pandas as pd

import market_memory as mm

logger = logging.getLogger(__name__)

ENABLED = True                 # master kill switch
ENGINE_ADJUST = "auto"         # "off" = never touch the engine probability;  "auto" = only when earned
FC_MIN_LIVE_N = 30             # live, validated forecasts with a known outcome needed before the forecaster may nudge the engine
FC_MIN_ACC = 0.55              # ... and they must have been right at least this often
FC_CAP = 0.12
PAT_MIN_EFF_N = 12             # a pattern needs this many INDEPENDENT-equivalent samples (~100 raw candles) ...
PAT_OVERLAP = 8.0              # ... neighbouring candles share most of their 60-min look-ahead, so n raw candles ~ n/8 independent ones
PAT_Z = 2.3                    # ... and its edge must hold at ~99% confidence
PAT_BONUS = 0.05
PAT_PENALTY = 0.06
TOTAL_CAP = 0.15
STRAT_MIN_EPISODES = 10
STRAT_SHRINK_K = 20            # prior strength: 20 imaginary "average" outcomes pull small samples back to neutral
EPISODE_GAP_MIN = 15           # a strategy still live within 15 min of its last sighting is the SAME episode
GEMINI_TEXT_MAX = 1500

_STATE: dict = {"fc": None, "pstats": {}, "pbase": {}, "active": [], "score": {}, "ts": 0.0}
_strat_cache: dict = {"key": None, "val": {}}
_last_metrics_save = {"t": 0.0}


def wilson(wins, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


# ----------------------------------------------------------------------------- 1. strategy scorecard
def _compute_strategy_stats(snaps):
    lab = [s for s in snaps if s.get("y") and s["y"].get("f25") in ("UP", "DN")]
    up_d = (sum(1 for s in lab if s["y"]["f25"] == "UP") / len(lab)) if len(lab) >= 60 else 0.5
    acc: dict = {}
    last_seen: dict = {}
    for s in snaps:
        if s.get("src") != "live":
            continue
        strats = (s.get("f") or {}).get("strats") or []
        if not isinstance(strats, list):
            continue
        t = pd.Timestamp(s["ts"])
        for item in strats:
            key, _, d = str(item).partition(":")
            if not key or d not in ("BUY", "SELL"):
                continue
            prev = last_seen.get((key, d))
            last_seen[(key, d)] = t
            if prev is not None and (t - prev).total_seconds() <= EPISODE_GAP_MIN * 60:
                continue                              # same episode, already counted at its first candle
            y = s.get("y")
            if not y or y.get("f25") not in ("UP", "DN"):
                continue
            win = (d == "BUY" and y["f25"] == "UP") or (d == "SELL" and y["f25"] == "DN")
            e = acc.setdefault(key, {"n": 0, "wins": 0, "exp": 0.0})
            e["n"] += 1
            e["wins"] += 1 if win else 0
            e["exp"] += up_d if d == "BUY" else (1 - up_d)
    out = {}
    for key, e in acc.items():
        if e["n"] < STRAT_MIN_EPISODES:
            continue
        excess = (e["wins"] - e["exp"]) / (e["n"] + STRAT_SHRINK_K)         # shrunk edge over a random candle
        out[key] = {"samples": e["n"], "wins": e["wins"], "raw_win_rate": round(100 * e["wins"] / e["n"], 1),
                    "expected_win_rate": round(100 * e["exp"] / e["n"], 1), "win_rate": round(50 + 100 * excess, 1)}
    return out


def strategy_stats(snaps=None):
    """{strategy_key: {'samples','win_rate'(50 = no better than a random candle),...}} -- shape strategy_engine already consumes."""
    try:
        if not ENABLED:
            return {}
        snaps = snaps if snaps is not None else mm.load_snapshots()
        key = (len(snaps), snaps[-1]["ts"] if snaps else "", sum(1 for s in snaps if s.get("y")))
        if _strat_cache["key"] == key:
            return _strat_cache["val"]
        val = _compute_strategy_stats(snaps)
        _strat_cache.update({"key": key, "val": val})
        return val
    except Exception:
        logger.exception("strategy_stats failed (ignored)")
        return {}


# ----------------------------------------------------------------------------- 3. engine adjustment (earned, bounded)
def update_state(fc, pstats, pbase, active, score):
    _STATE.update({"fc": fc, "pstats": {p["key"]: p for p in (pstats or [])}, "pbase": pbase or {}, "active": list(active or []),
                   "score": score or {}, "ts": time.time()})


def adjustment_status():
    """Plain-language: is each nudge currently earned?"""
    fc, sc = _STATE.get("fc") or {}, _STATE.get("score") or {}
    fc_ok = bool(fc.get("ready") and fc.get("trusted") and sc.get("n_trusted", 0) >= FC_MIN_LIVE_N and (sc.get("acc_trusted") or 0) >= FC_MIN_ACC)
    return {"engine_adjust": ENGINE_ADJUST if ENABLED else "off", "forecaster_earned": fc_ok,
            "fc_live_n": sc.get("n_trusted", 0), "fc_live_acc": sc.get("acc_trusted"),
            "patterns_with_enough_samples": sum(1 for p in (_STATE.get("pstats") or {}).values() if p["n"] / PAT_OVERLAP >= PAT_MIN_EFF_N)}


def engine_adjustment(direction):
    """-> {'delta_logit': +supports / -against the proposed trade, 'active': bool, 'note': str}.  Inert (0.0) unless earned."""
    out = {"delta_logit": 0.0, "active": False, "note": ""}
    try:
        if not ENABLED or ENGINE_ADJUST != "auto":
            return out
        d = str(direction).upper()
        if d not in ("BUY", "SELL") or time.time() - float(_STATE.get("ts") or 0) > 900:
            return out                                      # no fresh brain state -> do nothing
        fc, sc = _STATE.get("fc") or {}, _STATE.get("score") or {}
        total, notes = 0.0, []
        if fc.get("ready") and fc.get("trusted") and sc.get("n_trusted", 0) >= FC_MIN_LIVE_N and (sc.get("acc_trusted") or 0) >= FC_MIN_ACC:
            p = fc["p_up25"] if d == "BUY" else 1.0 - fc["p_up25"]
            adj = max(-FC_CAP, min(FC_CAP, (p - 0.5) * 0.8))
            total += adj
            notes.append(f"forecaster {p * 100:.0f}% ({adj:+.2f})")
        base = _STATE.get("pbase") or {}
        bd = base.get("up") if d == "BUY" else base.get("dn")
        if bd is not None:
            padj = 0.0
            for k in _STATE.get("active") or []:
                st = (_STATE.get("pstats") or {}).get(k)
                if not st or st["n"] / PAT_OVERLAP < PAT_MIN_EFF_N:
                    continue
                n_eff = st["n"] / PAT_OVERLAP
                rate = st["up"] if d == "BUY" else st["dn"]
                lo, hi = wilson(rate * n_eff, n_eff, PAT_Z)
                if lo > bd + 0.03:
                    padj += PAT_BONUS
                    notes.append(f"pattern '{st['text'][:40]}' supports ({rate * 100:.0f}% vs {bd * 100:.0f}% base, n={st['n']})")
                elif hi < bd - 0.03:
                    padj -= PAT_PENALTY
                    notes.append(f"pattern '{st['text'][:40]}' against ({rate * 100:.0f}% vs {bd * 100:.0f}% base, n={st['n']})")
            total += max(-0.10, min(0.10, padj))
        total = max(-TOTAL_CAP, min(TOTAL_CAP, total))
        out.update({"delta_logit": round(total, 3), "active": bool(notes), "note": "; ".join(notes)})
    except Exception:
        logger.exception("engine_adjustment failed (ignored)")
        return {"delta_logit": 0.0, "active": False, "note": ""}
    return out


# ----------------------------------------------------------------------------- 2. Gemini memory + scorecard
def _is_gemini_approved(t):
    g = t.get("gemini") or {}
    return isinstance(g, dict) and (g.get("reviewer") == "GEMINI" or g.get("status") == "APPROVED")


def gemini_scorecard(trades, shadow_by_gate=None):
    appr = [t for t in (trades or []) if _is_gemini_approved(t) and t.get("status") in ("WIN", "LOSS")]
    w = sum(1 for t in appr if t.get("status") == "WIN")
    rej = {}
    for gate, e in (shadow_by_gate or {}).items():
        if "gemini" in str(gate).lower() and e.get("n"):
            rej[gate] = e
    rn = sum(e["n"] for e in rej.values())
    rw = sum(e["wins"] for e in rej.values())
    return {"approved_n": len(appr), "approved_wins": w, "approved_wr": (w / len(appr)) if appr else None,
            "rejected_n": rn, "rejected_would_win": rw, "rejected_wr": (rw / rn) if rn else None}


def gemini_context(res, trades, gem_sc, strat):
    """Short English memory block for Gemini's review prompt (advisory)."""
    lines = []
    try:
        done = [t for t in (trades or []) if t.get("status") in ("WIN", "LOSS")][:10]
        if done:
            lines.append(f"- Engine's last {len(done)} resolved trades: {sum(1 for t in done if t['status'] == 'WIN')} wins / "
                         f"{sum(1 for t in done if t['status'] == 'LOSS')} losses. Most recent: "
                         + ", ".join(f"{t.get('direction')} {t.get('status')}" for t in done[:5]) + ".")
        if gem_sc.get("approved_n"):
            lines.append(f"- Your past approvals: {gem_sc['approved_wins']}/{gem_sc['approved_n']} won ({100 * gem_sc['approved_wr']:.0f}%)."
                         + (f" Setups you rejected would have won {gem_sc['rejected_would_win']}/{gem_sc['rejected_n']} times." if gem_sc.get("rejected_n") else "")
                         + " Be strict about chasing and extended entries if approvals are losing.")
        pst = {p["key"]: p for p in (res.get("pstats") or [])}
        act = (res.get("fc") or {}).get("patterns") or []
        for k in act[:3]:
            p = pst.get(k)
            if p:
                lines.append(f"- Active pattern now: {p['text']} -> within 60 min price touched +25 first {100 * p['up']:.0f}% / -25 first {100 * p['dn']:.0f}% (n={p['n']}).")
        fc = res.get("fc") or {}
        if fc.get("ready"):
            lines.append(f"- Statistical forecaster: lean {fc.get('lean')}, P(+25 first)={100 * fc['p_up25']:.0f}% "
                         f"({'validated' if fc.get('trusted') else 'NOT validated -- treat as weak information'}).")
        good = sorted(((k, v) for k, v in (strat or {}).items() if v["win_rate"] >= 55), key=lambda z: -z[1]["win_rate"])[:2]
        bad = sorted(((k, v) for k, v in (strat or {}).items() if v["win_rate"] <= 45), key=lambda z: z[1]["win_rate"])[:2]
        if good or bad:
            lines.append("- Strategy track record (live): " + "; ".join([f"{k} strong ({v['wins']}/{v['samples']})" for k, v in good] +
                                                                       [f"{k} weak ({v['wins']}/{v['samples']})" for k, v in bad]) + ".")
        L = res.get("lessons") or {}
        if L.get("majors"):
            lines.append(f"- Engine caught {L['caught']} of {L['majors']} big (50+ pt) moves seen so far.")
    except Exception:
        logger.exception("gemini_context failed (ignored)")
    return "\n".join(lines)[:GEMINI_TEXT_MAX]


def publish_gemini_text(text):
    """Hand the block to ai_trade_decision's review prompt (a plain module variable; read in the background review thread)."""
    try:
        import ai_trade_decision
        ai_trade_decision.LEARNED_CONTEXT_TEXT = text or ""
    except Exception:
        logger.exception("publish_gemini_text failed (ignored)")


# ----------------------------------------------------------------------------- 4. daily upgrade log + suggestions
def daily_metrics(res, trades, day):
    rep = ((res.get("fc") or {}).get("reports") or {}).get("dir25") or {}
    sc = res.get("score") or {}
    todays = [t for t in (trades or []) if str(t.get("timestamp") or "")[:10] == day and t.get("status") in ("WIN", "LOSS")]
    L = res.get("lessons") or {}
    return {"day": day, "lab_n": res.get("lab_n"), "live_n": res.get("live_n"),
            "fc_trusted": bool((res.get("fc") or {}).get("trusted")), "fc_auc": rep.get("val_auc"), "fc_acc": rep.get("val_acc"),
            "fc_base": rep.get("val_base_acc"), "live_n_scored": sc.get("n"), "live_acc": sc.get("acc"),
            "patterns": len(res.get("pstats") or []), "strategies_scored": len(strategy_stats()),
            "engine_w": sum(1 for t in todays if t["status"] == "WIN"), "engine_l": sum(1 for t in todays if t["status"] == "LOSS"),
            "majors": L.get("majors"), "caught": L.get("caught")}


def save_daily_metrics(m, force=False):
    try:
        if not m or not m.get("day"):
            return
        if not force and time.time() - _last_metrics_save["t"] < 900:
            return
        mm.save_day("stats-" + m["day"], m)
        _last_metrics_save["t"] = time.time()
    except Exception:
        logger.exception("save_daily_metrics failed (ignored)")


def suggestions(res, shadow_by_gate=None, strat=None):
    """Plain-language observations.  NEVER applied automatically."""
    out = []
    try:
        for gate, e in (shadow_by_gate or {}).items():
            if e.get("n", 0) >= 8 and e.get("win_rate") is not None:
                if e["win_rate"] >= 60:
                    out.append(f"'{gate}' gate ne {e['n']} setups roke, {e['win_rate']:.0f}% jeet jaate -- ye gate shayad zyada sakht hai.")
                elif e["win_rate"] <= 35:
                    out.append(f"'{gate}' gate sahi kaam kar raha hai: {e['n']} roke setups me sirf {e['win_rate']:.0f}% jeete.")
        for k, v in (strat or {}).items():
            if v["win_rate"] <= 42:
                out.append(f"Strategy '{k}' live me kamzor: {v['wins']}/{v['samples']} baar sahi disha. Uska score apne aap kam hota hai.")
            elif v["win_rate"] >= 58:
                out.append(f"Strategy '{k}' live me achhi: {v['wins']}/{v['samples']} baar sahi disha.")
        ps = {p["key"]: p for p in (res.get("pstats") or [])}
        base = res.get("pbase") or {}
        for key, side in (("at_day_low_after_fall", "SELL"), ("at_day_high_after_rise", "BUY")):
            p = ps.get(key)
            if p and base:
                mine = p["dn"] if side == "SELL" else p["up"]
                b = base["dn"] if side == "SELL" else base["up"]
                out.append(f"Chase check: {p['text']} ke baad {side} disha me +25 pehle sirf {100 * mine:.0f}% (normal {100 * b:.0f}%, n={p['n']}).")
        fc = res.get("fc") or {}
        if fc.get("ready") and not fc.get("trusted"):
            out.append("Forecaster abhi baseline se behtar sabit nahi hua, isliye engine ko koi nudge nahi mil raha.")
    except Exception:
        logger.exception("suggestions failed (ignored)")
    return out[:12]
