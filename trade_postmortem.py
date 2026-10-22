"""
trade_postmortem.py -- v48: har band trade ka "post-mortem" (kyon jeeta / kyon hara) aur usse roz seekhna.  Purely additive.

For every CLOSED trade (WIN / LOSS / EXPIRED) it builds a review from data the tool already stored plus the candles after the exit:
  * HOW the trade moved      best move (MFE) and worst move (MAE) in R, minutes held, SL size vs ATR
  * WHAT happened after exit for a LOSS: did price still reach the target within 90 min (stop hunted / timing), keep going against
                              us (wrong direction), or just chop (whipsaw)?
  * WHICH entry factors were right / wrong  global, news, Nifty-50 news, fundamentals, strategy / structure, flow, entry quality ...
  * a plain-Hinglish story of the trade.
Across trades it keeps a factor / factor-group scoreboard and a failure-mode table (Wilson bounds, "data kam" when n is small),
plus plain-language suggestions.

SAFETY  It never opens, blocks, resizes or edits a trade, never touches SL / target rules, and nothing here changes the engine
        probability.  Its only outputs are (1) dashboard tables, (2) a short advisory note for Gemini's review prompt, (3) one
        stored row per day.  `ENABLED = False` switches the whole module off.
HONESTY The reasons are what the stored data shows.  What news or big players actually did is not in the data; the story says so.
"""
from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)

ENABLED = True
LOOKAHEAD_MIN = 90          # minutes after the exit that are inspected to judge a LOSS
MIN_GROUP_N = 8             # a factor / group needs this many trades on EACH side before it gets a verdict
MIN_MODE_N = 5              # failure-mode suggestions need at least this many losses
GEMINI_NOTE_MAX = 440

GROUPS = {
    "Global market": ["global_aligned", "global_research_aligned"],
    "News (India + Nifty50)": ["news_sentiment_aligned", "nifty50_news_aligned", "external_ai_aligned"],
    "Nifty50 fundamentals": ["nifty50_fundamentals_aligned"],
    "Strategy / structure": ["main_signal_aligned", "sniper_setup_aligned", "mtf_stack_aligned", "regime_aligned",
                             "htf_1h_aligned", "htf_15min_aligned", "vwap_aligned", "ml_agrees", "level_pct_ge_65"],
    "Option chain / flow": ["oi_aligned", "order_flow_aligned", "pcr_velocity_aligned", "breadth_aligned", "banknifty_no_divergence"],
    "Entry quality / timing": ["entry_not_extended", "entry_confirmed_bounce", "entry_timing_safe", "no_opposite_transition",
                               "immediate_reversal_risk_low", "data_quality_pass"],
}
FACTOR_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}

MODE_TEXT = {
    "STOP_HUNT_THEN_WORKED": "SL chhua, phir 90 min me target bhi chhu liya (disha sahi thi, SL/timing ne maara)",
    "WRONG_DIRECTION": "SL ke baad bhi price ulti disha me chalta raha (disha hi galat thi)",
    "NEVER_WORKED": "Entry ke turant baad hi ulta gaya, kabhi 0.25R bhi favour me nahi gaya",
    "WORKED_THEN_FAILED": "Target ke kareeb (60%+) pahunch kar palta aur SL laga",
    "WHIPSAW": "Choppy market: SL laga, phir na target aaya na wapas bada move",
    "LOSS_PATH_PENDING": "SL laga; baad ka price-path abhi poora nahi (90 min baad pata chalega)",
    "CLEAN_WIN": "Saaf jeet: SL ke paas gaya hi nahi",
    "SURVIVED_SCARE": "Jeet, par SL ke bahut kareeb jaakar bachi (kismat ka hissa)",
    "NORMAL_WIN": "Jeet",
    "EXPIRED_STALLED": "Time-stop: trade chala hi nahi",
    "EXPIRED_DRIFT": "Time/EOD par band: chhota drift",
    "EXPIRED_OTHER": "Time/EOD par band",
}


def wilson(wins, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def _f(x, default=None):
    try:
        v = float(x)
        return v if v == v else default
    except Exception:
        return default


def _ts(x):
    try:
        t = pd.Timestamp(str(x)[:19])
        return None if pd.isna(t) else t
    except Exception:
        return None


def _naive_index(df):
    idx = df.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    except Exception:
        pass
    return idx


def _candles_after(df, t_from, minutes):
    """[(high, low)] of candles that OPENED after t_from within `minutes`, plus whether the window is fully in the data."""
    try:
        if df is None or len(df) < 2 or "High" not in df.columns or "Low" not in df.columns or t_from is None:
            return [], False
        idx = _naive_index(df)
        t_to = t_from + pd.Timedelta(minutes=minutes)
        out = []
        for t, hi, lo in list(zip(idx, df["High"], df["Low"]))[:-1]:        # last row is the still-forming candle
            if t_from < t <= t_to and t.date() == t_from.date():
                out.append((float(hi), float(lo)))
        complete = bool(len(idx) and idx[-1] >= t_to) or t_to.time() >= pd.Timestamp("15:30").time()
        # a window that runs past the close is complete once the session is over (a later day exists in the frame)
        if not complete:
            try:
                complete = bool(idx[-1].date() > t_from.date())
            except Exception:
                pass
        return out, complete
    except Exception:
        logger.exception("candles_after failed (ignored)")
        return [], False


def review_trade(t, df=None):
    """One closed trade -> post-mortem dict.  `t` comes from trade_learning.get_resolved_full()."""
    fx = t.get("factors") or {}
    meta = fx.get("_learning_meta") if isinstance(fx.get("_learning_meta"), dict) else {}
    feats = fx.get("_features") if isinstance(fx.get("_features"), dict) else {}
    entry_meta = fx.get("_entry_meta") if isinstance(fx.get("_entry_meta"), dict) else {}
    side = str(t.get("direction") or "").upper()
    is_buy = side == "BUY"
    entry, sl, tp = _f(t.get("entry")), _f(t.get("stop_loss")), _f(t.get("target"))
    status = str(t.get("status") or "")
    if entry is None or sl is None or tp is None or side not in ("BUY", "SELL"):
        return None
    risk = max(abs(entry - sl), 1e-9)
    reward = abs(tp - entry)
    mfe = _f(meta.get("max_favorable_pts"), 0.0) or 0.0
    mae = _f(meta.get("max_adverse_pts"), 0.0) or 0.0
    t0, t1 = _ts(t.get("timestamp")), _ts(t.get("exit_timestamp"))
    held = round((t1 - t0).total_seconds() / 60.0, 1) if t0 is not None and t1 is not None else None
    atr = _f(feats.get("atr"))
    flags = {k: v for k, v in fx.items() if not str(k).startswith("_") and isinstance(v, bool)}

    mode, after = None, {}
    if status == "LOSS":
        bars, complete = _candles_after(df, t1, LOOKAHEAD_MIN)
        hit_tp = any((hi >= tp) if is_buy else (lo <= tp) for hi, lo in bars)
        further = max([((sl - lo) if is_buy else (hi - sl)) for hi, lo in bars] or [0.0])
        after = {"bars": len(bars), "complete": bool(complete), "reached_target_after": bool(hit_tp), "extra_adverse_pts": round(further, 1)}
        if hit_tp:
            mode = "STOP_HUNT_THEN_WORKED"
        elif mfe >= 0.6 * reward and reward > 0:
            mode = "WORKED_THEN_FAILED"
        elif mfe < 0.25 * risk:
            mode = "NEVER_WORKED"
        elif not complete:
            mode = "LOSS_PATH_PENDING"
        elif further >= 1.0 * risk:
            mode = "WRONG_DIRECTION"
        else:
            mode = "WHIPSAW"
    elif status == "WIN":
        mode = "CLEAN_WIN" if mae < 0.35 * risk else "SURVIVED_SCARE" if mae >= 0.7 * risk else "NORMAL_WIN"
    else:
        er = _f(meta.get("expired_r"))
        mode = "EXPIRED_STALLED" if (mfe < 0.35 * risk) else "EXPIRED_DRIFT" if er is not None and abs(er) < 0.5 else "EXPIRED_OTHER"

    won = status == "WIN" or (status == "EXPIRED" and (_f(meta.get("expired_r"), 0.0) or 0.0) >= 0.5)
    lost = status == "LOSS" or (status == "EXPIRED" and (_f(meta.get("expired_r"), 0.0) or 0.0) <= -0.5)
    # which stored factors pointed the right / wrong way for THIS trade
    misled = sorted(k for k, v in flags.items() if v and lost)          # said "yes" but the trade lost
    warned = sorted(k for k, v in flags.items() if (not v) and lost)    # said "no / not aligned" and the trade did lose (warning was right)
    confirmed = sorted(k for k, v in flags.items() if v and won)
    missed = sorted(k for k, v in flags.items() if (not v) and won)     # said "not aligned" but the trade won (warning was wrong)

    pm = {"key": f"{str(t.get('timestamp'))[:19]}|{side}", "day": str(t.get("timestamp") or "")[:10], "time": str(t.get("timestamp") or "")[11:16],
          "side": side, "status": status, "mode": mode, "entry": entry, "sl": sl, "target": tp, "exit": _f(t.get("exit_price")),
          "risk_pts": round(risk, 1), "reward_pts": round(reward, 1), "mfe_r": round(mfe / risk, 2), "mae_r": round(mae / risk, 2),
          "held_min": held, "sl_atr": round(risk / atr, 2) if atr else None, "after": after,
          "playbook": str(fx.get("_playbook_label") or entry_meta.get("playbook") or "-"), "grade": entry_meta.get("grade"),
          "hour": _f(feats.get("hour")), "vix": _f(feats.get("vix")), "flags": flags,
          "misled": misled, "warned": warned, "confirmed": confirmed, "missed": missed,
          "final": bool(mode != "LOSS_PATH_PENDING")}
    pm["story"] = story(pm)
    return pm


def story(pm):
    """Plain Hinglish explanation (what the stored data shows)."""
    try:
        bits = [f"{pm['side']} {pm['entry']:,.0f}, SL {pm['risk_pts']:.0f} pt, target {pm['reward_pts']:.0f} pt"
                + (f", {pm['held_min']:.0f} min chala" if pm.get("held_min") is not None else "") + "."]
        bits.append(MODE_TEXT.get(pm["mode"], pm["mode"]) + f" (sabse accha {pm['mfe_r']:.2f}R, sabse bura {pm['mae_r']:.2f}R).")
        if pm.get("sl_atr") is not None and pm["status"] == "LOSS":
            bits.append(f"SL = {pm['sl_atr']:.1f}x ATR" + (" (ATR ke hisaab se tang)" if pm["sl_atr"] < 0.8 else "") + ".")

        def grp(names):
            g = {}
            for n in names:
                g.setdefault(FACTOR_GROUP.get(n, "Anya"), []).append(n)
            return [k for k in g if k != "Anya"]
        if pm["status"] == "LOSS":
            if pm["warned"]:
                w = grp(pm["warned"])
                if w:
                    bits.append("Pehle se chetawani thi (aligned nahi the): " + ", ".join(w[:4]) + ".")
            if pm["misled"]:
                m = grp(pm["misled"])
                if m:
                    bits.append("Ye 'haan' bolkar bhi galat nikle: " + ", ".join(m[:4]) + ".")
            bits.append("News / bade players ka asli asar data me nahi hota, isliye wo yahan confirm nahi hota.")
        elif pm["status"] == "WIN" and pm["confirmed"]:
            c = grp(pm["confirmed"])
            if c:
                bits.append("Jinhone saath diya: " + ", ".join(c[:4]) + ".")
        return " ".join(bits)
    except Exception:
        return ""


def build(df=None, limit=120):
    """Review every closed trade.  Returns a list newest first.  Never raises."""
    if not ENABLED:
        return []
    try:
        import trade_learning as tl
        rows = tl.get_resolved_full(limit=limit)
    except Exception:
        logger.exception("postmortem: trades unavailable (ignored)")
        return []
    out = []
    for t in rows:
        try:
            pm = review_trade(t, df)
            if pm:
                out.append(pm)
        except Exception:
            logger.exception("postmortem: one trade failed (ignored)")
    return out


# ----------------------------------------------------------------------------- aggregation

def decided(pms):
    """Only WIN / LOSS are used for factor statistics (EXPIRED is mixed)."""
    return [p for p in pms if p["status"] in ("WIN", "LOSS")]


def factor_board(pms):
    """-> (factor_rows, group_rows).  For every factor: win rate when aligned vs not aligned, with Wilson bounds."""
    rows = decided(pms)
    keys = sorted({k for p in rows for k in p["flags"]})
    out = []
    for k in keys:
        a = [p for p in rows if p["flags"].get(k) is True]
        b = [p for p in rows if p["flags"].get(k) is False]
        aw, bw = sum(p["status"] == "WIN" for p in a), sum(p["status"] == "WIN" for p in b)
        out.append(_row(k, FACTOR_GROUP.get(k, "-"), len(a), aw, len(b), bw))
    return out, _group_rows(rows)


def _row(name, group, na, aw, nb, bw):
    wa = aw / na if na else None
    wb = bw / nb if nb else None
    lo_a, hi_a = wilson(aw, na)
    lo_b, hi_b = wilson(bw, nb)
    verdict = "data kam"
    if na >= MIN_GROUP_N and nb >= MIN_GROUP_N:
        if lo_a > hi_b:
            verdict = "bharosemand (aligned hone par sach me behtar)"
        elif hi_a < lo_b:
            verdict = "bhatkane wala (aligned hone par bhi zyada haara)"
        else:
            verdict = "farq saabit nahi"
    return {"factor": name, "group": group, "aligned_n": na, "aligned_wr": wa, "not_n": nb, "not_wr": wb, "verdict": verdict}


def _group_rows(rows):
    out = []
    for g, fs in GROUPS.items():
        # a trade counts as "group aligned" when most of the group's recorded flags say yes
        a, b = [], []
        for p in rows:
            vals = [p["flags"][f] for f in fs if f in p["flags"]]
            if not vals:
                continue
            (a if sum(vals) * 2 >= len(vals) else b).append(p)
        aw, bw = sum(p["status"] == "WIN" for p in a), sum(p["status"] == "WIN" for p in b)
        out.append(_row(g, "-", len(a), aw, len(b), bw))
    return out


def mode_table(pms):
    losses = [p for p in pms if p["status"] == "LOSS"]
    cnt = {}
    for p in losses:
        cnt[p["mode"]] = cnt.get(p["mode"], 0) + 1
    n = len(losses)
    return [{"mode": k, "text": MODE_TEXT.get(k, k), "n": v, "share": v / n} for k, v in sorted(cnt.items(), key=lambda z: -z[1])] if n else []


def suggestions(pms):
    """Plain-language, never auto-applied."""
    out = []
    try:
        losses = [p for p in pms if p["status"] == "LOSS" and p.get("final")]
        n = len(losses)
        share = {m["mode"]: m["share"] for m in mode_table(pms)}
        if n >= MIN_MODE_N:
            if share.get("STOP_HUNT_THEN_WORKED", 0) >= 0.35:
                out.append(f"{n} losses me se {100 * share['STOP_HUNT_THEN_WORKED']:.0f}% me SL ke baad target bhi chhua: disha sahi, SL/timing tang ho sakta hai. "
                           "Sirf jaankari -- SL rules aap khud tay karein (apne aap nahi badlenge).")
            if share.get("WRONG_DIRECTION", 0) + share.get("NEVER_WORKED", 0) >= 0.5:
                out.append("Zyada losses turant ulte gaye ya SL ke baad bhi ulte chalte rahe: entry confirmation aur chase-filters par dhyan do.")
            if share.get("WHIPSAW", 0) >= 0.35:
                out.append("Losses me choppy/whipsaw zyada: range-bound ghanton me trade kam lena behtar ho sakta hai.")
        fr, gr = factor_board(pms)
        for g in gr:
            if g["verdict"].startswith("bhatkane"):
                out.append(f"'{g['group']}' aligned hone par bhi {100 * (g['aligned_wr'] or 0):.0f}% jeete ({g['aligned_n']} trades) vs na-aligned {100 * (g['not_wr'] or 0):.0f}% -- ye signal abhi bhatka raha hai.")
            elif g["verdict"].startswith("bharosemand"):
                out.append(f"'{g['group']}' sach me kaam kar raha hai: aligned {100 * (g['aligned_wr'] or 0):.0f}% vs na-aligned {100 * (g['not_wr'] or 0):.0f}% jeet.")
        for f in fr:
            if f["verdict"].startswith("bhatkane"):
                out.append(f"Factor '{f['factor']}' bhatka raha hai ({100 * (f['aligned_wr'] or 0):.0f}% vs {100 * (f['not_wr'] or 0):.0f}%).")
    except Exception:
        logger.exception("postmortem suggestions failed (ignored)")
    return out[:10]


def gemini_note(pms):
    """<= GEMINI_NOTE_MAX chars of advisory text about the last losses and the factor verdicts."""
    try:
        done = [p for p in pms if p.get("final")]
        if not done:
            return ""
        lines = []
        last_losses = [p for p in done if p["status"] == "LOSS"][:3]
        if last_losses:
            lines.append("- Recent losses (why): " + "; ".join(f"{p['side']} {p['time']} {p['mode'].lower().replace('_', ' ')}"
                                                                 + (f" [{', '.join(grp_names(p['misled'])[:2])} misled]" if grp_names(p["misled"]) else "") for p in last_losses) + ".")
        _, gr = factor_board(pms)
        bad = [g["group"] for g in gr if g["verdict"].startswith("bhatkane")]
        good = [g["group"] for g in gr if g["verdict"].startswith("bharosemand")]
        if good or bad:
            lines.append("- Signal groups proven on own trades: " + ("reliable: " + ", ".join(good) + ". " if good else "") + ("misleading: " + ", ".join(bad) + "." if bad else ""))
        return "\n".join(lines)[:GEMINI_NOTE_MAX]
    except Exception:
        logger.exception("postmortem gemini_note failed (ignored)")
        return ""


def grp_names(names):
    seen = []
    for n in names:
        g = FACTOR_GROUP.get(n)
        if g and g not in seen:
            seen.append(g)
    return seen


def daily_summary(pms, day):
    todays = [p for p in pms if p["day"] == day]
    w = sum(p["status"] == "WIN" for p in todays)
    l = sum(p["status"] == "LOSS" for p in todays)
    modes = {}
    for p in todays:
        modes[p["mode"]] = modes.get(p["mode"], 0) + 1
    return {"day": day, "n": len(todays), "w": w, "l": l, "modes": modes, "stories": [f"{p['time']} {p['status']}: {p['story']}" for p in todays][:12]}


def save_day_reviews(pms, day):
    """One stored row per day ('pm-YYYY-MM-DD'); only finished reviews.  Guarded."""
    try:
        import market_memory as mm
        todays = [p for p in pms if p["day"] == day and p.get("final")]
        if todays:
            mm.save_day("pm-" + day, daily_summary(todays, day))
    except Exception:
        logger.exception("save_day_reviews failed (ignored)")


# ----------------------------------------------------------------------------- v48b: warning-count loss filter (earned, bounded)
WARN_GATE = "auto"          # "off" = never touch the engine; "auto" = only when the trade history PROVES it (see warn_stats)
WARN_PENALTY = 0.10         # max logit penalty (~2.5 pct points).  Penalty only -- this filter never ADDS probability.
WARN_MIN_N = 10             # trades needed in BOTH the "many warnings" and "few warnings" buckets
_WARN_STATE = {"stats": None, "ts": 0.0}


def warn_count(flags):
    """How many signal GROUPS (of those recorded) were mostly NOT aligned with the trade at entry."""
    n = tot = 0
    for fs in GROUPS.values():
        vals = [flags[f] for f in fs if f in flags and isinstance(flags[f], bool)]
        if not vals:
            continue
        tot += 1
        if sum(vals) * 2 < len(vals):
            n += 1
    return n, tot


def warn_stats(pms):
    """Does 'many groups warned against the trade' really lose more on THIS user's own closed trades?"""
    rows = decided(pms)
    hi, lo = [], []
    for p in rows:
        n, tot = warn_count(p["flags"])
        if tot < 3:
            continue
        (hi if n * 2 >= tot else lo).append(p)       # half or more of the groups warned
    hw, lw = sum(p["status"] == "WIN" for p in hi), sum(p["status"] == "WIN" for p in lo)
    out = {"hi_n": len(hi), "hi_w": hw, "lo_n": len(lo), "lo_w": lw,
           "hi_wr": hw / len(hi) if hi else None, "lo_wr": lw / len(lo) if lo else None, "earned": False}
    if len(hi) >= WARN_MIN_N and len(lo) >= WARN_MIN_N:
        out["earned"] = bool(wilson(hw, len(hi))[1] < wilson(lw, len(lo))[0])
    return out


def update_state(stats):
    import time
    _WARN_STATE.update({"stats": stats, "ts": time.time()})


def entry_adjustment(flags):
    """-> {'penalty_logit': >= 0, 'note': str}.  Inert (0.0) unless the history proved that trades taken with many warning groups lose."""
    import time
    out = {"penalty_logit": 0.0, "note": ""}
    try:
        st = _WARN_STATE.get("stats")
        if not ENABLED or WARN_GATE != "auto" or not st or not st.get("earned") or time.time() - float(_WARN_STATE.get("ts") or 0) > 900:
            return out
        n, tot = warn_count({k: v for k, v in (flags or {}).items() if not str(k).startswith("_")})
        if tot >= 3 and n * 2 >= tot:
            out["penalty_logit"] = WARN_PENALTY
            out["note"] = (f"post-mortem: {n}/{tot} signal groups against this trade; such trades won {100 * st['hi_wr']:.0f}% "
                           f"({st['hi_n']}) vs {100 * st['lo_wr']:.0f}% ({st['lo_n']}) otherwise")
    except Exception:
        logger.exception("warn entry_adjustment failed (ignored)")
        return {"penalty_logit": 0.0, "note": ""}
    return out


# ----------------------------------------------------------------------------- v48b: missed-move review ("kahan trade lena chahiye tha")
def missed_review(days, window_min=20):
    """days = {day: autopsy summary}.  For every BIG (50+ pt) move the engine did not trade, the gate that was answering during the
    first minutes of that move (from gate_log).  -> (rows, by_gate)."""
    rows, by_gate = [], {}
    try:
        import gate_log
        for day, s in sorted((days or {}).items())[-10:]:
            if not isinstance(s, dict) or "legs" not in s:
                continue
            gl = None
            for lg in s["legs"]:
                if not lg.get("major") or lg.get("engine"):
                    continue
                if gl is None:
                    gl = gate_log.rows_for_day(day) or []
                t0 = pd.Timestamp(lg["start_ts"])
                near = [r for r in gl if r.get("candle") and t0 <= pd.Timestamp(r["candle"]) <= t0 + pd.Timedelta(minutes=window_min)]
                cnt = {}
                for r in near:
                    cnt[r["gate"]] = cnt.get(r["gate"], 0) + 1
                if cnt:
                    g = max(cnt, key=cnt.get)
                    reason = next((r["reason"] for r in near if r["gate"] == g and r.get("reason")), "")
                else:
                    g, reason = "engine log nahi mila", ""
                by_gate[g] = by_gate.get(g, 0) + 1
                rows.append({"day": day, "start": lg["start"], "dir": lg["dir"], "pts": lg["pts"], "gate": g, "reason": str(reason)[:140]})
    except Exception:
        logger.exception("missed_review failed (ignored)")
    return rows[-20:], by_gate
