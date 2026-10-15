"""
fast_lane.py -- v44 FAST LANE: catch the big moves the slow gates keep missing.

WHY
  The classic path needs: a named strategy + AI agreement + S/R location + confluence + a history-calibrated win chance.
  After a run of losses the calibrated win chance falls to 11-22%, so EVERY setup (even a 130-pt waterfall with 6 SELL
  strategies live) is blocked by "No positive edge".  The engine then stays flat in the biggest moves of the day.

WHAT THIS DOES
  Pure PRICE ACTION on the 5-min candles, three playbooks (both directions):
    MOMENTUM   displacement candle(s) that break the last-12-bar extreme / a key level, NOT yet extended (no chasing).
    REVERSAL   stretched move + rejection candle (wick / engulfing) AT a key level -> bounce / rejection trade.
    PULLBACK   established trend, pullback to EMA20/VWAP, bounce back in trend direction.
  Returns a candidate with entry / SL / target / score, or None.  Safety (stale data, entry window, event risk,
  cooldown, one-signal-one-trade) is applied by ai_trade_decision, not here.  No history-derived win-chance gate.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import strategy_engine

MIN_SCORE = 58.0            # fast-lane score needed to fire (0-100)
MIN_BARS = 30
MIN_ATR_PTS = 6.0           # below this the market is too dead for a 5-min fast lane
SL_MIN_PTS, SL_MAX_PTS = 14.0, 30.0
TARGET_R = 1.8              # default reward:risk of a fast-lane trade
MAX_12BAR_MOVE_ATR = 5.5    # already travelled this far in 12 bars = too late for a MOMENTUM entry (ATR lags in a sudden waterfall, so generous)
MAX_EMA_DIST_ATR = 3.6      # price this far from EMA20 = stretched; momentum entry refused (reversal may still fire)


def _f(v, d=float("nan")):
    try:
        v = float(v)
        return v if math.isfinite(v) else d
    except Exception:
        return d


def _candle(r) -> Dict[str, float]:
    o, h, l, c = _f(r["Open"]), _f(r["High"]), _f(r["Low"]), _f(r["Close"])
    rng = max(h - l, 1e-9)
    return {"o": o, "h": h, "l": l, "c": c, "rng": rng, "body": abs(c - o),
            "up_wick": h - max(o, c), "dn_wick": min(o, c) - l,
            "close_pos": (c - l) / rng}


def _key_levels(x: pd.DataFrame, sr_context: Optional[dict], prev_day: Optional[dict]) -> Dict[str, List[float]]:
    """Levels a reversal is allowed to react to: previous-day high/low, today's session high/low (before the last 3
    candles), validated S/R zones.  Returned as {'support': [...], 'resistance': [...]}."""
    sup: List[float] = []
    res: List[float] = []
    try:
        idx = pd.to_datetime(x.index)
        today = idx[-1].date()
        td = x[idx.date == today]
        hist = td.iloc[:-3] if len(td) > 6 else td.iloc[:0]
        if len(hist) >= 3:
            sup.append(float(hist["Low"].min()))
            res.append(float(hist["High"].max()))
        pd_ = x[idx.date < today]
        if len(pd_):
            last_day = pd_[pd.to_datetime(pd_.index).date == pd.to_datetime(pd_.index).date[-1]]
            sup.append(float(last_day["Low"].min()))
            res.append(float(last_day["High"].max()))
    except Exception:
        pass
    try:
        if isinstance(prev_day, dict):
            for k, tgt in (("pdl", sup), ("pdh", res)):
                if prev_day.get(k) is not None:
                    tgt.append(float(prev_day[k]))
    except Exception:
        pass
    try:
        if isinstance(sr_context, dict):
            for key, tgt in (("major_support", sup), ("nearest_support", sup), ("major_resistance", res), ("nearest_resistance", res)):
                z = sr_context.get(key)
                if isinstance(z, dict):
                    for kk in ("price", "low", "high"):
                        if z.get(kk) is not None:
                            tgt.append(float(z[kk]))
    except Exception:
        pass
    return {"support": sorted(set(round(v, 1) for v in sup if math.isfinite(v))),
            "resistance": sorted(set(round(v, 1) for v in res if math.isfinite(v)))}


def _trend_votes(r, d: int) -> int:
    """How many of (close>EMA9, EMA9>EMA20, close>VWAP, supertrend) agree with direction d."""
    n = 0
    c, e9, e20, vw, st = _f(r["Close"]), _f(r["EMA9"]), _f(r["EMA20"]), _f(r["VWAP"]), _f(r.get("SUPERTREND_DIR"), 0)
    n += 1 if (c - e9) * d > 0 else 0
    n += 1 if (e9 - e20) * d > 0 else 0
    n += 1 if (c - vw) * d > 0 else 0
    n += 1 if st * d > 0 else 0
    return n


def _momentum(x: pd.DataFrame, a: float, d: int) -> Optional[Dict[str, Any]]:
    r = x.iloc[-1]
    c = _f(r["Close"])
    cl = x["Close"].astype(float)
    move3 = (c - float(cl.iloc[-4])) * d / a
    cur, prv = _candle(x.iloc[-1]), _candle(x.iloc[-2])
    impulse_c = cur if (cur["body"] * 1.0 >= prv["body"]) else prv
    big_body = max(cur["body"], prv["body"]) / a
    close_ok = (impulse_c["close_pos"] >= 0.68) if d == 1 else (impulse_c["close_pos"] <= 0.32)
    displacement = (move3 >= 1.05) or (big_body >= 0.95 and close_ok)
    if not displacement:
        return None
    base = x.iloc[-15:-3]
    if len(base) < 8:
        return None
    level = float(base["High"].max()) if d == 1 else float(base["Low"].min())
    broke = (c > level) if d == 1 else (c < level)
    # allow a break of the last 12-bar extreme made by ANY of the last 3 candles (the break candle may be one back)
    recent_ext = float(x["High"].iloc[-3:].max()) if d == 1 else float(x["Low"].iloc[-3:].min())
    broke = broke or ((recent_ext > level) if d == 1 else (recent_ext < level))
    if not broke:
        return None
    # never enter WITH the move on a candle that is already rejecting it (the 'sell at the bottom / buy at the top' mistake)
    opp_wick = cur["dn_wick"] if d == -1 else cur["up_wick"]
    if cur["rng"] >= 0.5 * a and opp_wick >= 0.42 * cur["rng"]:
        return None
    if (d == -1 and cur["c"] > cur["o"] and cur["body"] >= 0.3 * a) or (d == 1 and cur["c"] < cur["o"] and cur["body"] >= 0.3 * a):
        return None
    if (d == -1 and cur["close_pos"] > 0.6) or (d == 1 and cur["close_pos"] < 0.4):
        return None
    votes = _trend_votes(r, d)
    if votes < 2:
        return None
    move12 = (c - float(cl.iloc[-13])) * d / a
    e20_dist = abs(c - _f(r["EMA20"])) / a
    if move12 > MAX_12BAR_MOVE_ATR or e20_dist > MAX_EMA_DIST_ATR:
        return None                                     # the move is already over: never chase
    dist_from_break = (c - level) * d / a
    if dist_from_break > 2.6:
        return None
    # a blow-off close (huge range, close right at the extreme) after a long run is exhaustion, not a fresh break
    score = 52.0 + min(14.0, (move3 - 1.0) * 10.0) + 5.0 * votes - (8.0 if move12 > 4.0 else 0.0)
    score += 6.0 if big_body >= 1.3 else 0.0
    sl_ref = float(x["Low"].iloc[-4:].min()) if d == 1 else float(x["High"].iloc[-4:].max())
    return {"playbook": "FL_MOMENTUM_BREAK", "score": score, "sl_ref": sl_ref, "level": level,
            "why": f"displacement {move3:.1f} ATR in 3 candles broke the 12-bar {'high' if d == 1 else 'low'} {level:.1f}; "
                   f"{votes}/4 trend votes; move so far {move12:.1f} ATR (not chasing)"}


def _reversal(x: pd.DataFrame, a: float, d: int, lv: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
    """d=+1 -> BUY bounce from a low; d=-1 -> SELL rejection from a high."""
    r = x.iloc[-1]
    c = _f(r["Close"])
    w = x.iloc[-12:]
    ext = float(w["Low"].min()) if d == 1 else float(w["High"].max())
    stretch = ((float(w["High"].max()) - ext) if d == 1 else (ext - float(w["Low"].min()))) / a
    if stretch < 2.0:
        return None
    # the extreme must be recent (last 5 candles) -> price is AT the turning point now
    recent = x.iloc[-5:]
    ext_recent = float(recent["Low"].min()) if d == 1 else float(recent["High"].max())
    if abs(ext_recent - ext) > 0.05 * a:
        return None
    cur, prv = _candle(x.iloc[-1]), _candle(x.iloc[-2])
    rej = None
    for cd, nm in ((cur, "current"), (prv, "previous")):
        wick = cd["dn_wick"] if d == 1 else cd["up_wick"]
        if cd["rng"] >= 0.55 * a and wick >= 0.45 * cd["rng"] and ((cd["close_pos"] >= 0.5) if d == 1 else (cd["close_pos"] <= 0.5)):
            rej = f"{'lower' if d == 1 else 'upper'}-wick rejection ({nm} candle)"
            break
    if rej is None:
        # engulfing: current candle reverses and closes beyond the previous candle's open
        if d == 1 and prv["c"] < prv["o"] and cur["c"] > cur["o"] and cur["c"] > prv["o"] and cur["body"] >= 0.5 * a:
            rej = "bullish engulfing"
        elif d == -1 and prv["c"] > prv["o"] and cur["c"] < cur["o"] and cur["c"] < prv["o"] and cur["body"] >= 0.5 * a:
            rej = "bearish engulfing"
    if rej is None:
        return None
    # not while a fresh marubozu is STILL running against the turn
    run = cur if True else prv
    against = (run["c"] < run["o"]) if d == 1 else (run["c"] > run["o"])
    if against and run["body"] >= 1.1 * a and (run["close_pos"] <= 0.2 if d == 1 else run["close_pos"] >= 0.8):
        return None
    levels = lv["support"] if d == 1 else lv["resistance"]
    near = [L for L in levels if abs(ext - L) <= 0.8 * a or (d == 1 and ext < L <= ext + 0.8 * a) or (d == -1 and ext > L >= ext - 0.8 * a)]
    at_level = bool(near)
    if not at_level and stretch < 3.0:
        return None                                     # no key level -> only a really violent stretch may reverse
    # confirmation that buyers/sellers actually took over: price back past the 3-candle midpoint
    mid = float(x["Close"].iloc[-4:-1].mean())
    reclaimed = (c > mid) if d == 1 else (c < mid)
    score = 50.0 + min(14.0, (stretch - 2.0) * 7.0) + (12.0 if at_level else 0.0) + (6.0 if reclaimed else -6.0)
    score += 4.0 if "engulfing" in rej else 0.0
    return {"playbook": "FL_REVERSAL_BOUNCE" if d == 1 else "FL_REVERSAL_REJECTION", "score": score,
            "sl_ref": ext, "level": (near[0] if near else ext),
            "why": f"{stretch:.1f} ATR stretch then {rej}{' at key level ' + format(near[0], '.1f') if at_level else ' (no key level; extreme stretch)'}"}


def _pullback(x: pd.DataFrame, a: float, d: int) -> Optional[Dict[str, Any]]:
    r = x.iloc[-1]
    if _trend_votes(r, d) < 4:
        return None
    if (_f(r["EMA20"]) - _f(r["EMA50"])) * d <= 0:
        return None
    w = x.iloc[-24:]
    leg = ((float(w["High"].max()) - float(w["Low"].min())) / a)
    if leg < 2.0:
        return None
    c = _f(r["Close"])
    e20, vw = _f(r["EMA20"]), _f(r["VWAP"])
    lows = x["Low"].iloc[-4:] if d == 1 else x["High"].iloc[-4:]
    touch_px = float(lows.min()) if d == 1 else float(lows.max())
    near_val = min(abs(touch_px - e20), abs(touch_px - vw)) <= 0.4 * a or ((touch_px <= e20) if d == 1 else (touch_px >= e20))
    if not near_val:
        return None
    cur = _candle(x.iloc[-1])
    resumed = ((c > _f(r["EMA9"])) and cur["c"] > cur["o"]) if d == 1 else ((c < _f(r["EMA9"])) and cur["c"] < cur["o"])
    if not resumed:
        return None
    retrace = ((float(w["High"].max()) - touch_px) / max(float(w["High"].max()) - float(w["Low"].min()), 1e-9)) if d == 1 else \
              ((touch_px - float(w["Low"].min())) / max(float(w["High"].max()) - float(w["Low"].min()), 1e-9))
    if retrace > 0.70:
        return None
    return {"playbook": "FL_TREND_PULLBACK", "score": 56.0 + min(10.0, (leg - 2.0) * 4.0) + (6.0 if retrace <= 0.5 else 0.0),
            "sl_ref": touch_px, "level": e20,
            "why": f"trend intact (4/4 votes), pulled back {retrace * 100:.0f}% of a {leg:.1f} ATR leg to EMA20/VWAP and resumed"}


def _rejecting(cur, a, d) -> bool:
    """True when the current candle is already rejecting a move in direction d."""
    opp_wick = cur["dn_wick"] if d == -1 else cur["up_wick"]
    if cur["rng"] >= 0.5 * a and opp_wick >= 0.42 * cur["rng"]:
        return True
    if (d == -1 and cur["c"] > cur["o"] and cur["body"] >= 0.3 * a) or (d == 1 and cur["c"] < cur["o"] and cur["body"] >= 0.3 * a):
        return True
    return (d == -1 and cur["close_pos"] > 0.6) or (d == 1 and cur["close_pos"] < 0.4)


def _flag_break(x: pd.DataFrame, a: float, d: int) -> Optional[Dict[str, Any]]:
    """Move already running, price paused in a tight flag (2-6 candles), now breaks out in the SAME direction.
    This catches the move when the first impulse candle was missed -- without chasing: the entry is at the flag edge."""
    r = x.iloc[-1]
    if _trend_votes(r, d) < 3:
        return None
    cur = _candle(r)
    if _rejecting(cur, a, d):
        return None
    c = _f(r["Close"])
    for n in (3, 4, 5, 6):
        flag = x.iloc[-1 - n:-1]
        if len(flag) < n:
            continue
        fh, fl = float(flag["High"].max()), float(flag["Low"].min())
        if (fh - fl) > 1.5 * a:
            continue
        pre = x.iloc[-1 - n - 10:-1 - n]
        if len(pre) < 6:
            continue
        leg = ((float(flag["High"].max()) - float(pre["Low"].min())) if d == 1 else (float(pre["High"].max()) - float(flag["Low"].min()))) / a
        if leg < 2.0:
            continue
        # flag must sit in the leading half of the leg (shallow pause, not a collapse)
        mid = (float(pre["Low"].min()) + fh) / 2 if d == 1 else (float(pre["High"].max()) + fl) / 2
        if (d == 1 and fl < mid) or (d == -1 and fh > mid):
            continue
        broke = (c > fh) if d == 1 else (c < fl)
        if not broke or abs(c - (fh if d == 1 else fl)) > 2.0 * a:
            continue
        if abs(c - _f(r["EMA20"])) / a > 5.0:
            continue
        return {"playbook": "FL_FLAG_BREAK", "score": 58.0 + min(12.0, (leg - 2.0) * 4.0) + (5.0 if n >= 4 else 0.0),
                "sl_ref": fl if d == 1 else fh, "level": fh if d == 1 else fl,
                "why": f"{leg:.1f} ATR leg, {n}-candle flag ({fh - fl:.0f} pts) broke {'up' if d == 1 else 'down'} through {(fh if d == 1 else fl):.1f}"}
    return None


def _level_break(x: pd.DataFrame, a: float, d: int, lv: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
    """Fresh close through previous-day high/low, session extreme or validated S/R, with a real body (not a wick poke)."""
    r = x.iloc[-1]
    cur = _candle(r)
    if _rejecting(cur, a, d) or cur["body"] < 0.5 * a or _trend_votes(r, d) < 2:
        return None
    c = _f(r["Close"])
    prior = x["Close"].astype(float).iloc[-4:-1]
    levels = lv["resistance"] if d == 1 else lv["support"]
    for L in sorted(levels, key=lambda v: abs(c - v)):
        if d == 1:
            fresh = c > L and float(prior.min()) <= L + 0.15 * a and (c - L) <= 1.4 * a
        else:
            fresh = c < L and float(prior.max()) >= L - 0.15 * a and (L - c) <= 1.4 * a
        if fresh:
            return {"playbook": "FL_LEVEL_BREAK", "score": 57.0 + min(10.0, cur["body"] / a * 6.0) + (4.0 if _vol_ok(x) else 0.0),
                    "sl_ref": (L - 0.3 * a) if d == 1 else (L + 0.3 * a), "level": L,
                    "why": f"fresh close {'above' if d == 1 else 'below'} key level {L:.1f} with a {cur['body'] / a:.1f} ATR body"}
    return None


def _vol_ok(x: pd.DataFrame) -> bool:
    try:
        v = x["Volume"].astype(float)
        return bool(v.iloc[-1] > 0 and v.iloc[-1] >= 1.1 * v.iloc[-21:-1].mean())
    except Exception:
        return False


def analyze(df: Optional[pd.DataFrame], sr_context: Optional[dict] = None, prev_day: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    """Best fast-lane candidate on the latest 5-min candle, or None."""
    try:
        if df is None or len(df) < MIN_BARS:
            return None
        x = strategy_engine._prepare(df)
        x = x.dropna(subset=["Open", "High", "Low", "Close", "ATR"]).copy()
        if len(x) < MIN_BARS:
            return None
        a = _f(x["ATR"].iloc[-1])
        if not (a >= MIN_ATR_PTS):
            return None
        lv = _key_levels(x, sr_context, prev_day)
        cands = []
        for d in (1, -1):
            for fn in (lambda: _momentum(x, a, d), lambda: _reversal(x, a, d, lv), lambda: _pullback(x, a, d),
                       lambda: _flag_break(x, a, d), lambda: _level_break(x, a, d, lv)):
                try:
                    c = fn()
                except Exception:
                    c = None
                if c:
                    c["d"] = d
                    cands.append(c)
        if not cands:
            return None
        best = max(cands, key=lambda c: c["score"])
        if best["score"] < MIN_SCORE:
            return None
        d = best["d"]
        entry = _f(x["Close"].iloc[-1])
        sl_dist = abs(entry - best["sl_ref"]) + 0.25 * a
        sl_dist = max(SL_MIN_PTS, min(SL_MAX_PTS, sl_dist, max(1.6 * a, SL_MIN_PTS)))
        sl_dist = round(sl_dist, 1)
        tgt_dist = round(TARGET_R * sl_dist, 1)
        sl = round(entry - d * sl_dist, 2)
        tp = round(entry + d * tgt_dist, 2)
        opposite = [c for c in cands if c["d"] != d and c["score"] >= MIN_SCORE - 6]
        return {"direction": "BUY" if d == 1 else "SELL", "playbook": best["playbook"], "score": round(min(best["score"], 95.0), 1),
                "underlying_entry": round(entry, 2), "stop_loss": sl, "target": tp, "risk_reward": round(tgt_dist / sl_dist, 2),
                "atr": round(a, 2), "why": best["why"], "level": best.get("level"),
                "formed_at": str(x.index[-1]), "conflicting_signal": bool(opposite),
                "all_candidates": [(c["playbook"], c["d"], round(c["score"], 1)) for c in cands]}
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------------------------------
# MISSED-MOVE AUDIT / REPLAY  -- "kaun sa 50+ pt move pakda, kaun sa chhoota, aur kyun"
# ---------------------------------------------------------------------------------------------------------------------
def swings(df: pd.DataFrame, min_pts: float = 50.0) -> List[Dict[str, Any]]:
    """Zig-zag legs of at least `min_pts` (High/Low based) in df.  Each: dir, start/end index, start/end price, pts."""
    out: List[Dict[str, Any]] = []
    if df is None or len(df) < 5:
        return out
    hi, lo = df["High"].astype(float).values, df["Low"].astype(float).values
    n = len(df)
    pivot_i, pivot_p, direction = 0, float(df["Close"].iloc[0]), 0
    ext_i, ext_p = 0, float(df["Close"].iloc[0])
    for i in range(1, n):
        if direction >= 0:
            if hi[i] > ext_p or direction == 0 and hi[i] - pivot_p >= min_pts:
                if direction == 0 and hi[i] - pivot_p >= min_pts:
                    direction = 1
                    ext_i, ext_p = i, hi[i]
                elif direction == 1 and hi[i] > ext_p:
                    ext_i, ext_p = i, hi[i]
            if direction == 1 and ext_p - lo[i] >= min_pts and i != ext_i:
                out.append({"dir": "UP", "i0": pivot_i, "i1": ext_i, "p0": pivot_p, "p1": ext_p, "pts": ext_p - pivot_p})
                pivot_i, pivot_p, direction = ext_i, ext_p, -1
                ext_i, ext_p = i, lo[i]
                continue
        if direction <= 0:
            if direction == 0 and pivot_p - lo[i] >= min_pts:
                direction = -1
                ext_i, ext_p = i, lo[i]
            elif direction == -1 and lo[i] < ext_p:
                ext_i, ext_p = i, lo[i]
            if direction == -1 and hi[i] - ext_p >= min_pts and i != ext_i:
                out.append({"dir": "DOWN", "i0": pivot_i, "i1": ext_i, "p0": pivot_p, "p1": ext_p, "pts": pivot_p - ext_p})
                pivot_i, pivot_p, direction = ext_i, ext_p, 1
                ext_i, ext_p = i, hi[i]
    if direction == 1 and ext_p - pivot_p >= min_pts:
        out.append({"dir": "UP", "i0": pivot_i, "i1": ext_i, "p0": pivot_p, "p1": ext_p, "pts": ext_p - pivot_p})
    elif direction == -1 and pivot_p - ext_p >= min_pts:
        out.append({"dir": "DOWN", "i0": pivot_i, "i1": ext_i, "p0": pivot_p, "p1": ext_p, "pts": pivot_p - ext_p})
    return out


def _resolve(df: pd.DataFrame, i: int, d: int, entry: float, sl: float, tp: float) -> Dict[str, Any]:
    """Walk forward from candle i+1: SL first if both touched in one candle (pessimistic)."""
    hi, lo = df["High"].astype(float).values, df["Low"].astype(float).values
    for j in range(i + 1, len(df)):
        hit_sl = (lo[j] <= sl) if d == 1 else (hi[j] >= sl)
        hit_tp = (hi[j] >= tp) if d == 1 else (lo[j] <= tp)
        if hit_sl:
            return {"result": "LOSS", "pts": round(-abs(entry - sl), 1), "exit_i": j}
        if hit_tp:
            return {"result": "WIN", "pts": round(abs(tp - entry), 1), "exit_i": j}
    last = float(df["Close"].iloc[-1])
    return {"result": "OPEN", "pts": round((last - entry) * d, 1), "exit_i": len(df) - 1}


def replay(df: pd.DataFrame, min_move_pts: float = 50.0, sr_context: Optional[dict] = None, prev_day: Optional[dict] = None,
           session_date=None, cooldown_bars: int = 3) -> Dict[str, Any]:
    """Re-run the fast lane candle by candle over one session and compare with the real swings.
    Pure price-action replay (no Gemini / research votes) -> an upper bound of what the fast lane SEES, not of what gets approved."""
    res: Dict[str, Any] = {"signals": [], "moves": [], "stats": {}}
    try:
        if df is None or len(df) < MIN_BARS + 5:
            return res
        idx = pd.to_datetime(df.index)
        day = session_date or idx[-1].date()
        first = int(np.argmax(idx.date == day)) if (idx.date == day).any() else 0
        start = max(first, MIN_BARS)
        sigs = []
        last_i = {1: -99, -1: -99}
        for i in range(start, len(df)):
            cand = analyze(df.iloc[: i + 1], sr_context=sr_context, prev_day=prev_day)
            if not cand:
                continue
            d = 1 if cand["direction"] == "BUY" else -1
            if i - last_i[d] <= cooldown_bars:
                continue
            last_i[d] = i
            r = _resolve(df, i, d, cand["underlying_entry"], cand["stop_loss"], cand["target"])
            sigs.append({"i": i, "time": str(df.index[i]), "direction": cand["direction"], "playbook": cand["playbook"],
                         "score": cand["score"], "entry": cand["underlying_entry"], "sl": cand["stop_loss"], "tp": cand["target"],
                         "why": cand["why"], **r})
        res["signals"] = sigs
        day_df = df.iloc[first:]
        mv = swings(day_df, min_move_pts)
        moves = []
        for m in mv:
            i0, i1 = m["i0"] + first, m["i1"] + first
            want = "BUY" if m["dir"] == "UP" else "SELL"
            hit = [s for s in sigs if s["direction"] == want and i0 <= s["i"] <= i1]
            row = {"dir": m["dir"], "from": str(df.index[i0]), "to": str(df.index[i1]), "move_pts": round(float(m["pts"]), 1)}
            if hit:
                h = hit[0]
                dcap = (m["p1"] - h["entry"]) if m["dir"] == "UP" else (h["entry"] - m["p1"])
                row.update({"caught": True, "signal_time": h["time"], "playbook": h["playbook"], "entry": h["entry"],
                            "left_after_entry_pts": round(float(dcap), 1), "trade_result": h["result"]})
            else:
                row.update({"caught": False})
            moves.append(row)
        res["moves"] = moves
        wins = sum(1 for s in sigs if s["result"] == "WIN")
        losses = sum(1 for s in sigs if s["result"] == "LOSS")
        res["stats"] = {"signals": len(sigs), "wins": wins, "losses": losses, "open": len(sigs) - wins - losses,
                        "net_pts": round(sum(s["pts"] for s in sigs), 1),
                        "moves": len(moves), "caught": sum(1 for m in moves if m["caught"]),
                        "missed": sum(1 for m in moves if not m["caught"])}
    except Exception:
        pass
    return res


# ---------------------------------------------------------------------------------------------------------------------
# PRE-PLANS -- "move aane se PEHLE pata chale"  (bounce / rejection / breakout / breakdown set-ups that are FORMING)
# ---------------------------------------------------------------------------------------------------------------------
PLAN_MAX = 3


def _plan_levels(x, a, sr_context, prev_day):
    lv = _key_levels(x, sr_context, prev_day)
    return lv


def plans(df: Optional[pd.DataFrame], sr_context: Optional[dict] = None, prev_day: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Conditional trade plans that are FORMING right now (nothing has triggered yet).
    Each plan can be shown to Gemini in advance; when its trigger fires the approval is already there (no review delay).
      BREAKOUT / BREAKDOWN : tight compression (<=1.3 ATR over 6-10 candles) sitting next to a key level / session extreme;
                             trigger = close beyond the range edge.
      BOUNCE / REJECTION   : price pressed into support / resistance (<=0.6 ATR) after a >=1.5 ATR approach;
                             trigger = the rejection candle (wick / engulfing), which the live detector then confirms.
    """
    out: List[Dict[str, Any]] = []
    try:
        if df is None or len(df) < MIN_BARS:
            return out
        x = strategy_engine._prepare(df).dropna(subset=["Open", "High", "Low", "Close", "ATR"]).copy()
        if len(x) < MIN_BARS:
            return out
        a = _f(x["ATR"].iloc[-1])
        if not (a >= MIN_ATR_PTS):
            return out
        c = _f(x["Close"].iloc[-1])
        lv = _plan_levels(x, a, sr_context, prev_day)
        stamp = str(x.index[-1])[:10]

        def mk(kind, d, level, trigger, sl_ref, why, score):
            sl_dist = max(SL_MIN_PTS, min(SL_MAX_PTS, abs(trigger - sl_ref) + 0.25 * a))
            sl_dist = round(sl_dist, 1)
            tgt = round(TARGET_R * sl_dist, 1)
            return {"id": f"{kind}:{'B' if d == 1 else 'S'}:{int(round(level / 5.0) * 5)}:{stamp}", "kind": kind,
                    "direction": "BUY" if d == 1 else "SELL", "level": round(float(level), 1), "trigger": round(float(trigger), 2),
                    "underlying_entry": round(float(trigger), 2), "stop_loss": round(trigger - d * sl_dist, 2),
                    "target": round(trigger + d * tgt, 2), "risk_reward": round(tgt / sl_dist, 2), "atr": round(a, 2),
                    "why": why, "score": round(score, 1), "created_at": str(x.index[-1])}

        # --- compression -> breakout / breakdown ---------------------------------------------------------------
        for n in (6, 8, 10):
            box = x.iloc[-n:]
            bh, bl = float(box["High"].max()), float(box["Low"].min())
            if (bh - bl) > 1.3 * a or (bh - bl) < 0.25 * a:
                continue
            if not (bl - 0.15 * a <= c <= bh + 0.15 * a):
                continue
            near_hi = [L for L in lv["resistance"] if abs(L - bh) <= 0.9 * a]
            near_lo = [L for L in lv["support"] if abs(L - bl) <= 0.9 * a]
            votes_up, votes_dn = _trend_votes(x.iloc[-1], 1), _trend_votes(x.iloc[-1], -1)
            out.append(mk("PLAN_BREAKOUT", 1, bh, bh + 0.1 * a, bl,
                          f"{n}-candle compression ({bh - bl:.0f} pts) under {bh:.1f}" + (f", key level {near_hi[0]:.1f} just above" if near_hi else ""),
                          54.0 + 5.0 * votes_up + (6.0 if near_hi else 0.0)))
            out.append(mk("PLAN_BREAKDOWN", -1, bl, bl - 0.1 * a, bh,
                          f"{n}-candle compression ({bh - bl:.0f} pts) above {bl:.1f}" + (f", key level {near_lo[0]:.1f} just below" if near_lo else ""),
                          54.0 + 5.0 * votes_dn + (6.0 if near_lo else 0.0)))
            break

        # --- approach into a key level -> bounce / rejection ----------------------------------------------------
        cl = x["Close"].astype(float)
        for d, levels, kind in ((1, lv["support"], "PLAN_BOUNCE"), (-1, lv["resistance"], "PLAN_REJECTION")):
            for L in sorted(levels, key=lambda v: abs(c - v)):
                dist = (c - L) * d                    # >0: price still on the safe side of the level
                if not (-0.3 * a <= dist <= 0.6 * a):
                    continue
                approach = (float(cl.iloc[-8]) - c) * d * -1.0 if False else ((float(cl.iloc[-8]) - c) if d == 1 else (c - float(cl.iloc[-8]))) / a
                if approach < 1.5:
                    continue
                out.append(mk(kind, d, L, c + d * 0.1 * a, L - d * 0.3 * a,
                              f"price pressed into {'support' if d == 1 else 'resistance'} {L:.1f} after a {approach:.1f} ATR approach; waiting for the rejection candle",
                              52.0 + min(10.0, (approach - 1.5) * 5.0) + 6.0))
                break
        out.sort(key=lambda p: -p["score"])
        return out[:PLAN_MAX]
    except Exception:
        return []
