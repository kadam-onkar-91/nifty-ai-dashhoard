"""
entry_quality.py -- "is this a good PLACE and TIME to enter?"  (price-action sanity, from the 5-min candles)

Why: a strategy can be valid and the direction right, yet the entry is bad because price has ALREADY run
(buying the top of an impulse, RSI 78, outside the Bollinger band, 3 ATR above the EMA).  Gemini rejects those as "chasing".
When Gemini is not available the local fallback used to approve them -- it had no price-action sense at all.
This module gives both the engine (soft probability penalty) and the local fallback (strict block) that sense.

assess_entry(df, direction, atr=None) -> {
    "chase_flags": [str], "confirm_flags": [str], "severe": bool,
    "penalty_logit": float,        # subtract from the win-probability log-odds (0 .. MAX_PENALTY_LOGIT), bonus allowed (<=0)
    "extended": bool, "confirmed": bool, "metrics": {...}, "summary": str }
Uses only COMPLETED candles for patterns (the last row is the forming candle) plus the live price for stretch.
"""
from __future__ import annotations

import math

MAX_PENALTY_LOGIT = 0.60
MAX_BONUS_LOGIT = 0.20
FLAG_COST = 0.12
SEVERE_COST = 0.25

RSI_CHASE = 70.0          # BUY chasing at/above, SELL chasing at/below (100 - this)
RSI_SEVERE = 76.0
EMA_STRETCH_ATR = 2.0     # live price this many ATR away from EMA20 = stretched
EMA_SEVERE_ATR = 3.0
IMPULSE_ATR = 2.0         # last completed candle at least this many ATR long, in the trade direction
RUN_ATR_6 = 3.0           # price already moved this many ATR in the trade direction over the last 6 candles
STREAK_CANDLES = 4        # same-colour closed candles in a row


def _f(x, default=None):
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def assess_entry(df, direction, atr=None):
    out = {"chase_flags": [], "confirm_flags": [], "severe": False, "penalty_logit": 0.0,
           "extended": False, "confirmed": False, "metrics": {}, "summary": "no data"}
    try:
        if df is None or len(df) < 25 or direction not in ("BUY", "SELL"):
            return out
        buy = direction == "BUY"
        sign = 1.0 if buy else -1.0
        closed = df.iloc[:-1]                      # completed candles only
        last = closed.iloc[-1]
        live = _f(df["Close"].iloc[-1])
        a = _f(atr) or _f(df["ATR"].iloc[-1]) if "ATR" in df.columns else _f(atr)
        if not a or a <= 0 or live is None:
            return out
        m = out["metrics"]
        flags, confirms = [], []
        severe = False

        # 1) RSI stretch (of the last completed candle)
        rsi = _f(last.get("RSI")) if hasattr(last, "get") else None
        if rsi is not None:
            m["rsi"] = round(rsi, 1)
            ext = rsi if buy else 100.0 - rsi
            if ext >= RSI_SEVERE:
                flags.append(f"RSI {rsi:.0f} is extreme ({'overbought' if buy else 'oversold'}) - entering at the end of the move"); severe = True
            elif ext >= RSI_CHASE:
                flags.append(f"RSI {rsi:.0f} {'overbought' if buy else 'oversold'}")

        # 2) outside the Bollinger band
        bbu, bbl = _f(last.get("BB_Upper")), _f(last.get("BB_Lower"))
        if buy and bbu is not None and live > bbu:
            flags.append(f"price {live - bbu:.0f} pts above the upper Bollinger band")
        if (not buy) and bbl is not None and live < bbl:
            flags.append(f"price {bbl - live:.0f} pts below the lower Bollinger band")

        # 3) stretch from EMA20
        ema = _f(last.get("EMA_20"))
        if ema is not None:
            dist = sign * (live - ema) / a
            m["ema20_dist_atr"] = round(dist, 2)
            if dist >= EMA_SEVERE_ATR:
                flags.append(f"price is {dist:.1f} ATR away from EMA20 (very stretched)"); severe = True
            elif dist >= EMA_STRETCH_ATR:
                flags.append(f"price is {dist:.1f} ATR away from EMA20")

        # 4) impulse candle / already ran
        body = sign * (float(last["Close"]) - float(last["Open"]))
        rng = float(last["High"]) - float(last["Low"])
        m["last_candle_atr"] = round(rng / a, 2)
        if body > 0 and rng >= IMPULSE_ATR * a:
            flags.append(f"entry right after a {rng:.0f}-pt impulse candle ({rng / a:.1f} ATR)")
        if len(closed) >= 7:
            run = sign * (live - float(closed["Close"].iloc[-7])) / a
            m["run_6c_atr"] = round(run, 2)
            if run >= RUN_ATR_6:
                flags.append(f"price already moved {run:.1f} ATR in the trade direction over the last 30 min")

        # 5) streak of same-colour candles
        streak = 0
        for i in range(len(closed) - 1, max(len(closed) - 9, 0), -1):
            c = closed.iloc[i]
            if sign * (float(c["Close"]) - float(c["Open"])) > 0:
                streak += 1
            else:
                break
        m["streak"] = streak
        if streak >= STREAK_CANDLES:
            flags.append(f"{streak} same-direction candles in a row (no pullback yet)")

        # --- confirmations (what a GOOD entry looks like) ---
        recent = closed.iloc[-5:]
        prior = closed.iloc[-16:-5]
        # liquidity sweep: a recent candle took out the prior swing extreme and closed back inside
        if len(prior) >= 8:
            if buy:
                lvl = float(prior["Low"].min())
                swept = any(float(r["Low"]) < lvl and float(r["Close"]) > lvl for _, r in recent.iterrows())
            else:
                lvl = float(prior["High"].max())
                swept = any(float(r["High"]) > lvl and float(r["Close"]) < lvl for _, r in recent.iterrows())
            if swept:
                confirms.append(f"liquidity sweep of {lvl:.0f} then closed back (stop-hunt reversal)")
        # rejection wick in the trade direction on one of the last 3 closed candles
        for _, r in closed.iloc[-3:].iterrows():
            r_rng = float(r["High"]) - float(r["Low"])
            if r_rng <= 0:
                continue
            wick = (min(float(r["Open"]), float(r["Close"])) - float(r["Low"])) if buy else (float(r["High"]) - max(float(r["Open"]), float(r["Close"])))
            if wick / r_rng >= 0.45 and r_rng >= 0.6 * a:
                confirms.append("rejection wick in the trade direction"); break
        # a pullback happened and price resumed (last candle in direction after at least one opposite candle in last 4)
        opp = [sign * (float(r["Close"]) - float(r["Open"])) < 0 for _, r in closed.iloc[-4:-1].iterrows()]
        if any(opp) and body > 0 and streak < STREAK_CANDLES:
            confirms.append("pullback then resumption (not a straight-line chase)")

        out["chase_flags"], out["confirm_flags"], out["severe"] = flags, confirms, severe
        out["extended"] = bool(flags)
        out["confirmed"] = bool(confirms)
        pen = FLAG_COST * len(flags) + (SEVERE_COST if severe else 0.0)
        bonus = min(MAX_BONUS_LOGIT, 0.10 * len(confirms))
        out["penalty_logit"] = round(max(-MAX_BONUS_LOGIT, min(MAX_PENALTY_LOGIT, pen - bonus)), 3)
        out["summary"] = ("; ".join(flags) if flags else "entry not extended") + \
                         (" | confirms: " + "; ".join(confirms) if confirms else "")
    except Exception:
        out["summary"] = "entry-quality check failed (ignored)"
    return out


def fallback_blocks(eq):
    """Strict rule for the LOCAL fallback (no Gemini): block a chase.  (blocked, reason)"""
    if not eq or not eq.get("metrics") and not eq.get("chase_flags"):
        return False, ""
    flags, conf = eq.get("chase_flags") or [], eq.get("confirm_flags") or []
    if eq.get("severe") and not conf:
        return True, "entry is extremely stretched: " + flags[0]
    if len(flags) >= 2 and not conf:
        return True, "entry looks like chasing (" + "; ".join(flags[:2]) + ") with no sweep/rejection/pullback to confirm it"
    if len(flags) >= 3:
        return True, "too many chase signs even with confirmation: " + "; ".join(flags[:3])
    return False, ""
