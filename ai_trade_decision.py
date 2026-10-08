"""
ai_trade_decision.py — NEW, purely additive module.

Reads EVERY factor this tool computes and decides, like a disciplined
real trader would, whether there's a genuine setup worth flagging right
now -- and if so, exactly which strike, CE or PE, at what entry, with
what stop-loss and target. If nothing qualifies, it says so instead of
inventing a trade -- this is the direct fix for "it used to trade at
literally every point with no logic."

This module never claims a guarantee and never reports a confidence
above trade_learning.CONFIDENCE_CEILING -- see trade_learning.py for why.
"""
from app_logging import get_logger
logger = get_logger(__name__)

import trade_learning
import edge_model
import event_calendar
import oi_buildup
import trade_diagnostics
import strategy_engine
import gemini_pool
import json
import time
import requests
import math
import pandas as pd
from datetime import datetime, timedelta, time as dtime

# ---------------------------------------------------------------------------
# v36 quality / frequency controls (one place, so they are easy to tune)
# ---------------------------------------------------------------------------
MIN_EV_R = 0.05                 # expected value (in R) a trade must show to be taken
DROUGHT_EV_RELIEF = 0.06        # in a dry spell the bar eases a little, but never below ~break-even
DROUGHT_AFTER_MIN = 120         # "dry spell" = no setup logged for this long inside the session ...
DROUGHT_CHECK_FROM = dtime(11, 30)   # ... or none at all today by this time
BAD_FORM_EV_EXTRA = 0.0         # was 0.10: demanding MORE after losses froze the engine (no trades -> no data -> no recovery).
                                # Recovery now comes from shadow trades + the mistake memory instead of a higher wall.
REVIEW_DEADLINE_S = 150.0       # the review runs in a BACKGROUND thread (page never waits), so it may take as long as a real
                                # grounded review needs (40 s search call + 25 s plain retry per key). 45 s / 25 s / 15 s was too
                                # tight: slow-but-working Gemini calls timed out and were wrongly reported as 'unavailable'.
MAX_SL_PTS = 28.0               # widest stop the structure-aware SL may use (same ceiling as the base ATR stop)
OI_WALL_SIGNIFICANCE = 0.60     # a wall counts when its OI >= this fraction of the strongest wall in range
OI_FRONT_RUN_MIN_PTS = 4.0      # target sits this many pts BEFORE the wall (or OI_FRONT_RUN_ATR x ATR, whichever is larger)
OI_FRONT_RUN_ATR = 0.25
TARGET_REACH_ATR = 0.7         # realistic reach inside the holding window = this x ATR x sqrt(hold bars)
OI_MIN_RR = 1.2                 # wall closer than this many R -> not a usable target (soft penalty, no new gate)
MAX_TARGET_R = 2.2              # cap for wall/structure targets; the ATR fallback stays at 1.5R
MIN_WIN_PROB = 0.35             # HARD floor on the calibrated win probability.  A trade the engine itself rates at ~31% (it loses
                                # 2 of 3) can still show a hair of positive EV at 2.2R and, with the dry-spell relief, used to slip through.
                                # Shadow trades now supply learning data, so the engine no longer needs low-quality trades to learn.
ROUND_TRIP_COST_PTS = 2.0       # bid-ask spread + brokerage/STT, as underlying points per trade.  Charged against EV: a setup must beat its costs.
FLOOR_RELAX_AFTER_TRADING_DAYS = 2   # safety valve against the old "no trade for days" problem: after this many trading days
FLOOR_RELAXED_PROB = 0.31            # without a logged setup the win-chance floor eases to this (never lower)
DROUGHT_RELIEF_MIN_PROB = 0.40  # the dry-spell relief may only ease the EV bar for setups that are at least this likely to win
ENFORCE_SESSION_WINDOW = True
NO_ENTRY_BEFORE = dtime(9, 25)  # first 10 min: opening auction noise / wide spreads
NO_ENTRY_AFTER = dtime(15, 10)  # entries allowed until the market's last minutes; the tool auto-closes open trades at 15:20
                                # (trade_learning.EOD_SQUAREOFF), so nothing opened after 15:10 could do anything.  Was 14:45.


def _round_to_strike(price, step=50):
    return int(round(price / step) * step)



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
    if isinstance(raw, dict):
        pressure = str(raw.get("pressure", "")).upper()
        imbalance = raw.get("imbalance_pct")
        return pressure, imbalance
    text = _flatten_context_text(raw).upper()
    pressure = "BUYING PRESSURE" if "BUYING PRESSURE" in text else ("SELLING PRESSURE" if "SELLING PRESSURE" in text else "")
    return pressure, None


def _extract_regime(context):
    raw = context.get("RAW Regime Engine")
    text = _flatten_context_text(raw)
    if "bullish" in text and "bearish" not in text:
        return "BULLISH"
    if "bearish" in text and "bullish" not in text:
        return "BEARISH"
    return "RANGE" if "range" in text else "UNKNOWN"


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
    """
    NEW — fixes the "zero setups for days" problem. The single swing-based
    nearest level (level_prediction) only fires when price is within 1x
    ATR of that ONE specific level AND momentum already matches -- a very
    narrow window. The round-number Ladder Calculator already computed 16
    other levels (every 50pts, both directions) with their own full factor
    scoring -- this scans THOSE for the nearest one, within a slightly
    wider proximity band, that already reads in the signal's direction.
    Returns a level_prediction-shaped dict if found, else None.
    """
    if not level_ladder or not atr:
        return None
    candidates = (level_ladder.get('supports') or []) + (level_ladder.get('resistances') or [])
    max_dist = max_atr_mult * atr
    best = None
    for lvl in candidates:
        if preferred_side and str(lvl.get('approaching', '')).lower() != str(preferred_side).lower():
            continue
        if preferred_side is None:
            inferred_side = 'support' if float(lvl.get('level_price', live_price)) < float(live_price) else 'resistance'
            if inferred_side not in ('support', 'resistance'):
                continue
        if lvl.get('distance_pts', 1e9) > max_dist:
            continue
        bias_lower = lvl.get('directional_bias', '').lower()
        matches = ("bullish" in bias_lower and direction == "BUY") or \
                  ("bearish" in bias_lower and direction == "SELL")
        if not matches:
            continue
        if best is None or lvl['distance_pts'] < best['distance_pts']:
            best = lvl
    if best is None:
        return None
    approaching = 'support' if best['level_price'] < live_price else 'resistance'
    return {
        'status': 'APPROACHING_LEVEL', 'approaching': approaching,
        'level_price': best['level_price'], 'distance_pts': best['distance_pts'],
        'break_pct': best['break_pct'], 'bounce_pct': best['bounce_pct'],
        'directional_bias': best['directional_bias'], 'factors': best.get('factors', [])
    }



def _chain_rows(raw_chain):
    """Flatten whatever the dashboard hands over (DataFrame / JSON string / list / dict) into a list of row dicts."""
    rows = []

    def collect(x):
        if isinstance(x, dict):
            if any(k in x for k in ("Strike", "strike", "strike_price", "strikePrice")):
                rows.append(x)
            for v in x.values():
                collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)
    if isinstance(raw_chain, pd.DataFrame):
        data = raw_chain.to_dict("records")
    elif isinstance(raw_chain, str):
        try:
            data = json.loads(raw_chain)
        except Exception:
            data = []
    else:
        data = raw_chain
    collect(data)
    return rows


def _row_oi(r, want):
    """OI on the side we care about. want='ce' (call wall above price) or 'pe' (put wall below price).
    BUG FIX: the live chain has columns "Strike" / "Call OI" / "Put OI".  The old parser only knew lower-case
    strike keys and a single 'oi' key, so it found NO rows and the OI-wall target silently never worked."""
    keys = ("Call OI", "call_oi", "CE OI", "ce_oi") if want == "ce" else ("Put OI", "put_oi", "PE OI", "pe_oi")
    for k in keys:
        if k in r and r[k] is not None:
            try:
                return float(r[k])
            except Exception:
                pass
    side = str(r.get("option_type", r.get("type", r.get("instrument_type", "")))).lower()
    if side and want in side:                       # legacy one-row-per-option format
        for k in ("oi", "open_interest", "openInterest", "OI"):
            try:
                if k in r and r[k] is not None:
                    return float(r[k])
            except Exception:
                pass
    return 0.0


def _oi_walls(raw_chain, live_price, direction, min_pts=12.0, max_pts=140.0):
    """Opposing OI walls in the trade direction: BUY -> call-OI walls ABOVE price, SELL -> put-OI walls BELOW.
    Returns [(distance_pts, strike, oi, is_significant)] nearest first.  Significant = OI >= 60% of the strongest
    wall in range, i.e. a wall the market will actually react to (not a thin strike)."""
    want = "ce" if direction == "BUY" else "pe"
    found = {}
    for r in _chain_rows(raw_chain):
        try:
            strike = float(r.get("Strike", r.get("strike", r.get("strike_price", r.get("strikePrice")))))
        except Exception:
            continue
        oi = _row_oi(r, want)
        if oi <= 0:
            continue
        dist = (strike - live_price) if direction == "BUY" else (live_price - strike)
        if dist < min_pts or dist > max_pts:
            continue
        found[strike] = max(found.get(strike, 0.0), oi)
    if not found:
        return []
    top = max(found.values())
    out = [(round((k - live_price) if direction == "BUY" else (live_price - k), 2), k, v, v >= OI_WALL_SIGNIFICANCE * top)
           for k, v in found.items()]
    out.sort(key=lambda z: z[0])
    return out


def _oi_wall_target(raw_chain, live_price, direction, min_pts=12.0, max_pts=140.0, atr=None):
    """Target = just IN FRONT of the nearest significant opposing OI wall (price tends to stall/bounce AT the wall,
    so a target placed on it is often missed by a few points).  Returns (target, wall_strike) or None."""
    walls = [w for w in _oi_walls(raw_chain, live_price, direction, min_pts, max_pts) if w[3]]
    if not walls:
        return None
    dist, strike, _oi, _sig = walls[0]
    buf = max(OI_FRONT_RUN_MIN_PTS, OI_FRONT_RUN_ATR * float(atr)) if atr else OI_FRONT_RUN_MIN_PTS
    tgt = strike - buf if direction == "BUY" else strike + buf
    return round(tgt, 2), float(strike)


def _gemini_compact_context(dashboard_context, max_total_chars=70000):
    """Build a bounded but broad live-data packet for Gemini.

    The deterministic engines remain the source of truth for calculations. Gemini
    receives the current Upstox-derived snapshot plus every named research section,
    but large historical tables are bounded so the 30-second dashboard refresh does
    not burn quota unnecessarily.
    """
    ctx = dashboard_context or {}
    preferred = [
        "Market Status", "Data Freshness", "Nifty Spot Price",
        "RAW Latest Price/Indicator Snapshot", "RAW Primary Price/Indicator Data (latest 500 candles)",
        "RAW Option Chain (all loaded strikes)", "RAW Full NIFTY 50 Breadth/Stock Data",
        "RAW Global Market Table", "RAW Global News/Sentiment Table", "RAW Global Research",
        "NIFTY Global Research Bias", "RAW Multi-Timeframe Structure", "RAW SMC Zones / FVG / OB / Sweeps",
        "RAW Liquidity Map", "RAW S/R Ladder", "RAW Comprehensive S/R Zones",
        "RAW Regime Engine", "RAW CPR/Pivots/Opening Range", "RAW Order Flow", "RAW ML Results",
        "RAW Backtest Report", "RAW Monte Carlo Report", "RAW Model Drift Report",
        "RAW Position Sizing", "RAW Risk Engine", "RAW Sniper Setup", "RAW Hybrid AI Analysis",
    ]
    out=[]; used=0
    for k in preferred:
        if k not in ctx: continue
        v=str(ctx.get(k))
        # Keep raw snapshot sections more complete than verbose narrative sections.
        lim=18000 if "Primary Price" in k or "Option Chain" in k else 6500
        if "Primary Price" in k:
            # BUG FIX: this table is 500 candles in chronological order (oldest first).  Cutting the first
            # 18000 characters handed Gemini the OLDEST candles (previous days, ~150 pts away from spot) and it
            # then rejected live setups with "raw candles show a price mismatch (~22700 vs 22552 spot)".
            # Send the LATEST candles, rounded, with an explicit label.
            try:
                rows=json.loads(v)
                rows=rows[-90:]
                for r in rows:
                    for kk,vv in list(r.items()):
                        if isinstance(vv,float): r[kk]=round(vv,2)
                v=("(chronological, OLDEST first, LAST row = newest/current candle; last "+str(len(rows))+" candles)\n"
                   +json.dumps(rows,separators=(",",":"),default=str))
                lim=24000
                if len(v)>lim:
                    v=v[:60]+"...\n"+v[-(lim-60):]
            except Exception:
                v=v[-lim:]
        v=v[:lim] if "Primary Price" not in k else v
        block=f"\n### {k}\n{v}\n"
        if used+len(block)>max_total_chars: break
        out.append(block); used += len(block)
    return "".join(out)


# ---------------------------------------------------------------------------
# GEMINI FINAL REVIEW  (v2 -- fixes the "Gemini review failed: HTTPError" dead-lock)
#
# ROOT CAUSE of the HTTPError: this function was hard-wired to `gemini-2.5-flash`.
# Google is retiring the Gemini 2.5 family (shutdown 16 Oct 2026) and retired models
# answer HTTP 404 -- raise_for_status() -> "HTTPError" -> the old code failed CLOSED
# and silently cancelled every trade the strategy+AI layers had approved.
# (The other modules in this project already use gemini-3.1-flash-lite, which is why
# only the trade review broke.)
#
# What is different now:
#   * model fallback chain, remembers the last working model, and quarantines dead ones
#   * real diagnostics (HTTP status + Google's error message) instead of "HTTPError"
#   * Gemini forms its OWN independent view first, then compares with the proposal
#   * three distinct outcomes:  APPROVED / REJECTED (explicit no)  /  UNAVAILABLE (tech failure)
#   * UNAVAILABLE no longer means "no trade forever": see local_fallback_review()
# ---------------------------------------------------------------------------
GEMINI_FAIL_OPEN = True   # Gemini *technically* unreachable (quota/5xx) -> the strict local fallback review decides (never after an explicit REJECT). Every trade records WHICH reviewer approved it.


def _gemini_extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        return json.loads(text[a:b + 1])
    raise ValueError("no JSON object in Gemini reply")


def gemini_trade_review(api_keys, dashboard_context, strategy_result, proposed_direction,
                        live_price=None, strategy_formed_at=None, decision=None, use_search=True):
    """Independent Gemini second opinion, called ONLY after a live named strategy and the
    self-learning AI engine already agree.  Gemini first forms its OWN view from the live
    packet, then approves only if that view matches the proposal.

    Returns status APPROVED | REJECTED (Gemini said no) | UNAVAILABLE (technical failure).
    """
    keys = [k for k in (api_keys or []) if k]
    if not keys:
        return {"status": "UNAVAILABLE", "approved": False, "reason": "No Gemini API key configured.", "risk_flags": ["gemini_unavailable"]}
    direction = str(proposed_direction).upper()
    if direction not in ("BUY", "SELL"):
        return {"status": "REJECTED", "approved": False, "reason": "Invalid proposed direction."}
    packet = _gemini_compact_context(dashboard_context)
    brief = (strategy_result or {}).get("strategy_brief") or {}
    strategy_json = json.dumps({
        "direction": (strategy_result or {}).get("direction"), "score": (strategy_result or {}).get("score"),
        "regime": (strategy_result or {}).get("regime"), "selected": (strategy_result or {}).get("selected_strategies"),
        "brief": brief,
    }, ensure_ascii=False, default=str)[:12000]
    _dec = {k: (decision or {}).get(k) for k in (
        "direction", "underlying_entry", "strike", "option_type", "level_price", "confidence_note", "entry", "entry_price", "stop_loss", "sl", "target", "target_1", "target_2", "risk_reward",
        "confidence_pct", "critical_confirmations", "major_conflicts", "reason", "reasons", "factor_flags",
        "option_symbol", "strategy_names") if (decision or {}).get(k) is not None}
    decision_json = json.dumps(_dec, ensure_ascii=False, default=str)[:6000]
    # Locally computed LIVE price-action facts so Gemini judges direction on the same evidence the engine used.
    try:
        _ls = (strategy_result or {}).get("live_state") or {}
        live_json = json.dumps({"live_bias": _ls.get("bias"), "live_score(-100..100)": _ls.get("score"),
                                "bounce_from_12bar_low_ATR": _ls.get("bounce_atr"), "drop_from_12bar_high_ATR": _ls.get("drop_atr"),
                                "move_last_12_bars_ATR": _ls.get("move12_atr"),
                                "buy_consensus": (strategy_result or {}).get("buy_consensus"),
                                "sell_consensus": (strategy_result or {}).get("sell_consensus"),
                                "strategy_switched_to_match_ai": bool((strategy_result or {}).get("rebased_from"))}, default=str)
    except Exception:
        live_json = "{}"
    prompt = (
        "You are the FINAL independent reviewer for an intraday NIFTY trading system.\n\n"
        f"A named strategy is live and the self-learning AI engine agrees. Proposed direction: {direction}.\n"
        "STEP 1: From the LIVE RESEARCH PACKET alone, form your OWN independent view: BUY, SELL or NO_TRADE.\n"
        "STEP 2: APPROVE only if your own view equals the proposed direction. You cannot invent a trade or flip direction.\n\n"
        "Rules:\n"
        "1) Use only the supplied packet; never treat missing data as bullish/bearish.\n"
        "2) REJECT if data is stale/frozen, clearly contradictory, or price is already extended away from its level.\n"
        "2b) REJECT if the entry would CHASE a move: e.g. a SELL after the index has already fallen sharply today and is at/below its nearest support or has just bounced, or a BUY after a sharp rise at/above resistance or just after a pullback starts. Prefer a pullback/retest entry.\n"
        "2c) The candle table is chronological (oldest first, LAST row = current candle) and spans several days: "
        "older rows at different price levels are NORMAL, never a data mismatch. Judge price from the LAST rows and the Live price given above.\n"
        "3) Secondary factors may disagree; require the MAIN evidence (structure, VWAP, order flow, option OI, regime) to lean your way.\n"
        "3b) Do NOT reject only because a minor/secondary factor disagrees or because a pullback is possible. Reject for: stale data, "
        "price chasing, or MAIN evidence leaning the opposite way. If the setup is a retest/pullback entry that is NOT extended, approve it.\n"
        "4) No certainty claims.\n\n"
        f"Live price: {live_price}\nStrategy formed at: {strategy_formed_at}\nStrategy summary:\n{strategy_json}\n\n"
        f"LIVE PRICE-ACTION CHECK (computed by the engine from the latest candles):\n{live_json}\n\n"
        f"AI TRADE DECISION REPORT (entry / SL / target / confidence / reasons):\n{decision_json}\n\n"
        "You may ALSO use Google Search for fresh NIFTY news / global cues / events today to add your own research, "
        "but price levels and indicators must come from the packet. Check the entry, SL and target levels are sensible "
        "for the live price (not already extended, SL not too tight, target realistic).\n\n"
        f"LIVE RESEARCH PACKET:\n{packet}\n\n"
        "Return ONLY JSON with keys: own_direction (BUY/SELL/NO_TRADE), approved (boolean), verdict (APPROVE/REJECT), "
        "confidence (0-100), reason (short string), risk_flags (array of strings)."
    )
    body_json = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}
    # Gemini cannot combine JSON-mode with the search tool -> grounded body has no responseMimeType
    body_search = {"contents": [{"parts": [{"text": prompt}]}], "tools": [{"google_search": {}}]}

    def _post(url, hdr, body, timeout):
        resp = requests.post(url, headers=hdr, json=body, timeout=timeout)
        if resp.status_code != 200:
            try:
                msg = (resp.json().get("error", {}) or {}).get("message", "")[:200]
            except Exception:
                msg = resp.text[:200]
            raise gemini_pool.PoolHTTPError(resp.status_code, msg)
        cand = (resp.json().get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def _one(key, model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        hdr = {"Content-Type": "application/json", "x-goog-api-key": key}
        if use_search:
            try:
                out = _gemini_extract_json(_post(url, hdr, body_search, 40))
                out["_search_used"] = True
                return out
            except Exception as exc_s:
                # ANY problem with the search-grounded call (400 unsupported, 429 search quota, timeout,
                # non-JSON prose...) -> retry the SAME key/model once WITHOUT search, so a search problem
                # never burns this key/model or cascades through all 5 keys.
                logger.warning("Grounded review failed (%s) -> plain JSON retry", type(exc_s).__name__)
        try:
            out = _gemini_extract_json(_post(url, hdr, body_json, 25))
        except ValueError:
            raise gemini_pool.PoolHTTPError(502, "Gemini reply was not valid JSON")
        out["_search_used"] = False
        return out

    try:
        parsed, _idx, model = gemini_pool.run(keys, gemini_pool.REVIEW_MODELS, _one, max_attempts=30, deadline_s=REVIEW_DEADLINE_S)
    except gemini_pool.GeminiUnavailable as gu:
        return {"status": "UNAVAILABLE", "approved": False, "direction": "NO_TRADE", "verdict": "UNAVAILABLE",
                "confidence": 0, "reason": f"{gu.last_exc}"[:200], "attempts": gu.attempts[-6:],
                "retry_in_s": round(gu.wait_s), "risk_flags": ["gemini_unavailable"], "model": None,
                "details": list(getattr(gu, "details", []) or [])[-8:]}
    own = str(parsed.get("own_direction", parsed.get("direction", ""))).upper()
    approved = bool(parsed.get("approved")) and str(parsed.get("verdict", "")).upper() == "APPROVE" and own in (direction, "")
    out = {"direction": own or direction, "confidence": parsed.get("confidence"), 
           "risk_flags": parsed.get("risk_flags") or [], "model": model, "own_direction": own,
           "search_used": bool(parsed.get("_search_used"))}
    if approved:
        out.update({"status": "APPROVED", "approved": True, "direction": direction, "verdict": "APPROVE",
                    "reason": parsed.get("reason") or "Gemini's own read matches the proposed setup."})
    else:
        out.update({"status": "REJECTED", "approved": False, "verdict": "REJECT",
                    "reason": parsed.get("reason") or "Gemini's own read does not support the proposed setup."})
    return out


def local_fallback_review(decision, gemini=None):
    """Used ONLY when Gemini is genuinely out of quota (never after an explicit Gemini REJECT, never after a mere timeout).

    It must behave like a careful human reviewer, not a rubber stamp.  Besides the numbers it now has PRICE-ACTION sense:
    it blocks a CHASE -- entering after the move already happened (RSI extreme, outside the Bollinger band, far from EMA20,
    straight-line candles, entry right after an impulse candle) unless a sweep / rejection wick / pullback confirms it.
      * normal tier : EV >= MIN_EV_R + 0.05, >= 5 direct confirmations, 0 major conflicts, R:R >= 1.5, no chase
      * strong tier : >= 8 direct confirmations -> up to 1 major conflict allowed (still no chase)
    """
    d = decision or {}
    ev = float(d.get("expectancy_r", -1.0) or -1.0)
    crit = int(d.get("critical_confirmations", 0) or 0)
    conflicts = int(d.get("major_conflicts", 0) or 0)
    rr = float(d.get("risk_reward", 0) or 0)
    strong = crit >= 8
    max_conflicts = 1 if strong else 0
    need_ev = MIN_EV_R + 0.05
    need_crit = 5
    import entry_quality as _eqm
    chase_block, chase_why = _eqm.fallback_blocks(d.get("entry_quality"))
    ok = (ev >= need_ev and crit >= need_crit and conflicts <= max_conflicts and rr >= 1.5 and not chase_block)
    tier = "strong" if strong else "normal"
    why = (f"[{tier} tier] expectancy {ev:+.2f}R (need {need_ev:+.2f}R), {crit} direct confirmations (need {need_crit}), "
           f"{conflicts} major conflicts (max {max_conflicts}), R:R {rr:.2f} (need 1.5)"
           + (f", CHASE BLOCK: {chase_why}" if chase_block else ""))
    g = ""
    if isinstance(gemini, dict) and gemini.get("reason"):
        g = f" | Gemini problem: {str(gemini.get('reason'))[:140]}"
        if gemini.get("details"):
            g += " [" + "; ".join(gemini["details"][-5:]) + "]"
        elif gemini.get("attempts"):
            g += f" ({', '.join(gemini['attempts'][-6:])})"
    return {"status": "LOCAL_FALLBACK_APPROVED" if ok else "LOCAL_FALLBACK_BLOCKED", "approved": ok,
            "direction": d.get("direction"),
            "reason": ("Gemini out of quota; local review " + ("APPROVED: " if ok else "BLOCKED: ") + why + g)}


_GATE_PATTERNS = (
    ("Gemini review pending (background)", "background me"),
    ("Gemini final review", "Final review did not approve"),
    ("No strategy formed", "No named strategy"),
    ("Strategy vs AI direction mismatch", "AI research is"),
    ("Stale data", "STALE/FROZEN"),
    ("Event risk window", "Event risk window"),
    ("Win-chance floor", "below the minimum"),
    ("Outside entry window", "entry window"),
    ("Cooldown / pause", "Cooldown"),
    ("Same setup already traded", "Same setup already traded"),
    ("Strong global veto", "Global research is STRONGLY"),
    ("Falling/rising knife", "will not buy the turning point"),
    ("Falling/rising knife", "will not sell the turning point"),
    ("S/R location / confirmation", "needs"),
    ("S/R location / confirmation", "rejection"),
    ("S/R map unclear", "S/R"),
    ("Not enough confluence", "confluence is not strong"),
    ("Entry too far from level", "too far from the validated level"),
    ("No positive edge (EV)", "No positive edge"),
    ("Self-learning segment paused", "segment '"),
    ("No level / strategy nearby", "No key level"),
)


# ---------------------------------------------------------------------------
# NON-BLOCKING GEMINI REVIEW.  The dashboard is one Streamlit script that re-runs every 30 s; the review used to run
# INSIDE it, so a slow/overloaded Gemini froze the whole page ("Stop" button, nothing loads).  Now the review runs in a
# background thread: the page renders immediately, and the finished verdict is picked up on the next refresh (30 s).
# Gemini remains the final block: while the verdict is pending, there is simply no trade.
# ---------------------------------------------------------------------------
import threading as _threading

_REVIEW_JOBS = {}            # sig -> {"state": "RUNNING"|"DONE", "t0": ts, "result": dict|None}
_REVIEW_LOCK = _threading.Lock()
REVIEW_STALE_S = 240.0       # a job that never finished (thread died) is abandoned after this long


def gemini_review_async(sig, api_keys, dashboard_context, strategy_result, proposed_direction, **kw):
    """Returns the finished review dict, or {"status": "PENDING", "approved": False, ...} while it is still running.
    `sig` identifies the setup (candle + direction); the same sig never starts two jobs."""
    now = time.time()
    with _REVIEW_LOCK:
        job = _REVIEW_JOBS.get(sig)
        if job and job["state"] == "DONE":
            return job["result"]
        if job and job["state"] == "RUNNING" and now - job["t0"] < REVIEW_STALE_S:
            return {"status": "PENDING", "approved": False,
                    "reason": f"Gemini review background me chal raha hai ({int(now - job['t0'])}s) -- next refresh me result aayega."}
        _REVIEW_JOBS[sig] = {"state": "RUNNING", "t0": now, "result": None}
        for k in [k for k, v in _REVIEW_JOBS.items() if now - v["t0"] > 3600]:
            _REVIEW_JOBS.pop(k, None)
    ctx_copy = dict(dashboard_context or {})          # the thread must not read a live Streamlit session dict
    strat_copy = dict(strategy_result or {})

    def _work():
        try:
            res = gemini_trade_review(api_keys, ctx_copy, strat_copy, proposed_direction, **kw)
        except Exception as exc:                       # never leave the job RUNNING forever
            logger.exception("background Gemini review crashed")
            res = {"status": "UNAVAILABLE", "approved": False, "verdict": "UNAVAILABLE",
                   "reason": f"review crashed: {type(exc).__name__}", "risk_flags": ["gemini_unavailable"]}
        with _REVIEW_LOCK:
            _REVIEW_JOBS[sig] = {"state": "DONE", "t0": time.time(), "result": res}

    _threading.Thread(target=_work, daemon=True, name="gemini-review").start()
    return {"status": "PENDING", "approved": False,
            "reason": "Gemini review background me shuru hua -- result next refresh (~30s) me aayega; tab tak koi trade nahi."}


def review_job_forget(sig):
    """Drop a finished verdict so the next refresh can ask again (used after short-lived outages)."""
    with _REVIEW_LOCK:
        _REVIEW_JOBS.pop(sig, None)


def classify_block(reason):
    """Name of the gate that stopped a setup, for the 'which gate blocks most' diagnostics."""
    t = str(reason or "")
    for name, needle in _GATE_PATTERNS:
        if needle.lower() in t.lower():
            return name
    return "Other"


def apply_gemini_verdict(decision, review, now=None):
    """Combine the engine's own decision with Gemini's final review.

    Gemini is the FINAL BLOCK: an explicit Gemini REJECT = no trade -- no probe, no reduced size.
    Only when Gemini is technically unreachable does the app swap in the strict local fallback review (GEMINI_FAIL_OPEN);
    that approval is stored as reviewer=LOCAL_FALLBACK so it can never be mistaken for a Gemini approval.
    (Skipping a doubtful trade is better than taking a loss.)  When Gemini approves, the learner records
    that in the `external_ai_aligned` factor so it can learn how much Gemini's approval is worth."""
    d = dict(decision or {})
    r = review or {}
    d["gemini_review"] = r
    if not d.get("has_setup"):
        return d
    if r.get("approved"):
        flags = dict(d.get("factor_flags") or {})
        ok = str(r.get("status", "")).upper() == "APPROVED"
        flags["gemini_final_review_approved"] = ok
        flags["external_ai_aligned"] = ok
        # PERMANENT PROOF stored with the trade: which review approved it (shown in the Recent AI Setups table).
        flags["_gemini"] = {"status": str(r.get("status", "")), "model": r.get("model"),
                            "search_used": bool(r.get("search_used")),
                            "confidence": r.get("confidence"),
                            "reason": str(r.get("reason") or "")[:500],
                            "gemini_error": str(r.get("gemini_error") or "")[:300] or None,   # why Gemini was unreachable (fallback only)
                            "reviewer": "GEMINI" if ok else "LOCAL_FALLBACK",
                            "at": trade_learning._now_str()}
        d["factor_flags"] = flags
        d["gemini_final_review_approved"] = ok
        return d
    d["has_setup"] = False
    d["reason"] = "Final review did not approve this strategy+AI setup: " + str(r.get("reason") or "REJECT/UNAVAILABLE")
    return d


def _win_floor(now):
    """(floor, note).  Normal floor MIN_WIN_PROB; if the engine has logged NO setup for FLOOR_RELAX_AFTER_TRADING_DAYS trading days
    the floor eases to FLOOR_RELAXED_PROB so the new quality gates can never silently freeze it again.  Shadow trades keep learning
    either way."""
    try:
        rows = trade_learning.get_recent_setups(limit=1)
        if not rows:
            return FLOOR_RELAXED_PROB, "no setup logged yet -> floor eased"
        last = datetime.strptime(str(rows[0].get("timestamp"))[:10], "%Y-%m-%d").date()
        import numpy as _np
        gap = int(_np.busday_count(last, now.date()))
        if gap >= FLOOR_RELAX_AFTER_TRADING_DAYS:
            return FLOOR_RELAXED_PROB, f"no setup for {gap} trading days -> floor eased to {100 * FLOOR_RELAXED_PROB:.0f}%"
    except Exception:
        pass
    return MIN_WIN_PROB, ""


def _is_drought(now):
    """True when the engine has been flat for long enough that the EV bar may ease slightly.
    Prevents the 'bar rises -> no trades -> no new data -> bar stays high' dead-lock."""
    try:
        rows = trade_learning.get_recent_setups(limit=1)
    except Exception:
        return False
    if not rows:
        return now.time() >= DROUGHT_CHECK_FROM
    try:
        last = datetime.strptime(str(rows[0].get("timestamp"))[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return False
    if last.date() != now.date():
        return now.time() >= DROUGHT_CHECK_FROM
    return (now - last) >= timedelta(minutes=DROUGHT_AFTER_MIN)


def generate_trade_decision(live_price, level_prediction, atr, max_pain=None,
                             signal_code=0, ml_agrees=False, banknifty_correlation_note=None,
                             breadth_advances=None, breadth_declines=None,
                             global_avg_change=None, live_vix=None, india_news_sentiment=None,
                             level_ladder=None, sr_context=None, sniper_bias=None, is_choppy=False,
                             dashboard_context=None, global_research=None,
                             nifty50_news_sentiment=None, nifty50_fundamentals_bias=None,
                             strategy_result=None, raw_option_chain=None, now_ist=None,
                             entry_quality=None, pcr_velocity=None):
    """
    Returns a dict:
      If NO qualifying setup right now:
        {"has_setup": False, "reason": "..."}
      If a qualifying setup exists:
        {"has_setup": True, "direction": "BUY"/"SELL", "strike": int,
         "option_type": "CE"/"PE", "underlying_entry": float,
         "stop_loss": float, "target": float, "confidence_pct": float,
         "confidence_note": str, "factor_flags": dict, "factors_true": int,
         "factors_total": int}

    The engine uses a weighted-confluence threshold rather than requiring
    every secondary factor to be true. Direct structure/level evidence gets
    more weight; contextual factors strengthen but do not dominate.
    """
    # The dashboard now passes its CURRENT master context directly into this
    # engine.  Keep the context intact (do not reduce it to a small score) so
    # every available module can be audited before a setup is accepted.
    dashboard_context = dashboard_context or {}
    required_context_sections = (
        "Market Status", "Data Freshness", "Nifty Spot Price",
        "RAW Option Chain (all loaded strikes)",
        "RAW Full NIFTY 50 Breadth/Stock Data",
        "RAW Global Market Table", "RAW Global News/Sentiment Table",
        "RAW Multi-Timeframe Structure", "RAW SMC Zones / FVG / OB / Sweeps",
        "RAW Liquidity Map", "RAW S/R Ladder", "RAW Comprehensive S/R Zones", "RAW Regime Engine",
        "RAW CPR/Pivots/Opening Range", "RAW Order Flow", "RAW ML Results",
        "RAW Backtest Report", "RAW Monte Carlo Report", "RAW Model Drift Report",
        "RAW Position Sizing", "RAW Risk Engine", "RAW Sniper Setup",
        "RAW Hybrid AI Analysis",
    )
    missing_context_sections = [k for k in required_context_sections if k not in dashboard_context]
    context_audit = {
        "sections_received": len(dashboard_context),
        "required_sections": len(required_context_sections),
        "missing_sections": missing_context_sections,
        "complete": not missing_context_sections,
    }

    # Do not let a lagging top-level signal freeze the engine.  If the
    # validated level predictor already has a strong directional read, use it
    # as an EARLY candidate and let the full factor-vote engine validate it.
    early_lp = level_prediction if isinstance(level_prediction, dict) else {}
    early_status = early_lp.get("status")
    early_bias = str(early_lp.get("directional_bias", "")).lower()
    early_dir = 1 if "bullish" in early_bias else (-1 if "bearish" in early_bias else 0)
    early_pct = float((early_lp.get("break_pct") if "bullish" in early_bias else early_lp.get("bounce_pct")) or 0.0)
    early_trigger = early_status == "APPROACHING_LEVEL" and early_dir != 0 and early_pct >= 60.0

    strategy_result = strategy_result or {}
    # ENTRY POLICY V24: the strategy layer is the trigger.  The self-learning
    # AI engine remains the independent research/confirmation layer.  A
    # directional trade is impossible without one currently-formed named
    # strategy AND a same-direction AI research bias.  This prevents the old
    # "AI says BUY somewhere, so enter" behaviour while keeping strategy
    # conditions soft enough that one valid strategy can qualify.
    strategy_ready = bool(strategy_result.get("has_setup")) and float(strategy_result.get("score", 0) or 0) >= strategy_engine.MIN_ENTRY_SCORE
    strategy_dir = 1 if str(strategy_result.get("direction")) == "BUY" else -1 if str(strategy_result.get("direction")) == "SELL" else 0
    strategy_path = strategy_ready and strategy_dir != 0
    ai_research_dir = signal_code if signal_code in (1, -1) else 0
    ai_research_source = "main_signal" if ai_research_dir else "none"
    if ai_research_dir == 0:
        # The headline signal is often 0 (neutral) even while the independent evidence
        # clearly leans one way -- that alone used to veto every strategy.  The AI engine
        # now does its own multi-factor research vote; it needs a clear lean (|net| >= 3).
        _op, _ = _extract_order_flow(dashboard_context)
        _rg = _extract_regime(dashboard_context)
        _mt = _extract_mtf(dashboard_context)
        _sn = str(sniper_bias or "").upper()
        _net = 0
        _net += 1 if _op == "BUYING PRESSURE" else -1 if _op == "SELLING PRESSURE" else 0
        _net += 1 if _rg == "BULLISH" else -1 if _rg == "BEARISH" else 0
        _net += 1 if _mt == "BULLISH" else -1 if _mt == "BEARISH" else 0
        if breadth_advances is not None and breadth_declines is not None:
            _net += 1 if breadth_advances > breadth_declines else -1 if breadth_declines > breadth_advances else 0
        if global_avg_change is not None:
            _net += 1 if global_avg_change > 0.1 else -1 if global_avg_change < -0.1 else 0
        _net += 1 if "BULLISH" in _sn else -1 if "BEARISH" in _sn else 0
        _net += 1 if (ml_agrees and strategy_dir == 1) else -1 if (ml_agrees and strategy_dir == -1) else 0
        # LIVE price action (EMA/VWAP/momentum/supertrend) is part of the AI's own research too.
        try:
            _lsc = float(((strategy_result.get("live_state") or {}).get("score", 0)) or 0)
            _net += 2 if _lsc >= 40 else -2 if _lsc <= -40 else 1 if _lsc >= 20 else -1 if _lsc <= -20 else 0
        except Exception:
            pass
        if abs(_net) >= 3:
            ai_research_dir = 1 if _net > 0 else -1
            ai_research_source = f"multi_factor_vote({_net:+d})"

    # v35: the AI research has a clear direction but the top strategy points the other way (or nothing
    # qualified): do NOT dead-lock - look through ALL live, live-valid strategies for the best one in the
    # AI's direction.  If one exists, it becomes the entry strategy.  If none exists, we wait honestly.
    _rebased_note = None
    if ai_research_dir != 0 and (not strategy_path or ai_research_dir != strategy_dir):
        _want = "BUY" if ai_research_dir == 1 else "SELL"
        _rb = strategy_engine.rebase_to_direction(strategy_result, _want)
        if _rb:
            _rebased_note = (f"Top strategy was {strategy_result.get('direction')}; switched to best live {_want} strategy "
                             f"{', '.join(_rb.get('selected_strategies') or [])} because AI research + live price action say {_want}.")
            strategy_result = _rb
            strategy_ready = True
            strategy_dir = ai_research_dir
            strategy_path = True

    if not strategy_path:
        return {"has_setup": False,
                "reason": "No named strategy is currently formed strongly enough. AI research continues, but it cannot open an entry by itself.",
                "context_audit": context_audit, "strategy_required": True}

    if ai_research_dir == 0:
        # AI research is NEUTRAL (no clear opposite view): it no longer blocks the strategy.
        # The strategy's direction is judged by the full-data confluence + confidence checks below,
        # and then by Gemini. Only an OPPOSITE AI direction vetoes.
        ai_research_dir = strategy_dir
        ai_research_source = "neutral_ai_strategy_judged_by_full_data"

    if ai_research_dir != strategy_dir:
        return {"has_setup": False,
                "reason": f"Strategy is {('BUY' if strategy_dir == 1 else 'SELL')} but AI research is {('BUY' if ai_research_dir == 1 else 'SELL')} and no live {('BUY' if ai_research_dir == 1 else 'SELL')} strategy has formed yet; waiting (no trade until both directions match).",
                "context_audit": context_audit, "strategy_required": True,
                "strategy_direction": "BUY" if strategy_dir == 1 else "SELL",
                "ai_research_direction": "BUY" if ai_research_dir == 1 else "SELL"}

    # Once both layers agree, the strategy direction is the entry direction.
    # Other dashboard factors remain weighted evidence/risk checks, not a
    # requirement that every indicator must agree simultaneously.
    effective_signal = strategy_dir

    freshness = str(dashboard_context.get("Data Freshness", "")).upper()
    if "STALE" in freshness or "FROZEN" in freshness:
        return {"has_setup": False,
                "reason": "Dashboard data is STALE/FROZEN. Full-data trade engine will not open a fresh setup until live data is fresh.",
                "context_audit": context_audit}

    _now = now_ist or trade_learning._now_ist()
    if ENFORCE_SESSION_WINDOW and not (NO_ENTRY_BEFORE <= _now.time() <= NO_ENTRY_AFTER):
        return {"has_setup": False,
                "reason": (f"Outside the entry window ({NO_ENTRY_BEFORE.strftime('%H:%M')}-{NO_ENTRY_AFTER.strftime('%H:%M')} IST): "
                           "the first minutes are opening noise and late entries cannot play out before the end-of-day square-off."),
                "context_audit": context_audit}

    # EVENT RISK: scheduled news (RBI / Fed reaction / Budget ...) moves price on news, not on the levels this engine reads.
    _event = event_calendar.event_risk(_now)
    if _event.get("level") == "BLOCK":
        return {"has_setup": False,
                "reason": f"Event risk window: {_event.get('reason')}. No NEW entries until the news has been absorbed (edit event_calendar.py to change).",
                "context_audit": context_audit}

    # Context completeness is audit information, not a trade gate. Missing
    # secondary sections are treated as UNKNOWN evidence; genuinely stale data
    # remains a hard safety stop. This prevents the old "wait for all 70
    # sections" deadlock.

    raw_direction = "BUY" if effective_signal == 1 else "SELL"
    direction = raw_direction
    reversal_used = False
    reversal_reason = ""

    # EARLY REVERSAL PATH: the main directional model can lag when a strong
    # trend suddenly breaks.  We allow an earlier SELL/BUY only when the
    # completed-candle momentum break is backed by multiple independent
    # context checks.  A single bearish/bullish candle can never flip the
    # trade direction.
    _sr_conf_preview = (sr_context or dashboard_context.get("RAW Comprehensive S/R Zones") or {}).get("confirmation", {}) if isinstance((sr_context or dashboard_context.get("RAW Comprehensive S/R Zones") or {}), dict) else {}
    _loc_preview = str((sr_context or dashboard_context.get("RAW Comprehensive S/R Zones") or {}).get("location", "UNKNOWN")).upper() if isinstance((sr_context or dashboard_context.get("RAW Comprehensive S/R Zones") or {}), dict) else "UNKNOWN"
    _order_pressure_preview, _ = _extract_order_flow(dashboard_context)
    _regime_preview = _extract_regime(dashboard_context)
    _mtf_preview = _extract_mtf(dashboard_context)
    _smc_text_preview = _flatten_context_text(dashboard_context.get("RAW SMC Zones / FVG / OB / Sweeps"))
    _liq_text_preview = _flatten_context_text(dashboard_context.get("RAW Liquidity Map"))
    _breadth_reversal = False
    if breadth_advances is not None and breadth_declines is not None:
        _breadth_reversal = (breadth_advances - breadth_declines) < 0 if raw_direction == "BUY" else (breadth_advances - breadth_declines) > 0

    _bearish_reversal_checks = [
        bool(_sr_conf_preview.get("bearish_momentum_break_confirmed")),
        _order_pressure_preview == "SELLING PRESSURE",
        _mtf_preview == "BEARISH",
        _regime_preview in ("BEARISH", "RANGE"),
        ("bearish" in _smc_text_preview and ("sweep" in _smc_text_preview or "order block" in _smc_text_preview or "fvg" in _smc_text_preview)),
        ("liquidity sweep" in _liq_text_preview or "equal high" in _liq_text_preview),
        _breadth_reversal,
    ]
    _bullish_reversal_checks = [
        bool(_sr_conf_preview.get("bullish_momentum_break_confirmed")),
        _order_pressure_preview == "BUYING PRESSURE",
        _mtf_preview == "BULLISH",
        _regime_preview in ("BULLISH", "RANGE"),
        ("bullish" in _smc_text_preview and ("sweep" in _smc_text_preview or "order block" in _smc_text_preview or "fvg" in _smc_text_preview)),
        ("liquidity sweep" in _liq_text_preview or "equal low" in _liq_text_preview),
        not _breadth_reversal,
    ]
    _bearish_reversal_score = sum(_bearish_reversal_checks)
    _bullish_reversal_score = sum(_bullish_reversal_checks)

    _bearish_transition = bool(_sr_conf_preview.get('bearish_trend_transition_confirmed'))
    _bullish_transition = bool(_sr_conf_preview.get('bullish_trend_transition_confirmed'))

    # NEVER BUY A FALLING KNIFE / SELL A RISING KNIFE. If the latest completed
    # candle has started reversing the immediately preceding move, the raw
    # signal is vetoed unless the opposite direction has a full-context
    # structural reversal at the correct boundary.
    if raw_direction == 'BUY' and _bearish_transition and not (
        _bullish_reversal_score >= 5 and _loc_preview in ('AT_SUPPORT', 'NEAR_SUPPORT')
    ):
        return {"has_setup": False,
                "reason": "BUY blocked: price has started turning down after the prior rise. The engine will not buy the turning point; it will wait for full-context bullish reversal/support defence.",
                "context_audit": context_audit}
    if raw_direction == 'SELL' and _bullish_transition and not (
        _bearish_reversal_score >= 5 and _loc_preview in ('AT_RESISTANCE', 'NEAR_RESISTANCE')
    ):
        return {"has_setup": False,
                "reason": "SELL blocked: price has started turning up after the prior fall. The engine will not sell the turning point; it will wait for full-context bearish reversal/resistance rejection.",
                "context_audit": context_audit}

    # Direction flip is deliberately harder than an ordinary entry: price
    # action + order flow + MTF + regime + at least one structural/liquidity
    # confirmation are required. This prevents "dekhte hi SELL" behaviour.
    # Strategy-led entries NEVER flip: strategy and AI must agree on one direction, so flipping
    # it afterwards would silently break that rule (e.g. SELL strategy turned into a BUY).
    if strategy_path:
        pass
    elif raw_direction == "BUY" and _bearish_reversal_score >= 5 and _loc_preview in ("AT_RESISTANCE", "NEAR_RESISTANCE"):
        direction = "SELL"
        reversal_used = True
        reversal_reason = f"Early bearish reversal accepted after {_bearish_reversal_score}/7 independent confirmations; full context was checked before changing BUY to SELL."
    elif raw_direction == "SELL" and _bullish_reversal_score >= 5 and _loc_preview in ("AT_SUPPORT", "NEAR_SUPPORT"):
        direction = "BUY"
        reversal_used = True
        reversal_reason = f"Early bullish reversal accepted after {_bullish_reversal_score}/7 independent confirmations; full context was checked before changing SELL to BUY."

    # COOLDOWN: do not re-enter a direction that was just stopped out / pause during a loss streak.
    _cd_blocked, _cd_reason = trade_learning.entry_cooldown(direction, now=_now)
    if _cd_blocked:
        return {"has_setup": False, "reason": _cd_reason, "context_audit": context_audit}

    # ONE SETUP, ONE TRADE: after a trade from a strategy signal closes (target OR stop), the same still-alive signal
    # must not open another trade.  Needs a signal that formed after the previous same-direction trade was opened.
    _used, _used_reason = trade_learning.signal_already_traded(direction, strategy_result.get("formed_at"))
    if _used:
        return {"has_setup": False, "reason": _used_reason, "context_audit": context_audit}

    # GLOBAL RESEARCH VETO (documented in v29 but never implemented): a STRONG opposite global read
    # blocks the trade; moderate/neutral global context only shapes the learned confidence.
    _gr = global_research if isinstance(global_research, dict) else dashboard_context.get("RAW Global Research")
    _gr_aligned = False
    if isinstance(_gr, dict) and str(_gr.get("status", "")).upper() == "AVAILABLE":
        _gb = str(_gr.get("directional_bias", "")).upper()
        _gs = str(_gr.get("strength", "")).upper()
        _g_oppose = (_gb == "SELL" and direction == "BUY") or (_gb == "BUY" and direction == "SELL")
        _gr_aligned = not _g_oppose
        if _g_oppose and _gs == "STRONG":
            return {"has_setup": False,
                    "reason": f"Global research is STRONGLY {_gb} against this {direction}; domestic setup alone is not enough.",
                    "context_audit": context_audit}

    # LOCATION-FIRST HARD GATE.  A directional score is not enough to buy
    # directly into resistance or sell directly into support.  The engine
    # must first prove the level was accepted/rejected with price action.
    sr_context = sr_context or dashboard_context.get("RAW Comprehensive S/R Zones")
    soft_flags = []   # strategy-led entries: S/R location problems are penalties, not vetoes
    if strategy_path and (not isinstance(sr_context, dict) or sr_context.get('status') not in ('OK', 'READY', 'SUFFICIENT')):
        soft_flags.append("S/R map unavailable")
        sr_context = {}
    if not strategy_path and (not isinstance(sr_context, dict) or sr_context.get('status') not in ('OK', 'READY', 'SUFFICIENT')):
        return {"has_setup": False,
                "reason": "Validated multi-source S/R map is unavailable/insufficient. The engine will not manufacture an entry from the directional signal or round-number ladder.",
                "context_audit": context_audit}
    if isinstance(sr_context, dict):
        loc = str(sr_context.get('location', 'UNKNOWN')).upper()
        conf = sr_context.get('confirmation') or {}
        major_support = sr_context.get('major_support') or sr_context.get('nearest_support')
        major_resistance = sr_context.get('major_resistance') or sr_context.get('nearest_resistance')

        # No validated major boundary = do not manufacture an S/R signal from
        # a tiny swing, VWAP, option strike or psychological number.
        if not major_support and not major_resistance and strategy_path:
            soft_flags.append("no validated major S/R nearby")
        elif not major_support and not major_resistance:
            return {"has_setup": False,
                    "reason": "No strong multi-source support/resistance is validated near price. Micro S/R levels are ignored; waiting for a real structural zone.",
                    "context_audit": context_audit}

        # If the S/R engine found overlapping/conflicting boxes around spot,
        # the location is ambiguous. This is exactly the situation that used
        # to produce fake pairs such as 23392-23397 support and 23397-23402 resistance.
        if sr_context.get('overlapping_zone_warning') and strategy_path:
            soft_flags.append("overlapping S/R zones")
        elif sr_context.get('overlapping_zone_warning'):
            return {"has_setup": False,
                    "reason": "Support/resistance zones overlap around current price. Direction is ambiguous, so the engine will not trade until a clean boundary is established.",
                    "context_audit": context_audit}

        at_support = loc in ('AT_SUPPORT', 'NEAR_SUPPORT')
        at_resistance = loc in ('AT_RESISTANCE', 'NEAR_RESISTANCE')

        # EARLY VALIDATED ENTRY is calculated BEFORE any timing/confirmation
        # gate uses it. This keeps the single-entry path alive when the main
        # signal is late but several independent live factors already agree.
        _early_level_pct = 0.0
        if level_prediction and isinstance(level_prediction, dict):
            _early_level_pct = float((level_prediction.get('break_pct') if direction == 'BUY' else level_prediction.get('bounce_pct')) or 0.0)
        _early_order_ok = (direction == 'BUY' and _order_pressure_preview == 'BUYING PRESSURE') or (direction == 'SELL' and _order_pressure_preview == 'SELLING PRESSURE')
        _early_regime_ok = (direction == 'BUY' and _regime_preview == 'BULLISH') or (direction == 'SELL' and _regime_preview == 'BEARISH')
        _early_mtf_ok = (direction == 'BUY' and _mtf_preview == 'BULLISH') or (direction == 'SELL' and _mtf_preview == 'BEARISH')
        _early_structure_ok = (direction == 'BUY' and any(x in _smc_text_preview for x in ('bullish displacement','bullish sweep','bullish order block','bullish fvg'))) or (direction == 'SELL' and any(x in _smc_text_preview for x in ('bearish displacement','bearish sweep','bearish order block','bearish fvg')))
        _early_breadth_ok = not _breadth_reversal
        _early_checks = sum(bool(x) for x in (_early_order_ok, _early_regime_ok, _early_mtf_ok, _early_structure_ok, _early_breadth_ok))
        _early_entry_ok = (_early_level_pct >= 60.0 and _early_checks >= 2)

        # ENTRY-TIMING SAFETY: a trend can still look bullish/bearish while
        # the immediate move is already losing momentum at the exact entry.
        # Do not enter simply because the broad direction is correct. Require
        # the proposed side to pass the adverse-move risk check first.
        timing = sr_context.get('entry_timing') or {}
        if direction == 'BUY' and bool(timing.get('buy_adverse_move_risk')) and not _early_entry_ok and strategy_path:
            soft_flags.append("BUY entry-timing risk")
        elif direction == 'BUY' and bool(timing.get('buy_adverse_move_risk')) and not _early_entry_ok:
            return {"has_setup": False,
                    "reason": "BUY blocked: immediate entry-timing risk is high and the early evidence is not strong enough.",
                    "context_audit": context_audit}
        if direction == 'SELL' and bool(timing.get('sell_adverse_move_risk')) and not _early_entry_ok and strategy_path:
            soft_flags.append("SELL entry-timing risk")
        elif direction == 'SELL' and bool(timing.get('sell_adverse_move_risk')) and not _early_entry_ok:
            return {"has_setup": False,
                    "reason": "SELL blocked: immediate entry-timing risk is high and the early evidence is not strong enough.",
                    "context_audit": context_audit}

        # The key correction: direction alone must NOT choose the side of the
        # level. A BUY is a support-reaction trade when price is at support;
        # it is a breakout trade only when resistance has already been broken
        # and then successfully retested. Likewise, a SELL at resistance is a
        # rejection trade, while a SELL at support requires a confirmed
        # breakdown + retest. This stops the old behaviour of buying into
        # resistance simply because the wider signal was bullish.
        if at_resistance and direction == 'BUY':
            if not conf.get('breakout_retest_confirmed') and not _early_entry_ok and strategy_path:
                soft_flags.append("BUY at resistance without retest")
            elif not conf.get('breakout_retest_confirmed') and not _early_entry_ok:
                return {"has_setup": False, "reason": "BUY at resistance needs breakout/retest confirmation or a strong multi-factor early breakout read.", "context_audit": context_audit}
        if at_support and direction == 'SELL':
            if not conf.get('breakdown_retest_confirmed') and not _early_entry_ok and strategy_path:
                soft_flags.append("SELL at support without retest")
            elif not conf.get('breakdown_retest_confirmed') and not _early_entry_ok:
                return {"has_setup": False, "reason": "SELL at support needs breakdown/retest confirmation or a strong multi-factor early breakdown read.", "context_audit": context_audit}

        # An early reversal is the one controlled exception: when the main
        # model is lagging but a dangerous momentum break at the boundary has
        # already passed the 5/7 multi-factor reversal gate, enter on the
        # break/reaction itself instead of waiting for a second retest candle.
        if reversal_used and direction == 'SELL' and at_resistance and not conf.get('resistance_rejection_confirmed') and not strategy_path:
            if not conf.get('bearish_momentum_break_confirmed'):
                return {"has_setup": False, "reason": "Potential bearish reversal near resistance, but the momentum break is not confirmed yet.", "context_audit": context_audit}
        if reversal_used and direction == 'BUY' and at_support and not conf.get('support_rejection_confirmed') and not strategy_path:
            if not conf.get('bullish_momentum_break_confirmed'):
                return {"has_setup": False, "reason": "Potential bullish reversal near support, but the momentum break is not confirmed yet.", "context_audit": context_audit}

        # At support, a BUY needs actual rejection/defence confirmation; at
        # resistance, a SELL needs actual rejection. A location label alone is
        # never an entry trigger.
        if at_support and direction == 'BUY' and not conf.get('support_rejection_confirmed') and not _early_entry_ok and strategy_path:
            soft_flags.append("BUY at support without rejection")
        elif at_support and direction == 'BUY' and not conf.get('support_rejection_confirmed') and not _early_entry_ok:
            return {"has_setup": False, "reason": "BUY near support needs rejection/defence confirmation or a strong early bullish reaction.", "context_audit": context_audit}
        if at_resistance and direction == 'SELL' and not conf.get('resistance_rejection_confirmed') and not _early_entry_ok and strategy_path:
            soft_flags.append("SELL at resistance without rejection")
        elif at_resistance and direction == 'SELL' and not conf.get('resistance_rejection_confirmed') and not _early_entry_ok:
            return {"has_setup": False, "reason": "SELL near resistance needs rejection confirmation or a strong early bearish reaction.", "context_audit": context_audit}
        # ENTRY LOCATION IS MANDATORY. The round-number ladder is allowed
        # to help with targets/confluence, but it can NEVER create an entry
        # in the middle of a range. This prevents trades such as BUY 23401
        # when the nearest real structure is elsewhere.
        if loc in ('MIDRANGE', 'BETWEEN_LEVELS', 'UNKNOWN') and not strategy_path:
            # Do not manufacture a trade from a random price. However, when
            # the validated ladder has a nearby directional level AND the live
            # full-context evidence is already strong, allow the single entry
            # path to participate instead of remaining flat for days.
            _ladder_candidate_ok = bool(level_ladder) and bool(_early_entry_ok or (
                _order_pressure_preview in ('BUYING PRESSURE','SELLING PRESSURE') and
                _regime_preview in ('BULLISH','BEARISH') and
                _mtf_preview in ('BULLISH','BEARISH')
            ))
            if not _ladder_candidate_ok:
                return {"has_setup": False,
                        "reason": "No valid structural entry location yet; waiting for a confirmed S/R reaction or a strong multi-factor ladder setup.",
                        "context_audit": context_audit}

    # First try the single swing-based nearest level (narrow but most
    # "official" read). If that's not currently in its approach zone,
    # fall back to scanning the full 16-level round-number Ladder --
    # this is what stops the engine from sitting idle for days just
    # because ONE specific swing level never happened to be nearby.
    lp = None
    expected_side = None
    if isinstance(sr_context, dict):
        loc_now = str(sr_context.get('location', 'UNKNOWN')).upper()
        if direction == 'BUY' and loc_now in ('AT_SUPPORT', 'NEAR_SUPPORT'):
            expected_side = 'support'
        elif direction == 'BUY' and loc_now in ('AT_RESISTANCE', 'NEAR_RESISTANCE'):
            expected_side = 'resistance'
        elif direction == 'SELL' and loc_now in ('AT_RESISTANCE', 'NEAR_RESISTANCE'):
            expected_side = 'resistance'
        elif direction == 'SELL' and loc_now in ('AT_SUPPORT', 'NEAR_SUPPORT'):
            expected_side = 'support'

    if level_prediction is not None and level_prediction.get('status') == 'APPROACHING_LEVEL':
        predicted_side = str(level_prediction.get('approaching', '')).lower()
        if expected_side is None or predicted_side == expected_side:
            lp = level_prediction
    if lp is None:
        lp = _find_best_ladder_level(level_ladder, live_price, atr, direction, preferred_side=expected_side)
    if lp is None and strategy_path:
        # Strategy-led entries may be pullbacks/VWAP/momentum setups in the
        # middle of a range. Give the strategy a neutral synthetic location
        # instead of forcing a fake S/R level.
        lp = {"status":"APPROACHING_LEVEL", "approaching":"support" if direction=="BUY" else "resistance",
              "level_price":live_price, "distance_pts":0.0,
              "break_pct":max(65.0,float(strategy_result.get("score",0))),
              "bounce_pct":max(65.0,float(strategy_result.get("score",0))),
              "directional_bias":"Bullish" if direction=="BUY" else "Bearish",
              "factors":["Strategy engine validated setup"]}
    if lp is None:
        return {"has_setup": False, "reason": "No key level or validated strategy setup is close enough with a matching directional read right now.",
                "context_audit": context_audit}

    # Never let the legacy early-warning predictor override the validated
    # structural S/R map. If its level is merely a tiny local swing far away
    # from the selected major zone, discard it and wait for a real level.
    if isinstance(sr_context, dict):
        major_prices = []
        for z in (sr_context.get('major_support'), sr_context.get('major_resistance')):
            if isinstance(z, dict) and z.get('price') is not None:
                major_prices.append(float(z['price']))
        if major_prices and not strategy_path:
            try:
                if min(abs(float(lp.get('level_price', live_price)) - p) for p in major_prices) > max(2.0, 0.50 * float(atr)):
                    return {"has_setup": False,
                            "reason": "The proposed level is only a micro/local level and does not align with the validated multi-source S/R map — trade blocked.",
                            "context_audit": context_audit}
            except Exception:
                logger.exception("Broad exception caught; fallback path executed")
                pass

    bias_lower = lp['directional_bias'].lower()
    level_confirms = ("bullish" in bias_lower and direction == "BUY") or \
                      ("bearish" in bias_lower and direction == "SELL")
    if not level_confirms:
        return {"has_setup": False, "reason": "Main signal and nearest level's read disagree -- staying flat.",
                "context_audit": context_audit}

    confirming_pct = (
        (lp['bounce_pct'] if lp['approaching'] == 'support' else lp['break_pct'])
        if direction == "BUY" else
        (lp['break_pct'] if lp['approaching'] == 'support' else lp['bounce_pct'])
    )

    if confirming_pct < 50.0 and not strategy_path:
        return {"has_setup": False, "reason": f"Validated level reaction is only {confirming_pct:.1f}%, below the 50% minimum. The engine is waiting for a stronger support/rejection or breakout/breakdown setup.", "context_audit": context_audit}

    factors_text = " ".join(lp.get('factors', [])).lower()

    # Momentum-exhaustion / reversal guard. A bullish rejection pattern into
    # support is dangerous for a fresh SELL; a bearish rejection into
    # resistance is dangerous for a fresh BUY. Do not fight a confirmed
    # reversal merely because the main score still points the old way.
    bullish_rejection = any(k in factors_text for k in (
        'bullish engulfing', 'bullish pin bar', 'hammer', 'bullish sweep', 'buyers actively defending'
    ))
    bearish_rejection = any(k in factors_text for k in (
        'bearish engulfing', 'bearish pin bar', 'shooting star', 'bearish sweep', 'sellers actively defending'
    ))
    if direction == 'SELL' and bullish_rejection and isinstance(sr_context, dict) and str(sr_context.get('location','')).upper() in ('AT_SUPPORT','NEAR_SUPPORT'):
        return {"has_setup": False, "reason": "Support rejection/exhaustion detected against SELL — old momentum is being rejected, so the engine stays flat until breakdown is confirmed.", "context_audit": context_audit}
    if direction == 'BUY' and bearish_rejection and isinstance(sr_context, dict) and str(sr_context.get('location','')).upper() in ('AT_RESISTANCE','NEAR_RESISTANCE'):
        return {"has_setup": False, "reason": "Resistance rejection/exhaustion detected against BUY — the engine will not enter after the move has already stalled.", "context_audit": context_audit}

    breadth_ok = (breadth_advances is not None and breadth_declines is not None
                  and (breadth_advances - breadth_declines) * (1 if direction == "BUY" else -1) > 0)
    global_ok = (global_avg_change is not None
                 and global_avg_change * (1 if direction == "BUY" else -1) > 0.1)
    banknifty_ok = not (banknifty_correlation_note and "DIVERGENCE WARNING" in banknifty_correlation_note.upper())
    low_vix = live_vix is not None and live_vix < 15.0
    news_up = (india_news_sentiment or "").upper()
    news_aligned = ("BULLISH" in news_up and direction == "BUY") or ("BEARISH" in news_up and direction == "SELL")
    sniper_up = (sniper_bias or "").upper()
    sniper_aligned = ("BULLISH" in sniper_up and direction == "BUY") or ("BEARISH" in sniper_up and direction == "SELL")
    # Max Pain: price tends to gravitate TOWARD max pain near expiry (a
    # "magnet" effect) -- so moving AWAY from it is the easier, less
    # resisted direction for a fresh move to continue in.
    max_pain_ok = (max_pain is not None and
                   ((direction == "BUY" and live_price > max_pain) or
                    (direction == "SELL" and live_price < max_pain)))

    order_pressure, order_imbalance = _extract_order_flow(dashboard_context)
    regime_bias = _extract_regime(dashboard_context)
    mtf_bias = _extract_mtf(dashboard_context)
    order_flow_ok = ((direction == "BUY" and order_pressure == "BUYING PRESSURE") or
                     (direction == "SELL" and order_pressure == "SELLING PRESSURE"))
    # A balanced book is neutral evidence, not a reason to reject an otherwise
    # valid setup. A strongly opposite book is handled as a penalty below.
    order_flow_conflict = ((direction == "BUY" and order_pressure == "SELLING PRESSURE") or
                           (direction == "SELL" and order_pressure == "BUYING PRESSURE"))
    regime_ok = ((direction == "BUY" and regime_bias == "BULLISH") or
                 (direction == "SELL" and regime_bias == "BEARISH"))
    regime_conflict = ((direction == "BUY" and regime_bias == "BEARISH") or
                       (direction == "SELL" and regime_bias == "BULLISH"))
    mtf_ok = ((direction == "BUY" and mtf_bias == "BULLISH") or
              (direction == "SELL" and mtf_bias == "BEARISH"))
    mtf_conflict = ((direction == "BUY" and mtf_bias == "BEARISH") or
                    (direction == "SELL" and mtf_bias == "BULLISH"))

    factor_flags = {
        "main_signal_aligned": True,  # gated on already above
        "ml_agrees": bool(ml_agrees),
        "level_pct_ge_65": confirming_pct >= 60.0,
        "banknifty_no_divergence": banknifty_ok,
        "breadth_aligned": bool(breadth_ok),
        "global_aligned": bool(global_ok),
        "oi_aligned": ("heavy put oi" in factors_text and direction == "BUY") or
                      ("heavy call oi" in factors_text and direction == "SELL"),
        "fvg_ob_confluence": "order block" in factors_text and "no order block" not in factors_text,
        "vwap_aligned": ("above a rising vwap" in factors_text and direction == "BUY") or
                         ("below a falling vwap" in factors_text and direction == "SELL"),
        "htf_1h_aligned": "1-hour" in factors_text and "bullish" in factors_text if direction == "BUY" \
                           else "1-hour" in factors_text and "bearish" in factors_text,
        "htf_15min_aligned": "15-minute" in factors_text and "bullish" in factors_text if direction == "BUY" \
                              else "15-minute" in factors_text and "bearish" in factors_text,
        "round_number_level": "round number" in factors_text,
        "liquidity_sweep": "liquidity sweep already detected" in factors_text,
        "low_vix": low_vix,
        "news_sentiment_aligned": news_aligned,
        "sniper_setup_aligned": sniper_aligned,
        "away_from_max_pain": bool(max_pain_ok),
        "order_flow_aligned": bool(order_flow_ok),
        "regime_aligned": bool(regime_ok),
        "mtf_stack_aligned": bool(mtf_ok),
        "data_quality_pass": bool(context_audit.get("complete") and not ("STALE" in freshness or "FROZEN" in freshness)),
        "no_opposite_transition": not ((_bearish_transition and direction == "BUY") or (_bullish_transition and direction == "SELL")),
        "global_research_aligned": bool(_gr_aligned),
        "nifty50_news_aligned": (str(nifty50_news_sentiment or "").upper() == ("BULLISH" if direction == "BUY" else "BEARISH")),
        "nifty50_fundamentals_aligned": (str(nifty50_fundamentals_bias or "").upper() in ("NEUTRAL", "BULLISH" if direction == "BUY" else "BEARISH")),
        "entry_timing_safe": not bool((sr_context.get("entry_timing") or {}).get("buy_adverse_move_risk" if direction == "BUY" else "sell_adverse_move_risk")),
    }

    # Ladder confluence is a secondary confirmation; calculate it from the
    # same candidate set used for target selection rather than adding it late.
    if level_ladder:
        same_side_levels = level_ladder.get("resistances" if direction == "BUY" else "supports") or []
        agreeing_preview = 0
        for lvl in same_side_levels[:3]:
            lvl_bias = str(lvl.get("directional_bias", "")).lower()
            if ("bullish" in lvl_bias and direction == "BUY") or ("bearish" in lvl_bias and direction == "SELL"):
                agreeing_preview += 1
        factor_flags["ladder_confluence_aligned"] = agreeing_preview >= 2
    else:
        factor_flags["ladder_confluence_aligned"] = False

    # Learn strong structural anchors separately: previous-day H/L/Close and
    # Opening Range are stronger locations, but never automatic entries.
    _zone = {}
    if isinstance(sr_context, dict):
        _zone = (sr_context.get("major_support") if direction == "BUY" else sr_context.get("major_resistance")) or {}
    _src_upper = " ".join(str(x) for x in (_zone.get("sources") or [])).upper()
    factor_flags["strong_daily_level"] = any(k in _src_upper for k in ("PDH", "PDL", "PDC", "PREV DAY", "PREVIOUS DAY"))
    factor_flags["strong_opening_range_level"] = "OPENING RANGE" in _src_upper
    factor_flags["immediate_reversal_risk_low"] = bool(factor_flags.get("entry_timing_safe") and factor_flags.get("no_opposite_transition"))

    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    # Do not require every secondary factor.  A few direct confirmations
    # should be able to trigger a trade, while major contradictions still
    # block it. This increases frequency modestly without turning the engine
    # into an always-trading system.
    critical_true = sum(bool(factor_flags.get(k)) for k in (
        "main_signal_aligned", "level_pct_ge_65", "htf_1h_aligned",
        "htf_15min_aligned", "vwap_aligned", "order_flow_aligned",
        "regime_aligned", "mtf_stack_aligned", "oi_aligned", "liquidity_sweep", "data_quality_pass",
        "strong_daily_level", "strong_opening_range_level", "immediate_reversal_risk_low"
    ))
    hard_conflicts = int(order_flow_conflict) + int(regime_conflict) + int(mtf_conflict)
    unknown_major = int(regime_bias == "UNKNOWN") + int(mtf_bias == "UNKNOWN") + int(order_pressure == "")
    _unknown_limit = 3 if strategy_path else 2   # strategy-led: a missing module is unknown, not a veto
    _timing_block = (not factor_flags.get('entry_timing_safe') and not _early_entry_ok and not strategy_path)
    if critical_true < 3 or hard_conflicts >= 2 or unknown_major >= _unknown_limit or not factor_flags.get('no_opposite_transition') or _timing_block or (reversal_used and ((direction == "SELL" and _bearish_reversal_score < 5) or (direction == "BUY" and _bullish_reversal_score < 5))):
        return {"has_setup": False,
                "reason": f"Full-context confluence is not strong enough ({critical_true} direct confirmations; {hard_conflicts} major conflicts). Waiting for a cleaner setup.",
                "context_audit": context_audit}

    strike = _round_to_strike(live_price)
    option_type = "CE" if direction == "BUY" else "PE"
    underlying_entry = live_price
    # SL: base 20 pts (user's fixed stop). In faster candles a flat 20 pts got stopped by normal noise
    # (4 of 5 recent setups), so it widens slightly with ATR -- hard-capped at 28 pts so trades stay frequent.
    try:
        sl_distance = min(28.0, max(20.0, round(0.9 * float(atr), 1))) if atr else 20.0
    except Exception:
        sl_distance = 20.0
    # STRUCTURE-AWARE STOP (a parameter, not a gate): a stop that sits INSIDE the support/resistance zone it is
    # defending gets hit by the normal wick through the zone.  If the validated zone's far edge is beyond the
    # base stop, move the stop to just past that edge (+0.25 ATR).  Capped so R:R and risk stay sane.
    try:
        if isinstance(sr_context, dict) and atr:
            _z = (sr_context.get("major_support") if direction == "BUY" else sr_context.get("major_resistance")) or {}
            _edge = _z.get("low") if direction == "BUY" else _z.get("high")
            if _edge is not None:
                _d_edge = (live_price - float(_edge)) if direction == "BUY" else (float(_edge) - live_price)
                if _d_edge > 0:
                    _struct = round(_d_edge + 0.25 * float(atr), 1)
                    if _struct > sl_distance:
                        sl_distance = min(MAX_SL_PTS, _struct)
    except Exception:
        logger.exception("structure-aware SL failed; keeping ATR stop")

    # Minimum acceptable reward:risk ratio -- target must be at least this
    # many times FARTHER than the stop-loss, or the setup isn't worth
    # taking even if confidence looks fine. Fixes the bug where the
    # nearest ladder level sometimes landed just 3-4 points away while
    # the stop-loss was 25+ points -- a terrible risk:reward that no
    # confidence number should paper over.
    MIN_RR_RATIO = 1.5

    # -----------------------------------------------------------------
    # NEW — use the FULL round-number Ladder Calculator (every 50pt level,
    # both directions), not just the single nearest level:
    #   1. Extra confluence factor: do at least 2 of the next 3 ladder
    #      levels in this SAME direction also read favorably? (i.e. is
    #      this a run of aligned levels, not a one-off single reading)
    #   2. Smarter target: walk OUTWARD through the ladder and take the
    #      FIRST level that gives at least MIN_RR_RATIO reward vs the
    #      stop-loss -- an actual real support/resistance rung, not an
    #      arbitrary distance, AND never a too-close, bad-R:R target.
    # -----------------------------------------------------------------
    ladder_confluence_aligned = False
    target = None

    # Prefer a real opposing option-OI wall when available; the user wants the
    # target near the next meaningful OI barrier. Structural/ladder levels remain
    # fallback references when the live option chain has no usable wall.
    _oi_raw = raw_option_chain if raw_option_chain is not None else dashboard_context.get("RAW Option Chain (all loaded strikes)")
    _oi_hit = _oi_wall_target(_oi_raw, live_price, direction, atr=atr)
    oi_wall_note = None
    oi_wall_penalty = 0.0
    oi_target = None
    if _oi_hit is not None:
        _oi_t, _wall_strike = _oi_hit
        oi_dist = (_oi_t - live_price) if direction == "BUY" else (live_price - _oi_t)
        if oi_dist >= OI_MIN_RR * sl_distance:
            oi_target = _oi_t
            target = oi_target
            oi_wall_note = f"Target {_oi_t:g} sits just before the {_wall_strike:g} OI wall ({oi_dist:.0f} pts away)."
        else:
            # a strong wall is sitting right in front of the entry: price is likely to stall there.  Do not invent a
            # farther target through it; keep the normal target but charge the trade some probability (soft).
            oi_wall_penalty = 0.25
            oi_wall_note = f"Strong OI wall at {_wall_strike:g} only {abs(_wall_strike - live_price):.0f} pts ahead -- probability reduced."

    # Prefer the next VALIDATED structural boundary as target when no usable OI
    # wall was found. Round-number ladder levels are secondary.
    if isinstance(sr_context, dict) and target is None:
        structural = sr_context.get('major_resistance') if direction == 'BUY' else sr_context.get('major_support')
        if isinstance(structural, dict):
            structural_target = structural.get('low') if direction == 'BUY' else structural.get('high')
            try:
                structural_target = float(structural_target)
                structural_distance = ((structural_target - live_price) if direction == 'BUY'
                                       else (live_price - structural_target))
                if structural_distance >= MIN_RR_RATIO * sl_distance:
                    target = round(structural_target, 2)
            except Exception:
                logger.exception("Broad exception caught; fallback path executed")
                target = None

    if level_ladder and target is None:
        same_side_levels = level_ladder.get('resistances' if direction == "BUY" else 'supports') or []
        agreeing = 0
        for lvl in same_side_levels[:3]:
            lvl_bias = lvl.get('directional_bias', '').lower()
            if ("bullish" in lvl_bias and direction == "BUY") or ("bearish" in lvl_bias and direction == "SELL"):
                agreeing += 1
        ladder_confluence_aligned = agreeing >= 2

        for lvl in same_side_levels:
            if lvl['distance_pts'] >= MIN_RR_RATIO * sl_distance:
                target = lvl['level_price']
                break
        # (if no ladder level is far enough out, target stays None and
        # falls through to the guaranteed-good-R:R ATR fallback below)

    factor_flags["ladder_confluence_aligned"] = ladder_confluence_aligned
    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    if target is None:
        # A valid entry should not be discarded merely because the next
        # structural rung is too close. Use a deterministic ATR target that
        # still respects the minimum reward:risk requirement.
        target = round(underlying_entry + MIN_RR_RATIO * sl_distance, 2) if direction == 'BUY' else round(underlying_entry - MIN_RR_RATIO * sl_distance, 2)
    if direction == "BUY":
        stop_loss = round(underlying_entry - sl_distance, 2)
    else:
        stop_loss = round(underlying_entry + sl_distance, 2)
    # A wall 140 pts away with a 20 pt stop is a "7R" target that almost never fills inside the
    # 2-hour holding window and silently inflates the reward:risk.  Cap it at a realistic distance.
    try:
        # late in the session there is less time before the 15:20 auto-close: scale the realistic target reach to the minutes left
        try:
            _eod = datetime.combine(_now.date(), trade_learning.EOD_SQUAREOFF)
            _mins_left = (_eod - _now.replace(tzinfo=None)).total_seconds() / 60.0
        except Exception:
            _mins_left = float(trade_learning.MAX_HOLD_MINUTES)
        _hold_min = max(10.0, min(float(trade_learning.MAX_HOLD_MINUTES), _mins_left))
        _bars = max(1.0, _hold_min / 5.0)
        _reach = TARGET_REACH_ATR * float(atr) * math.sqrt(_bars) if atr else None
    except Exception:
        _reach = None
    # A target farther than price can plausibly travel in the holding window almost never fills (an 84 pt target on a
    # 28 pt stop was ~a full 2-hour range).  Cap at the lower of MAX_TARGET_R and the ATR-based reach, but never below 1.5R.
    _max_tgt = MAX_TARGET_R * sl_distance
    if _reach:
        _max_tgt = min(_max_tgt, max(_reach, 1.5 * sl_distance))
    if abs(target - underlying_entry) > _max_tgt:
        target = round(underlying_entry + _max_tgt, 2) if direction == "BUY" else round(underlying_entry - _max_tgt, 2)

    # Entry-quality gate: even a correct directional read is not a good
    # entry if price is already too far from the structural level. This is
    # the explicit "right place" filter: react at the zone or enter close
    # to a confirmed breakout, never chase a move several ATRs away.
    level_distance = abs(float(live_price) - float(lp.get("level_price", live_price)))
    # Reaction entries must stay very close to the structural zone; breakout
    # entries get a little more room because the retest itself is above/below
    # the original boundary. Never chase a move that has already travelled far.
    loc_for_distance = str(sr_context.get('location', '')).upper() if isinstance(sr_context, dict) else ''
    breakout_entry = (direction == 'BUY' and loc_for_distance in ('AT_RESISTANCE', 'NEAR_RESISTANCE')) or \
                     (direction == 'SELL' and loc_for_distance in ('AT_SUPPORT', 'NEAR_SUPPORT'))
    max_entry_distance = max((0.80 if not breakout_entry else 0.90) * float(atr), 8.0 if not breakout_entry else 10.0)
    if level_distance > max_entry_distance:
        return {"has_setup": False, "reason": f"Entry is too far from the validated level ({level_distance:.1f} pts > {max_entry_distance:.1f} pts). Setup may be directionally correct but the location is poor; waiting for a better entry.", "context_audit": context_audit}

    # ------------------------------------------------------------------
    # v36 CALIBRATED EDGE.  The old score was `50 + 1.8 x factors` and the entry floor was 50, so it
    # filtered almost nothing.  Now: calibrated win probability -> expectancy in R -> trade only if
    # the expectancy clears a bar that reflects the engine's REAL recent form.
    # ------------------------------------------------------------------
    rr = abs(target - underlying_entry) / max(abs(underlying_entry - stop_loss), 1e-9)
    if strategy_path:
        factor_flags["strategy_engine_aligned"] = True
        factor_flags["strategy_edge"] = float(strategy_result.get("edge", 0) or 0) >= 7.0
        for _name in (strategy_result.get("selected_strategies") or []):
            factor_flags[f"strategy::{_name}"] = True
    else:
        factor_flags["strategy_engine_aligned"] = False
        factor_flags["strategy_edge"] = False
    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    _sscore = float(strategy_result.get("score", 0) or 0) if strategy_path else 0.0
    _penalty = 0.20 * hard_conflicts                       # each major contradiction costs real probability
    _penalty += min(0.40, 0.10 * len(soft_flags))          # S/R-location caveats
    _penalty += 0.15 if is_choppy else 0.0                 # squeezed/choppy volatility: more false breaks
    _penalty += 0.10 if (not regime_ok and regime_bias == "RANGE") else 0.0
    _penalty += oi_wall_penalty
    # ENTRY QUALITY (price-action): chasing an extended move costs probability; a sweep / rejection / pullback earns some back.
    _eq = (entry_quality or {}).get(direction) if isinstance(entry_quality, dict) and direction in entry_quality else entry_quality
    _eq = _eq if isinstance(_eq, dict) and ("penalty_logit" in _eq) else None
    if _eq:
        _penalty += float(_eq.get("penalty_logit", 0.0))
        factor_flags["entry_not_extended"] = not _eq.get("chase_flags")
        factor_flags["entry_confirmed_bounce"] = bool(_eq.get("confirm_flags"))
    else:
        factor_flags["entry_not_extended"] = True
        factor_flags["entry_confirmed_bounce"] = False
    # PCR velocity: fast-moving PCR in our direction = writers are positioned with us; against us = cost a little
    _pv = pcr_velocity if isinstance(pcr_velocity, dict) else {}
    _pvb = str(_pv.get("bias", "NEUTRAL")).upper()
    factor_flags["pcr_velocity_aligned"] = (_pvb == direction)
    if _pvb in ("BUY", "SELL") and _pvb != direction:
        _penalty += 0.10
    # OI BUILDUP: is fresh writer money behind this move, or is the move being absorbed?  (option-chain OI change vs yesterday)
    _eqm = (_eq or {}).get("metrics", {}) if _eq else {}
    _dm = _eqm.get("day_move_pct")
    _signed_move = (float(_dm) * (1.0 if direction == "BUY" else -1.0)) if _dm is not None else None
    _oib = oi_buildup.assess(raw_option_chain, live_price, direction, _signed_move)
    _penalty += float(_oib.get("penalty_logit", 0.0))
    # EVENT CAUTION (scheduled news earlier today / yesterday night)
    if _event.get("level") == "CAUTION":
        _penalty += float(_event.get("penalty_logit", 0.0))
    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)
    if strategy_path:                                      # a strong named-strategy score is evidence, a weak one is not
        _penalty -= max(-0.15, min(0.30, (_sscore - 50.0) / 50.0 * 0.30))
    # LEARNED ADJUSTMENT: mistake memory ("this looks like setups that already lost") + the self-validated tiny neural net.
    _feats = {"rsi": (_eq or {}).get("metrics", {}).get("rsi"), "ema20_dist_atr": (_eq or {}).get("metrics", {}).get("ema20_dist_atr"),
              "last_candle_atr": (_eq or {}).get("metrics", {}).get("last_candle_atr"), "run_6c_atr": (_eq or {}).get("metrics", {}).get("run_6c_atr"),
              "streak": (_eq or {}).get("metrics", {}).get("streak"), "atr": float(atr) if atr else None, "rr": round(rr, 3),
              "hour": _now.hour + _now.minute / 60.0, "vix": live_vix, "pcr_per15": _pv.get("per15"),
              "strategy_score": _sscore if strategy_path else None, "is_buy": 1.0 if direction == "BUY" else 0.0}
    _feats.update({"day_move_pct": _eqm.get("day_move_pct"), "day_pos": _eqm.get("day_pos"),
                   "oi_flow": float(_oib.get("flow", 0)), "expiry_day": 1.0 if _event.get("expiry_day") else 0.0,
                   "event_caution": 1.0 if _event.get("level") == "CAUTION" else 0.0})
    _feats = {k: (round(float(v), 3) if isinstance(v, (int, float)) and v == v else None) for k, v in _feats.items()}
    try:
        _adj = trade_learning.learned_adjustment(factor_flags, _feats)
    except Exception:
        logger.exception("learned adjustment failed (ignored)")
        _adj = {"delta_logit": 0.0, "note": "", "nn": {}, "memory": {}}
    _penalty -= float(_adj.get("delta_logit", 0.0))
    # REPEAT-MISTAKE CHECK: tags (extended entry, late session, huge day move ...) that have already lost often -> small penalty
    _tags = trade_diagnostics.risk_tags(factor_flags, _feats, hour=_now.hour + _now.minute / 60.0)
    try:
        _tag_pen, _tag_note = trade_diagnostics.tag_penalty(_tags, trade_learning._labelled_rows_full())
    except Exception:
        logger.exception("tag penalty failed (ignored)")
        _tag_pen, _tag_note = 0.0, ""
    _penalty += _tag_pen
    edge = trade_learning.compute_edge(factor_flags, rr, _penalty)
    edge["learning_note"] = _adj.get("note", "")
    confidence_pct = edge["confidence_pct"]
    used_learning = edge["learned_count"] > 0 or edge["calibrated_on_trades"] > 0
    learned_count = edge["learned_count"]

    # bar the expectancy must clear
    _min_ev = MIN_EV_R
    _form = trade_learning.recent_performance()
    _bar_notes = []
    if _form.get("bad"):
        _min_ev += BAD_FORM_EV_EXTRA
        _bar_notes.append(f"recent form {_form.get('wins')}/{_form.get('n')} -> bar +{BAD_FORM_EV_EXTRA:.2f}R")
    if _is_drought(_now) and float(edge.get("win_probability") or 0.0) >= DROUGHT_RELIEF_MIN_PROB:
        _min_ev = max(0.0, _min_ev - DROUGHT_EV_RELIEF)
        _bar_notes.append(f"dry spell -> bar -{DROUGHT_EV_RELIEF:.2f}R")
    try:
        _cost_r = ROUND_TRIP_COST_PTS / max(abs(float(underlying_entry) - float(stop_loss)), 1e-9)
        _min_ev += _cost_r
        _bar_notes.append(f"trading costs ~{ROUND_TRIP_COST_PTS:.0f} pts -> bar +{_cost_r:.2f}R")
    except Exception:
        pass
    edge["required_ev_r"] = round(_min_ev, 3)
    edge["bar_notes"] = _bar_notes
    edge["required_probability"] = round(edge_model.required_probability(rr, _min_ev), 4)

    if edge["expectancy_r"] < _min_ev:
        return {"has_setup": False,
                "reason": (f"No positive edge: estimated win chance {confidence_pct:.1f}% at R:R {rr:.2f} gives expectancy "
                           f"{edge['expectancy_r']:+.2f}R, below the required {_min_ev:+.2f}R "
                           f"(break-even needs {100*edge['breakeven_probability']:.0f}%). Waiting for a better setup."
                           + (f" [{'; '.join(_bar_notes)}]" if _bar_notes else "")),
                "context_audit": context_audit, "expectancy_r": edge["expectancy_r"], "confidence_pct": confidence_pct,
                "factors_true": factors_true, "factors_total": factors_total, "factor_flags": factor_flags,
                "learning_note": edge.get("learning_note", ""),
                "candidate": {"direction": direction, "underlying_entry": underlying_entry, "stop_loss": stop_loss, "target": target,
                              "risk_reward": round(rr, 2), "factor_flags": dict(factor_flags, _features=_feats)}}

    _wp = float(edge.get("win_probability") or 0.0)
    _floor, _floor_note = _win_floor(_now)
    if _wp < _floor:
        return {"has_setup": False,
                "reason": (f"Win-chance floor: the engine's own win probability for this setup is {100 * _wp:.1f}%, below the minimum "
                           f"{100 * _floor:.0f}%{(' (' + _floor_note + ')') if _floor_note else ''}. Even with a positive-looking R:R it would lose about {100 * (1 - _wp):.0f} of 100 times. "
                           "Waiting for a better setup (this setup is still tracked as a shadow trade to learn from)."),
                "context_audit": context_audit, "expectancy_r": edge["expectancy_r"], "confidence_pct": confidence_pct,
                "factors_true": factors_true, "factors_total": factors_total, "factor_flags": factor_flags,
                "learning_note": edge.get("learning_note", ""),
                "candidate": {"direction": direction, "underlying_entry": underlying_entry, "stop_loss": stop_loss, "target": target,
                              "risk_reward": round(rr, 2), "factor_flags": dict(factor_flags, _features=dict(_feats, win_prob=round(_wp, 3)))}}

    # Self-learning segment gate: pause a playbook / grade / hour / side that has PROVEN to lose
    # (this existed in trade_learning.py but was never called, and no trade ever stored the meta it needs).
    _playbook = (strategy_result.get("selected_strategies") or ["none"])[0] if strategy_path else "ai_only"
    _grade = "A" if (critical_true >= 7 and hard_conflicts == 0) else ("B" if critical_true >= 5 else "C")
    _entry_meta = {"playbook": str(_playbook), "grade": _grade, "hour_bucket": f"{_now.hour:02d}h",
                   "side": direction, "probe": False, "logic_version": trade_learning.LOGIC_VERSION}
    _seg_blocked, _seg_reason = trade_learning.segment_gate(_entry_meta)
    if _seg_blocked:
        return {"has_setup": False, "reason": _seg_reason, "context_audit": context_audit,
                "factors_true": factors_true, "factors_total": factors_total,
                "candidate": {"direction": direction, "underlying_entry": underlying_entry, "stop_loss": stop_loss, "target": target,
                              "risk_reward": round(rr, 2), "factor_flags": dict(factor_flags, _features=_feats)}}
    _n = edge["sample_size"]
    if edge["calibrated_on_trades"]:
        confidence_note = (f"Calibrated win probability (adjusted to the engine's real record: {_n} resolved trades, "
                           f"{edge['empirical_win_rate']}% win rate; {learned_count} learned factor(s)). "
                           f"Expectancy {edge['expectancy_r']:+.2f}R at R:R {rr:.2f}.")
    elif _n > 0:
        confidence_note = (f"Mostly rule-based prior ({_n} resolved trades so far, need {edge_model.MIN_TRADES_FOR_CALIBRATION} to calibrate). "
                           f"Expectancy {edge['expectancy_r']:+.2f}R at R:R {rr:.2f}. Gets more honest as trades resolve.")
    else:
        confidence_note = (f"Rule-based prior only -- no resolved trades yet, so this is a conservative estimate, not a measured win rate. "
                           f"Expectancy {edge['expectancy_r']:+.2f}R at R:R {rr:.2f}.")
    factor_flags["_entry_meta"] = _entry_meta
    factor_flags["_playbook_label"] = str(_playbook)
    factor_flags["_setup_score"] = round(_sscore, 1)
    factor_flags["_reasons"] = list(lp.get("factors", []))[:3]
    factor_flags["_edge"] = {"p": edge["win_probability"], "ev_r": edge["expectancy_r"], "rr": round(rr, 2)}
    _feats["win_prob"] = round(_wp, 3)
    factor_flags["_features"] = _feats
    factor_flags["_risk_tags"] = trade_diagnostics.risk_tags(factor_flags, _feats, hour=_now.hour + _now.minute / 60.0)
    factor_flags["_oi_buildup"] = _oib.get("label")
    if _tag_note:
        factor_flags["_repeat_mistake_note"] = _tag_note
    if _event.get("level") != "NONE":
        factor_flags["_event_note"] = _event.get("reason")
    if edge.get("learning_note"):
        factor_flags["_learning_note"] = edge["learning_note"]
    if oi_wall_note:
        factor_flags["_oi_wall_note"] = oi_wall_note

    return {
        "has_setup": True, "direction": direction, "strike": strike, "option_type": option_type,
        "underlying_entry": round(underlying_entry, 2), "stop_loss": stop_loss, "target": target,
        "confidence_pct": confidence_pct, "confidence_note": confidence_note,
        "factor_flags": factor_flags, "factors_true": factors_true, "factors_total": factors_total,
        "level_price": lp['level_price'], "level_pct": confirming_pct,
        "context_audit": context_audit,
        "score_model": "calibrated_edge_v36",
        "strategy_engine": strategy_result,
        "strategy_switch_note": _rebased_note,
        "strategy_path": strategy_path,
        "oi_target_reference": oi_target,
        "critical_confirmations": critical_true,
        "major_conflicts": hard_conflicts,
        "order_flow_pressure": order_pressure,
        "regime_bias": regime_bias,
        "mtf_bias": mtf_bias,
        "entry_location_quality": "STRUCTURAL_ZONE_CONFIRMED",
        "level_distance_pts": round(level_distance, 2),
        "risk_reward": round(abs(target - underlying_entry) / max(abs(underlying_entry - stop_loss), 1e-9), 2),
        "analysis_mode": "FULL_CONTEXT_BEFORE_ENTRY",
        "raw_direction": raw_direction,
        "reversal_used": reversal_used,
        "reversal_reason": reversal_reason,
        "reversal_confirmation_count": (_bearish_reversal_score if direction == "SELL" else _bullish_reversal_score),
        "early_entry_validated": bool(_early_entry_ok),
        "early_entry_checks": int(_early_checks),
        "soft_flags": soft_flags,
        "ai_research_source": ai_research_source,
        "target_note": oi_wall_note, "learning_note": edge.get("learning_note", ""),
        "entry_quality": _eq, "pcr_velocity": _pv or None,
        "oi_buildup": _oib, "event_risk": _event, "risk_tags": factor_flags.get("_risk_tags"), "repeat_mistake_note": _tag_note,
        "expectancy_r": edge["expectancy_r"], "win_probability": edge["win_probability"],
        "breakeven_probability": edge["breakeven_probability"], "required_ev_r": edge["required_ev_r"],
        "edge": edge,
    }
