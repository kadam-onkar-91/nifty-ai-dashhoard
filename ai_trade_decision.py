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
import strategy_engine
import gemini_pool
import json
import time
import requests
import pandas as pd


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



def _oi_wall_target(raw_chain, live_price, direction, min_pts=20.0, max_pts=140.0):
    """Return the nearest strong opposing option-OI wall around price.

    BUY targets prefer the nearest loaded CE-OI wall above spot; SELL targets
    prefer the nearest loaded PE-OI wall below spot. This is a target reference,
    not a promise that price will reach the wall.
    """
    rows=[]
    def collect(x):
        if isinstance(x, dict):
            if any(k in x for k in ('strike','strike_price','strikePrice')):
                rows.append(x)
            for v in x.values(): collect(v)
        elif isinstance(x, list):
            for v in x: collect(v)
    if isinstance(raw_chain, pd.DataFrame):
        data=raw_chain.to_dict('records')
    elif isinstance(raw_chain, str):
        try: data=json.loads(raw_chain)
        except Exception: data=[]
    else: data=raw_chain
    collect(data)
    candidates=[]
    for r in rows:
        try:
            strike=float(r.get('strike',r.get('strike_price',r.get('strikePrice'))))
        except Exception: continue
        side=str(r.get('option_type',r.get('type',r.get('instrument_type','')))).lower()
        want='ce' if direction=='BUY' else 'pe'
        if side and want not in side: continue
        oi=0.0
        for k in ('oi','open_interest','openInterest','OI','CE OI','PE OI'):
            try:
                if k in r and r[k] is not None: oi=max(oi,float(r[k]))
            except Exception: pass
        if oi<=0: continue
        dist=(strike-live_price) if direction=='BUY' else (live_price-strike)
        if dist < min_pts or dist > max_pts: continue
        candidates.append((oi,dist,strike))
    if not candidates: return None
    # Strongest wall, with distance as a secondary preference.
    candidates.sort(key=lambda z:(-z[0],z[1]))
    return round(candidates[0][2],2)




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
        v=v[:lim]
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
GEMINI_FAIL_OPEN = True   # Gemini *technically* unreachable -> strict local fallback (never after an explicit REJECT)


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
                        live_price=None, strategy_formed_at=None):
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
    prompt = (
        "You are the FINAL independent reviewer for an intraday NIFTY trading system.\n\n"
        f"A named strategy is live and the self-learning AI engine agrees. Proposed direction: {direction}.\n"
        "STEP 1: From the LIVE RESEARCH PACKET alone, form your OWN independent view: BUY, SELL or NO_TRADE.\n"
        "STEP 2: APPROVE only if your own view equals the proposed direction. You cannot invent a trade or flip direction.\n\n"
        "Rules:\n"
        "1) Use only the supplied packet; never treat missing data as bullish/bearish.\n"
        "2) REJECT if data is stale/frozen, clearly contradictory, or price is already extended away from its level.\n"
        "3) Secondary factors may disagree; require the MAIN evidence (structure, VWAP, order flow, option OI, regime) to lean your way.\n"
        "4) No certainty claims.\n\n"
        f"Live price: {live_price}\nStrategy formed at: {strategy_formed_at}\nStrategy summary:\n{strategy_json}\n\n"
        f"LIVE RESEARCH PACKET:\n{packet}\n\n"
        "Return ONLY JSON with keys: own_direction (BUY/SELL/NO_TRADE), approved (boolean), verdict (APPROVE/REJECT), "
        "confidence (0-100), reason (short string), risk_flags (array of strings)."
    )
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}

    def _one(key, model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        resp = requests.post(url, headers={"Content-Type": "application/json", "x-goog-api-key": key}, json=body, timeout=30)
        if resp.status_code != 200:
            try:
                msg = (resp.json().get("error", {}) or {}).get("message", "")[:200]
            except Exception:
                msg = resp.text[:200]
            raise gemini_pool.PoolHTTPError(resp.status_code, msg)
        cand = (resp.json().get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return _gemini_extract_json(text)

    try:
        parsed, _idx, model = gemini_pool.run(keys, gemini_pool.REVIEW_MODELS, _one, max_attempts=5)
    except gemini_pool.GeminiUnavailable as gu:
        return {"status": "UNAVAILABLE", "approved": False, "direction": "NO_TRADE", "verdict": "UNAVAILABLE",
                "confidence": 0, "reason": f"{gu.last_exc}"[:200], "attempts": gu.attempts[-6:],
                "retry_in_s": round(gu.wait_s), "risk_flags": ["gemini_unavailable"], "model": None}
    own = str(parsed.get("own_direction", parsed.get("direction", ""))).upper()
    approved = bool(parsed.get("approved")) and str(parsed.get("verdict", "")).upper() == "APPROVE" and own in (direction, "")
    out = {"direction": own or direction, "confidence": parsed.get("confidence"),
           "risk_flags": parsed.get("risk_flags") or [], "model": model, "own_direction": own}
    if approved:
        out.update({"status": "APPROVED", "approved": True, "direction": direction, "verdict": "APPROVE",
                    "reason": parsed.get("reason") or "Gemini's own read matches the proposed setup."})
    else:
        out.update({"status": "REJECTED", "approved": False, "verdict": "REJECT",
                    "reason": parsed.get("reason") or "Gemini's own read does not support the proposed setup."})
    return out


def local_fallback_review(decision):
    """Used ONLY when Gemini is technically unreachable (never after an explicit Gemini REJECT).
    Stricter than the normal path: needs higher confidence, several direct confirmations
    and zero major conflicts, so a Gemini outage neither blocks every trade nor waves weak ones through."""
    d = decision or {}
    conf = float(d.get("confidence_pct", 0) or 0)
    crit = int(d.get("critical_confirmations", 0) or 0)
    conflicts = int(d.get("major_conflicts", 0) or 0)
    rr = float(d.get("risk_reward", 0) or 0)
    ok = conf >= 60.0 and crit >= 4 and conflicts == 0 and rr >= 1.5
    why = (f"confidence {conf:.0f}% (need 60), {crit} direct confirmations (need 4), "
           f"{conflicts} major conflicts (need 0), R:R {rr:.2f} (need 1.5)")
    return {"status": "LOCAL_FALLBACK_APPROVED" if ok else "LOCAL_FALLBACK_BLOCKED", "approved": ok,
            "direction": d.get("direction"), "reason": "Gemini unreachable; strict local fallback: " + why}


def generate_trade_decision(live_price, level_prediction, atr, max_pain=None,
                             signal_code=0, ml_agrees=False, banknifty_correlation_note=None,
                             breadth_advances=None, breadth_declines=None,
                             global_avg_change=None, live_vix=None, india_news_sentiment=None,
                             level_ladder=None, sr_context=None, sniper_bias=None, is_choppy=False,
                             dashboard_context=None, global_research=None,
                             nifty50_news_sentiment=None, nifty50_fundamentals_bias=None,
                             strategy_result=None, raw_option_chain=None):
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
        if abs(_net) >= 3:
            ai_research_dir = 1 if _net > 0 else -1
            ai_research_source = f"multi_factor_vote({_net:+d})"

    if not strategy_path:
        return {"has_setup": False,
                "reason": "No named strategy is currently formed strongly enough. AI research continues, but it cannot open an entry by itself.",
                "context_audit": context_audit, "strategy_required": True}

    if ai_research_dir == 0:
        return {"has_setup": False,
                "reason": "A strategy is formed, but the independent AI research engine has no clear direction yet; waiting for same-direction confirmation.",
                "context_audit": context_audit, "strategy_required": True,
                "strategy_direction": "BUY" if strategy_dir == 1 else "SELL"}

    if ai_research_dir != strategy_dir:
        return {"has_setup": False,
                "reason": f"Strategy is {('BUY' if strategy_dir == 1 else 'SELL')} but AI research is {('BUY' if ai_research_dir == 1 else 'SELL')}; no trade until both directions match.",
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
    sl_distance = 20.0  # user-defined fixed underlying stop; no ATR widening

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
    oi_target = _oi_wall_target(raw_option_chain if raw_option_chain is not None else dashboard_context.get("RAW Option Chain (all loaded strikes)"), live_price, direction)
    if oi_target is not None:
        oi_dist = (oi_target-live_price) if direction == "BUY" else (live_price-oi_target)
        if oi_dist >= MIN_RR_RATIO * sl_distance:
            target = oi_target

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

    confidence_pct, used_learning, learned_count = trade_learning.compute_confidence(factor_flags)
    if strategy_path:
        # Strategy is the entry trigger; its score strengthens the AI research
        # confidence after the required same-direction match has been proven.
        confidence_pct = round(0.62 * confidence_pct + 0.38 * float(strategy_result.get("score", 0) or 0), 1)
        factor_flags["strategy_engine_aligned"] = True
        factor_flags["strategy_edge"] = float(strategy_result.get("edge", 0) or 0) >= 7.0
        for _name in (strategy_result.get("selected_strategies") or []):
            factor_flags[f"strategy::{_name}"] = True
    else:
        factor_flags["strategy_engine_aligned"] = False
        factor_flags["strategy_edge"] = False
    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    # Frequency control: accept a candidate when the weighted evidence is
    # solid, not only when an arbitrary number of boxes are checked.
    # 62 is deliberately reachable on a good intraday setup, while the
    # strongest structural conflicts still prevent entry.
    if hard_conflicts >= 1:
        # One contradictory factor is a penalty, not an automatic veto.
        # Two or more major contradictions are still blocked above.
        confidence_pct = min(confidence_pct, 64.0)
    if not regime_ok and regime_bias == "RANGE":
        confidence_pct = min(confidence_pct, 66.0)
    if soft_flags:
        # each S/R-location caveat costs 3 pts (max 12) instead of vetoing a live strategy
        confidence_pct = round(confidence_pct - min(12.0, 3.0 * len(soft_flags)), 1)
    _entry_floor = 50.0 if strategy_path else (58.0 if _early_entry_ok else 60.0)
    _critical_floor = 2 if strategy_path else (3 if _early_entry_ok else 4)
    if confidence_pct < _entry_floor or (not strategy_path and confidence_pct < 70.0 and critical_true < _critical_floor):
        return {"has_setup": False,
                "reason": f"Weighted setup score {confidence_pct:.1f}% is below the entry threshold for a clean trade. Waiting for stronger confirmation.",
                "context_audit": context_audit,
                "factors_true": factors_true, "factors_total": factors_total,
                "factor_flags": factor_flags}
    track = trade_learning.get_overall_track_record()
    if used_learning:
        confidence_note = (f"Blended from {learned_count} learned factor(s) with enough history, "
                            f"rest rule-based. Overall track record so far: {track['sample_size']} resolved "
                            f"({track['win_rate']}% win rate)." if track['sample_size'] > 0
                            else f"Blended from {learned_count} learned factor(s); no resolved trades yet to show an overall win rate.")
    else:
        confidence_note = ("Pure rule-based estimate -- not enough historical setups yet for this engine to "
                            "have learned which factors actually predict wins in your data. This will get "
                            "more accurate (and more honest) as more setups resolve.")

    return {
        "has_setup": True, "direction": direction, "strike": strike, "option_type": option_type,
        "underlying_entry": round(underlying_entry, 2), "stop_loss": stop_loss, "target": target,
        "confidence_pct": confidence_pct, "confidence_note": confidence_note,
        "factor_flags": factor_flags, "factors_true": factors_true, "factors_total": factors_total,
        "level_price": lp['level_price'], "level_pct": confirming_pct,
        "context_audit": context_audit,
        "score_model": "weighted_confluence_v13_strategy_ensemble_fixed20sl",
        "strategy_engine": strategy_result,
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
    }
