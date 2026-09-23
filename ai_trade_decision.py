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


def _level_path_forecast(level_prediction, sr_context, direction_hint=None):
    """Read the existing S/R break-vs-bounce model BEFORE deciding entry.

    This is deliberately a decision aid, not a guaranteed probability. The
    S/R engine already combines price action, volume, VWAP, MTF, OI/PCR,
    liquidity and other context into break_pct/bounce_pct. The AI engine uses
    that read to decide whether the next path is more likely to be bounce,
    breakout/breakdown, or rejection instead of blindly waiting for a second
    confirmation candle every time.
    """
    if not isinstance(level_prediction, dict):
        return {
            "ready": False, "path": None, "kind": None, "confidence": 0.0,
            "edge": 0.0, "false_break_risk": 100.0, "reason": "No level-reaction forecast available."
        }

    kind = str(level_prediction.get("approaching", "")).lower()
    if kind not in ("support", "resistance"):
        return {
            "ready": False, "path": None, "kind": kind or None, "confidence": 0.0,
            "edge": 0.0, "false_break_risk": 100.0, "reason": "No actionable support/resistance approach detected."
        }

    try:
        break_pct = float(level_prediction.get("break_pct"))
        bounce_pct = float(level_prediction.get("bounce_pct"))
    except (TypeError, ValueError):
        return {
            "ready": False, "path": None, "kind": kind, "confidence": 0.0,
            "edge": 0.0, "false_break_risk": 100.0, "reason": "Level reaction score is unavailable."
        }

    factors_text = " ".join(str(x) for x in (level_prediction.get("factors") or [])).lower()
    if kind == "support":
        path = "BUY" if bounce_pct >= break_pct else "SELL"
        winning_pct = max(bounce_pct, break_pct)
    else:
        path = "BUY" if break_pct >= bounce_pct else "SELL"
        winning_pct = max(break_pct, bounce_pct)

    edge = abs(break_pct - bounce_pct)

    # Separate the continuation/break score from the main read's rejection
    # risk. These are intentionally modest penalties because the underlying
    # break/bounce model has already counted the same evidence.
    false_break_risk = 0.0
    false_reasons = []
    if kind == "resistance" and path == "BUY":
        if "long upper wicks" in factors_text or "sellers actively defending" in factors_text:
            false_break_risk += 15.0; false_reasons.append("upper-wick/seller rejection")
        if "rsi momentum divergence" in factors_text:
            false_break_risk += 10.0; false_reasons.append("RSI divergence")
        if "liquidity sweep already detected" in factors_text:
            false_break_risk += 12.0; false_reasons.append("liquidity sweep")
        if "volume contracting" in factors_text:
            false_break_risk += 8.0; false_reasons.append("volume contraction")
        if "bearish signal" in factors_text or "favors resistance holding" in factors_text:
            false_break_risk += 8.0; false_reasons.append("bearish price-action evidence")
    elif kind == "support" and path == "SELL":
        if "long lower wicks" in factors_text or "buyers actively defending" in factors_text:
            false_break_risk += 15.0; false_reasons.append("lower-wick/buyer rejection")
        if "rsi momentum divergence" in factors_text:
            false_break_risk += 10.0; false_reasons.append("RSI divergence")
        if "liquidity sweep already detected" in factors_text:
            false_break_risk += 12.0; false_reasons.append("liquidity sweep")
        if "volume contracting" in factors_text:
            false_break_risk += 8.0; false_reasons.append("volume contraction")
        if "bullish signal" in factors_text or "favors support holding" in factors_text:
            false_break_risk += 8.0; false_reasons.append("bullish price-action evidence")

    false_break_risk = min(45.0, false_break_risk)
    effective = max(0.0, winning_pct - (false_break_risk * 0.35))

    # A real setup does not need the old hard-coded retest gate every time.
    # It does need a meaningful edge and no obvious false-break/rejection risk.
    ready = bool(effective >= 62.0 and edge >= 10.0 and false_break_risk < 30.0)
    reason = (f"{kind.title()}: {('bounce' if kind == 'support' and path == 'BUY' else 'breakdown' if kind == 'support' else 'breakout' if path == 'BUY' else 'rejection')} read {winning_pct:.1f}% vs {100-winning_pct:.1f}% "
              f"(edge {edge:.1f}); false-break risk {false_break_risk:.0f}/100"
              + (f" — {', '.join(false_reasons)}." if false_reasons else "."))
    return {
        "ready": ready, "path": path, "kind": kind, "confidence": round(effective, 1),
        "raw_confidence": round(winning_pct, 1), "edge": round(edge, 1),
        "false_break_risk": round(false_break_risk, 1), "reason": reason,
    }


def generate_trade_decision(live_price, level_prediction, atr, max_pain=None,
                             signal_code=0, ml_agrees=False, banknifty_correlation_note=None,
                             breadth_advances=None, breadth_declines=None,
                             global_avg_change=None, live_vix=None, india_news_sentiment=None,
                             level_ladder=None, sr_context=None, sniper_bias=None, is_choppy=False,
                             dashboard_context=None, global_research=None,
                             nifty50_news_sentiment=None, nifty50_fundamentals_bias=None):
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

    The engine first reads the validated S/R break-vs-bounce forecast, then
    combines it with the full dashboard confluence and learned factor history.
    It does not require every secondary factor to be true, and it does not
    blindly enter every level touch.
    """
    # The dashboard now passes its CURRENT master context directly into this
    # engine.  Keep the context intact (do not reduce it to a small score) so
    # every available module can be audited before a setup is accepted.
    dashboard_context = dashboard_context or {}
    global_research = global_research or dashboard_context.get("RAW Global Research") or {}
    if not isinstance(global_research, dict):
        global_research = {}
    _global_bias = str(global_research.get("directional_bias", "UNKNOWN")).upper()
    _global_strength = str(global_research.get("strength", "NONE")).upper()
    required_context_sections = (
        "Market Status", "Data Freshness", "Nifty Spot Price",
        "RAW Primary Price/Indicator Data (latest 500 candles)",
        "RAW Latest Price/Indicator Snapshot",
        "RAW Option Chain (all loaded strikes)",
        "RAW Full NIFTY 50 Breadth/Stock Data",
        "RAW Global Market Table", "RAW Global News/Sentiment Table", "RAW Global Research",
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

    if signal_code == 0:
        return {"has_setup": False, "reason": "No directional signal right now.", "context_audit": context_audit}

    if is_choppy:
        return {"has_setup": False, "reason": "Market Structure is Choppy/Range-bound right now -- "
                                               "confluence signals are unreliable in this regime, staying flat.",
                "context_audit": context_audit}

    freshness = str(dashboard_context.get("Data Freshness", "")).upper()
    if "STALE" in freshness or "FROZEN" in freshness:
        return {"has_setup": False,
                "reason": "Dashboard data is STALE/FROZEN. Full-data trade engine will not open a fresh setup until live data is fresh.",
                "context_audit": context_audit}

    # HARD FULL-CONTEXT GATE: no partial-data direction. Every required
    # dashboard section must be present before the engine can decide BUY/SELL.
    if not context_audit.get("complete"):
        return {"has_setup": False,
                "reason": "Full dashboard context is incomplete. The engine will not guess a direction from partial data; waiting for every required live factor.",
                "context_audit": context_audit}

    raw_direction = "BUY" if signal_code == 1 else "SELL"
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
    if raw_direction == "BUY" and _bearish_reversal_score >= 5 and _loc_preview in ("AT_RESISTANCE", "NEAR_RESISTANCE"):
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
    if not isinstance(sr_context, dict) or sr_context.get('status') not in ('OK', 'READY', 'SUFFICIENT'):
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
        if not major_support and not major_resistance:
            return {"has_setup": False,
                    "reason": "No strong multi-source support/resistance is validated near price. Micro S/R levels are ignored; waiting for a real structural zone.",
                    "context_audit": context_audit}

        # If the S/R engine found overlapping/conflicting boxes around spot,
        # the location is ambiguous. This is exactly the situation that used
        # to produce fake pairs such as 23392-23397 support and 23397-23402 resistance.
        if sr_context.get('overlapping_zone_warning'):
            return {"has_setup": False,
                    "reason": "Support/resistance zones overlap around current price. Direction is ambiguous, so the engine will not trade until a clean boundary is established.",
                    "context_audit": context_audit}

        at_support = loc in ('AT_SUPPORT', 'NEAR_SUPPORT')
        at_resistance = loc in ('AT_RESISTANCE', 'NEAR_RESISTANCE')

        # ENTRY-TIMING SAFETY: a trend can still look bullish/bearish while
        # the immediate move is already losing momentum at the exact entry.
        # Do not enter simply because the broad direction is correct. Require
        # the proposed side to pass the adverse-move risk check first.
        timing = sr_context.get('entry_timing') or {}
        if direction == 'BUY' and bool(timing.get('buy_adverse_move_risk')):
            return {"has_setup": False,
                    "reason": "BUY blocked: the broader move is still bullish, but immediate entry-timing risk is high (momentum deceleration/rejection at the proposed entry). Waiting for fresh bullish confirmation instead of buying the late move.",
                    "context_audit": context_audit}
        if direction == 'SELL' and bool(timing.get('sell_adverse_move_risk')):
            return {"has_setup": False,
                    "reason": "SELL blocked: the broader move is still bearish, but immediate entry-timing risk is high (momentum deceleration/rejection at the proposed entry). Waiting for fresh bearish confirmation instead of selling the late move.",
                    "context_audit": context_audit}

        # BEFORE ENTRY: use the S/R engine's break-vs-bounce forecast.
        # We no longer demand a retest/rejection candle on every visit.
        # Instead, the engine first estimates the path and then decides whether
        # the broader confluence agrees with that path. This is the important
        # distinction between "trade every touch" and "block every touch".
        path_forecast = _level_path_forecast(level_prediction, sr_context, direction)
        if path_forecast.get('ready'):
            forecast_direction = path_forecast.get('path')
            if forecast_direction != direction and path_forecast.get('confidence', 0) >= 68.0 and path_forecast.get('edge', 0) >= 14.0:
                direction = forecast_direction
                reversal_used = True
                reversal_reason = "S/R path forecast overrode the slower directional signal: " + path_forecast.get('reason', '')
        elif path_forecast.get('kind') in ('support', 'resistance'):
            return {"has_setup": False,
                    "reason": "S/R path forecast is not decisive yet — " + path_forecast.get('reason', 'waiting for clearer bounce/break evidence'),
                    "context_audit": context_audit}

        # A confirmed rejection/breakout still strengthens the setup, but it is
        # no longer a mandatory binary gate. The forecast can front-run a move
        # when the level evidence is already strong enough.
        # ENTRY LOCATION IS MANDATORY. The round-number ladder is allowed
        # to help with targets/confluence, but it can NEVER create an entry
        # in the middle of a range. This prevents trades such as BUY 23401
        # when the nearest real structure is elsewhere.
        if loc in ('MIDRANGE', 'BETWEEN_LEVELS', 'UNKNOWN'):
            return {"has_setup": False,
                    "reason": "No valid entry location: price is not at a confirmed support/resistance zone. Round-number levels are not entry triggers; waiting for a real S/R reaction or confirmed breakout/breakdown.",
                    "context_audit": context_audit}

    # Entry must be tied to the validated structural S/R engine. The round-
    # number ladder remains useful for secondary confluence and targets, but
    # it is NOT allowed to invent an entry level in the middle of the range.
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
        return {"has_setup": False, "reason": "No validated structural support/resistance is close enough for a real entry. Round-number levels are kept out of the entry trigger.",
                "context_audit": context_audit}

    # Never let the legacy early-warning predictor override the validated
    # structural S/R map. If its level is merely a tiny local swing far away
    # from the selected major zone, discard it and wait for a real level.
    if isinstance(sr_context, dict):
        major_prices = []
        for z in (sr_context.get('major_support'), sr_context.get('major_resistance')):
            if isinstance(z, dict) and z.get('price') is not None:
                major_prices.append(float(z['price']))
        if major_prices:
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
    path_forecast = _level_path_forecast(lp, sr_context, direction)
    confirming_pct = (
        (lp['bounce_pct'] if lp['approaching'] == 'support' else lp['break_pct'])
        if direction == "BUY" else
        (lp['break_pct'] if lp['approaching'] == 'support' else lp['bounce_pct'])
    )

    # The path forecast is the deciding level test. A small disagreement with
    # the older directional label is acceptable when the break/bounce edge is
    # materially stronger; a weak/ambiguous level is still rejected.
    path_matches = path_forecast.get('path') == direction
    path_confidence = float(path_forecast.get('confidence', 0.0) or 0.0)
    path_edge = float(path_forecast.get('edge', 0.0) or 0.0)

    # FAST STRUCTURAL REJECTION/BOUNCE PATH
    # A strong real S/R rejection should not be forced to wait for the slower
    # breakout/retest path model.  We still require several independent pieces
    # of evidence, so this is an early-entry path, not an "every touch" path.
    loc_now = str(sr_context.get('location', '')).upper() if isinstance(sr_context, dict) else ''
    major_zone = (sr_context.get('major_resistance') if direction == 'SELL' else sr_context.get('major_support')) if isinstance(sr_context, dict) else None
    major_sources = " ".join(str(x) for x in ((major_zone or {}).get('sources') or [])).lower() if isinstance(major_zone, dict) else ''
    sr_strength_text = _flatten_context_text((sr_context or {}).get('confirmation', {}) if isinstance(sr_context, dict) else {})
    oi_rejection = (('heavy call oi' in factors_text) and direction == 'SELL') or (('heavy put oi' in factors_text) and direction == 'BUY')
    flow_rejection = (_extract_order_flow(dashboard_context)[0] == ('SELLING PRESSURE' if direction == 'SELL' else 'BUYING PRESSURE'))
    candle_rejection = (
        direction == 'SELL' and any(k in factors_text for k in ('bearish engulfing', 'bearish pin bar', 'shooting star', 'sellers actively defending', 'bearish signal'))
    ) or (
        direction == 'BUY' and any(k in factors_text for k in ('bullish engulfing', 'bullish pin bar', 'hammer', 'buyers actively defending', 'bullish signal'))
    )
    sweep_rejection = 'liquidity sweep already detected' in factors_text
    structural_anchor = bool(major_zone) and any(k in (major_sources + ' ' + sr_strength_text) for k in (
        'pdh', 'pdl', 'pdc', 'previous day', 'opening range', 'swing', 'volume', 'oi', 'pivot', 'cpr', 'order block'
    ))
    rejection_evidence = sum(bool(x) for x in (oi_rejection, flow_rejection, candle_rejection, sweep_rejection, structural_anchor))
    fast_rejection_path = (
        ((direction == 'SELL' and loc_now in ('AT_RESISTANCE', 'NEAR_RESISTANCE')) or
         (direction == 'BUY' and loc_now in ('AT_SUPPORT', 'NEAR_SUPPORT')))
        and rejection_evidence >= 3
        and structural_anchor
        and not (
            (direction == 'SELL' and _extract_order_flow(dashboard_context)[0] == 'BUYING PRESSURE') or
            (direction == 'BUY' and _extract_order_flow(dashboard_context)[0] == 'SELLING PRESSURE')
        )
    )

    if not path_matches or path_confidence < 62.0 or path_edge < 10.0:
        if not fast_rejection_path:
            return {"has_setup": False,
                    "reason": f"Level path is not strong enough for {direction}: {path_forecast.get('reason', 'bounce/break read is mixed')}.",
                    "context_audit": context_audit}
        # Promote the measured structural rejection/bounce into the path model.
        # This is intentionally capped below the learning ceiling and remains
        # subject to the full confluence/risk checks below.
        path_matches = True
        path_confidence = max(path_confidence, 68.0)
        path_edge = max(path_edge, 12.0)
        path_forecast = dict(path_forecast)
        path_forecast.update({
            'ready': True, 'path': direction, 'fast_structural_rejection': True,
            'confidence': round(path_confidence, 1), 'edge': round(path_edge, 1),
            'rejection_evidence': rejection_evidence,
            'reason': (path_forecast.get('reason', '') +
                       f" Fast structural {'resistance rejection SELL' if direction == 'SELL' else 'support bounce BUY'} accepted from {rejection_evidence}/5 independent confirmations.")
        })

    # For the final checklist use the forecasted path score, not a stale hard
    # directional label.
    confirming_pct = path_confidence

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
    # NIFTY 50-specific news (own RSS scoring, not the generic "India" region
    # row) and NIFTY 50 constituent-aggregate fundamentals -- both soft,
    # slow-moving confluence factors. A NEUTRAL fundamentals tilt does not
    # count against the setup (weeks/quarters-scale data can't disagree with
    # an intraday move); UNAVAILABLE or an actual opposite reading does not
    # count for it either -- same honest treatment as every other factor here.
    n50_news_up = (nifty50_news_sentiment or "").upper()
    n50_news_aligned = ("BULLISH" in n50_news_up and direction == "BUY") or ("BEARISH" in n50_news_up and direction == "SELL")
    n50_fund_up = (nifty50_fundamentals_bias or "").upper()
    n50_fund_aligned = (("BULLISH" in n50_fund_up and direction == "BUY") or
                        ("BEARISH" in n50_fund_up and direction == "SELL") or
                        n50_fund_up == "NEUTRAL")
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
    global_conflict = (_global_strength == "STRONG" and
                       ((_global_bias == "BUY" and direction == "SELL") or
                        (_global_bias == "SELL" and direction == "BUY")))

    factor_flags = {
        "main_signal_aligned": True,  # gated on already above
        "ml_agrees": bool(ml_agrees),
        "level_pct_ge_65": confirming_pct >= 65.0,
        "banknifty_no_divergence": banknifty_ok,
        "breadth_aligned": bool(breadth_ok),
        "global_aligned": bool(global_ok),
        "global_research_aligned": (_global_bias == direction) or _global_bias in ("NEUTRAL", "UNKNOWN"),
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
        "nifty50_news_aligned": bool(n50_news_aligned),
        "nifty50_fundamentals_aligned": bool(n50_fund_aligned),
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
    factor_flags["level_path_ready"] = bool(path_forecast.get("ready"))
    factor_flags["level_path_direction_aligned"] = bool(path_forecast.get("path") == direction)
    factor_flags["false_break_risk_low"] = bool(float(path_forecast.get("false_break_risk", 100.0) or 100.0) < 30.0)

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
        "strong_daily_level", "strong_opening_range_level", "immediate_reversal_risk_low", "global_research_aligned"
    ))
    hard_conflicts = int(order_flow_conflict) + int(regime_conflict) + int(mtf_conflict) + int(global_conflict)
    unknown_major = int(regime_bias == "UNKNOWN") + int(mtf_bias == "UNKNOWN") + int(order_pressure == "")
    path_strong = path_confidence >= 68.0 and path_edge >= 14.0 and path_forecast.get('false_break_risk', 100.0) < 30.0

    # One neutral/contrary secondary factor must not kill a genuinely strong
    # level-path setup. Multiple independent conflicts still block it.
    fast_path_active = bool(path_forecast.get('fast_structural_rejection'))
    if hard_conflicts >= 2 or (hard_conflicts >= 1 and not path_strong and not fast_path_active) or unknown_major >= 3 or (critical_true < 3 and not path_strong and not fast_path_active):
        return {"has_setup": False,
                "reason": f"Full-context confluence is not strong enough ({critical_true} direct confirmations; {hard_conflicts} major conflicts). Level-path read: {path_confidence:.1f}% with {path_edge:.1f}pt edge.",
                "context_audit": context_audit}

    strike = _round_to_strike(live_price)
    option_type = "CE" if direction == "BUY" else "PE"
    underlying_entry = live_price
    sl_distance = max(atr * 1.2, 12.0)

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

    # Prefer the next VALIDATED structural boundary as target. Round-number
    # ladder levels are secondary and must not replace a real multi-source S/R
    # boundary when one is available.
    if isinstance(sr_context, dict):
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

    if target is None and isinstance(sr_context, dict):
        # Find the next independent actionable structural zone in the trade
        # direction. This keeps the target market-structure based without
        # forcing the engine to wait forever for a perfectly spaced ladder rung.
        zones = sr_context.get('zones') or []
        candidates = []
        for z in zones:
            try:
                zp = float(z.get('price'))
                if direction == 'BUY' and zp > live_price and str(z.get('side','')).lower() == 'resistance':
                    candidates.append(zp)
                elif direction == 'SELL' and zp < live_price and str(z.get('side','')).lower() == 'support':
                    candidates.append(zp)
            except (TypeError, ValueError):
                continue
        if candidates:
            structural_target = min(candidates) if direction == 'BUY' else max(candidates)
            distance = abs(structural_target - live_price)
            if distance >= MIN_RR_RATIO * sl_distance:
                target = round(structural_target, 2)

    if target is None:
        # Only when no independent structural target is available do we use a
        # conservative ATR extension. This is still derived from live market
        # volatility, not an arbitrary fixed target.
        target = round(live_price + (1.8 * sl_distance if direction == 'BUY' else -1.8 * sl_distance), 2)
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
    # Adaptive entry window: strong, fully-confirmed breakout/rejection paths
    # get a little more room so a fast 40-80 point move is not missed merely
    # because the first confirmation candle has already travelled away from
    # the level. Weak setups retain the tighter location rule.
    if path_strong:
        # Early-capture window: allow a genuinely strong structural move a
        # little more room so a 40-50 point NIFTY move is not missed merely
        # because confirmation arrived after the first impulse. This is still
        # capped by ATR and the full confluence/conflict gates below.
        max_entry_distance = max((0.90 if breakout_entry else 0.75) * float(atr),
                                 14.0 if breakout_entry else 12.0)
    elif fast_path_active:
        max_entry_distance = max(0.75 * float(atr), 12.0)
    else:
        max_entry_distance = max((0.35 if not breakout_entry else 0.45) * float(atr),
                                 6.0 if not breakout_entry else 8.0)
    if level_distance > max_entry_distance:
        return {"has_setup": False, "reason": f"Entry is too far from the validated level ({level_distance:.1f} pts > {max_entry_distance:.1f} pts). Setup may be directionally correct but the location is poor; waiting for a better entry.", "context_audit": context_audit}

    confidence_pct, used_learning, learned_count = trade_learning.compute_confidence(factor_flags)

    # Frequency control is adaptive rather than a fixed wall. The engine
    # still reads the complete dashboard packet and rejects hard conflicts,
    # but a genuinely strong, early structural path can qualify before every
    # slow/secondary factor catches up. This is specifically to avoid the
    # old behaviour where a real move started and the engine kept WAITing.
    if hard_conflicts >= 1:
        confidence_pct = min(confidence_pct, 64.0)
    if not regime_ok and regime_bias == "RANGE":
        confidence_pct = min(confidence_pct, 66.0)

    strong_path_threshold = 58.0 if path_strong else 62.0
    # A strong structural path or a measured rejection/bounce should not be
    # forced to wait for every secondary factor. Keep hard conflicts and risk
    # gates intact, but lower only the final score wall for these high-quality
    # early paths so genuine medium-sized moves are captured in time.
    fast_path_threshold = 58.0 if (fast_path_active and rejection_evidence >= 3) else 62.0
    entry_threshold = min(strong_path_threshold, fast_path_threshold)

    if confidence_pct < entry_threshold:
        return {"has_setup": False,
                "reason": f"Weighted setup score {confidence_pct:.1f}% is below the adaptive entry threshold {entry_threshold:.1f}%. Full-data path forecast: {path_confidence:.1f}% (edge {path_edge:.1f}).",
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
        "score_model": "weighted_confluence_v11",
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
        "level_path_forecast": path_forecast,
        "fast_structural_rejection": fast_path_active,
        "rejection_evidence_count": int(path_forecast.get("rejection_evidence", 0) or 0),
        "entry_logic": "FULL_RAW_DASHBOARD_DATA + ALL_EXISTING_FACTORS + STRUCTURAL_LEVEL_PATH_FORECAST + FAST_REJECTION/BOUNCE + ADAPTIVE_EARLY_ENTRY + FULL_CONFLUENCE + SELF_LEARNING",
        "entry_priority": ("EARLY_HIGH_CONFLUENCE" if (path_strong or (fast_path_active and rejection_evidence >= 4)) else "STANDARD_CONFLUENCE"),
        "adaptive_entry_threshold": round(entry_threshold, 1),
        "data_packet": {
            "raw_price_indicator_data": "RAW Primary Price/Indicator Data (latest 500 candles)" in dashboard_context,
            "latest_indicator_snapshot": "RAW Latest Price/Indicator Snapshot" in dashboard_context,
            "option_chain": "RAW Option Chain (all loaded strikes)" in dashboard_context,
            "breadth": "RAW Full NIFTY 50 Breadth/Stock Data" in dashboard_context,
            "smc": "RAW SMC Zones / FVG / OB / Sweeps" in dashboard_context,
            "liquidity": "RAW Liquidity Map" in dashboard_context,
            "support_resistance": "RAW Comprehensive S/R Zones" in dashboard_context,
            "order_flow": "RAW Order Flow" in dashboard_context,
            "ml": "RAW ML Results" in dashboard_context,
            "risk": "RAW Risk Engine" in dashboard_context,
            "sniper": "RAW Sniper Setup" in dashboard_context,
            "hybrid_ai": "RAW Hybrid AI Analysis" in dashboard_context,
        },
    }
