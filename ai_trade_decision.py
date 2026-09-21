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

def _find_best_ladder_level(level_ladder, live_price, atr, direction, max_atr_mult=1.6):
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


def generate_trade_decision(live_price, level_prediction, atr, max_pain=None,
                             signal_code=0, ml_agrees=False, banknifty_correlation_note=None,
                             breadth_advances=None, breadth_declines=None,
                             global_avg_change=None, live_vix=None, india_news_sentiment=None,
                             level_ladder=None, sr_context=None, sniper_bias=None, is_choppy=False,
                             dashboard_context=None):
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

    direction = "BUY" if signal_code == 1 else "SELL"

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

        if loc in ('AT_RESISTANCE', 'NEAR_RESISTANCE') and direction == 'BUY' and not conf.get('breakout_confirmed'):
            return {"has_setup": False, "reason": "Price is at/near a validated resistance zone, but a confirmed breakout is not present — BUY blocked. Wait for a completed close above the zone with volume/strong-body confirmation.", "context_audit": context_audit}
        if loc in ('AT_SUPPORT', 'NEAR_SUPPORT') and direction == 'SELL' and not conf.get('breakdown_confirmed'):
            return {"has_setup": False, "reason": "Price is at/near a validated support zone, but a confirmed breakdown is not present — SELL blocked. Wait for a completed close below the zone with volume/strong-body confirmation.", "context_audit": context_audit}

        # At support, a BUY needs actual rejection/defence confirmation; at
        # resistance, a SELL needs actual rejection. A location label alone is
        # never an entry trigger.
        if loc in ('AT_SUPPORT', 'NEAR_SUPPORT') and direction == 'BUY' and not conf.get('support_rejection_confirmed'):
            return {"has_setup": False, "reason": "Price is near support, but buyer rejection/defence is not confirmed yet — BUY blocked to avoid entering before the bounce is proven.", "context_audit": context_audit}
        if loc in ('AT_RESISTANCE', 'NEAR_RESISTANCE') and direction == 'SELL' and not conf.get('resistance_rejection_confirmed'):
            return {"has_setup": False, "reason": "Price is near resistance, but seller rejection is not confirmed yet — SELL blocked to avoid entering after an exhausted move without confirmation.", "context_audit": context_audit}
        # ENTRY LOCATION IS MANDATORY. The round-number ladder is allowed
        # to help with targets/confluence, but it can NEVER create an entry
        # in the middle of a range. This prevents trades such as BUY 23401
        # when the nearest real structure is elsewhere.
        if loc in ('MIDRANGE', 'BETWEEN_LEVELS', 'UNKNOWN'):
            return {"has_setup": False,
                    "reason": "No valid entry location: price is not at a confirmed support/resistance zone. Round-number levels are not entry triggers; waiting for a real S/R reaction or confirmed breakout/breakdown.",
                    "context_audit": context_audit}

    # First try the single swing-based nearest level (narrow but most
    # "official" read). If that's not currently in its approach zone,
    # fall back to scanning the full 16-level round-number Ladder --
    # this is what stops the engine from sitting idle for days just
    # because ONE specific swing level never happened to be nearby.
    lp = None
    if level_prediction is not None and level_prediction.get('status') == 'APPROACHING_LEVEL':
        lp = level_prediction
    if lp is None:
        lp = _find_best_ladder_level(level_ladder, live_price, atr, direction)
    if lp is None:
        return {"has_setup": False, "reason": "No key level (from the validated S/R engine OR the full ladder) is close enough with a matching directional read right now.",
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
        "level_pct_ge_65": confirming_pct >= 65.0,
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
        "mtf_aligned": bool(mtf_ok),
    }

    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    # Do not require every secondary factor.  A few direct confirmations
    # should be able to trigger a trade, while major contradictions still
    # block it. This increases frequency modestly without turning the engine
    # into an always-trading system.
    critical_true = sum(bool(factor_flags.get(k)) for k in (
        "main_signal_aligned", "level_pct_ge_65", "htf_1h_aligned",
        "htf_15min_aligned", "vwap_aligned", "order_flow_aligned",
        "regime_aligned", "mtf_aligned", "oi_aligned", "liquidity_sweep"
    ))
    hard_conflicts = int(order_flow_conflict) + int(regime_conflict) + int(mtf_conflict)
    if critical_true < 3 or hard_conflicts >= 2:
        return {"has_setup": False,
                "reason": f"Confluence is not strong enough ({critical_true} direct confirmations; {hard_conflicts} major conflicts). Waiting for a cleaner setup.",
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
        tgt_distance = max(atr * 2.4, 24.0, MIN_RR_RATIO * sl_distance)
        target = round(underlying_entry + tgt_distance, 2) if direction == "BUY" else round(underlying_entry - tgt_distance, 2)
    if direction == "BUY":
        stop_loss = round(underlying_entry - sl_distance, 2)
    else:
        stop_loss = round(underlying_entry + sl_distance, 2)

    confidence_pct, used_learning, learned_count = trade_learning.compute_confidence(factor_flags)

    # Frequency control: accept a candidate when the weighted evidence is
    # solid, not only when an arbitrary number of boxes are checked.
    # 62 is deliberately reachable on a good intraday setup, while the
    # strongest structural conflicts still prevent entry.
    if hard_conflicts >= 1:
        confidence_pct = min(confidence_pct, 64.0)
    if not regime_ok and regime_bias == "RANGE":
        confidence_pct = min(confidence_pct, 66.0)
    if confidence_pct < 62.0 or (confidence_pct < 66.0 and critical_true < 4):
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
        "score_model": "weighted_confluence_v11",
        "critical_confirmations": critical_true,
        "major_conflicts": hard_conflicts,
        "order_flow_pressure": order_pressure,
        "regime_bias": regime_bias,
        "mtf_bias": mtf_bias,
    }
