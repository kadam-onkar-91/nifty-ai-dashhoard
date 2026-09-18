"""AI trade decision engine.

Design goals: use the complete dashboard context, keep safety gates hard,
and use weighted confluence for the remaining evidence so the engine does
not require every indicator to agree.  No confidence number is a guarantee.
"""
import math
import re
import trade_learning


def _round_to_strike(price, step=50):
    return int(round(float(price) / step) * step)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{k}:{_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    return str(value)


def _norm(v):
    return _text(v).lower().replace("_", " ")


def _directional(text, direction):
    t = _norm(text)
    bull = any(x in t for x in ("bullish", "buying pressure", "positive", "uptrend", "breakout", "long"))
    bear = any(x in t for x in ("bearish", "selling pressure", "negative", "downtrend", "breakdown", "short"))
    return bull if direction == "BUY" else bear


def _extract_order_flow(ctx):
    x = ctx.get("RAW Order Flow", {})
    t = _norm(x)
    if "buying pressure" in t:
        return 1
    if "selling pressure" in t:
        return -1
    m = re.search(r"imbalance[^-+0-9]*([+-]?\d+(?:\.\d+)?)", t)
    if m:
        try:
            n = float(m.group(1))
            return 1 if n > 0.08 else -1 if n < -0.08 else 0
        except ValueError:
            pass
    return 0


def _extract_regime(ctx):
    t = _norm(ctx.get("RAW Regime Engine", {}))
    if "trending bullish" in t or "bullish trend" in t:
        return "BULLISH"
    if "trending bearish" in t or "bearish trend" in t:
        return "BEARISH"
    if "range" in t or "sideways" in t:
        return "RANGE"
    return "UNKNOWN"


def _extract_mtf(ctx, direction):
    t = _norm(ctx.get("RAW Multi-Timeframe Structure", {}))
    if "fully aligned" in t:
        return True
    bull = len(re.findall(r"bullish", t))
    bear = len(re.findall(r"bearish", t))
    return bull >= 2 if direction == "BUY" else bear >= 2


def _extract_confirmation(ctx, lp, direction):
    t = _norm(ctx.get("RAW Comprehensive S/R Zones", {})) + " " + _norm(ctx.get("RAW SMC Zones / FVG / OB / Sweeps", {}))
    factors = _norm(lp.get("factors", []))
    t += " " + factors
    if direction == "BUY":
        return any(x in t for x in ("breakout confirmed", "bullish confirmation", "bullish displacement", "bullish close", "rejection confirmed", "confirmed sweep"))
    return any(x in t for x in ("breakdown confirmed", "bearish confirmation", "bearish displacement", "bearish close", "rejection confirmed", "confirmed sweep"))


def _find_best_ladder_level(level_ladder, live_price, atr, direction, max_atr_mult=1.8):
    if not isinstance(level_ladder, dict) or not atr:
        return None
    candidates = (level_ladder.get("supports") or []) + (level_ladder.get("resistances") or [])
    best = None
    for lvl in candidates:
        try:
            dist = float(lvl.get("distance_pts", abs(float(lvl.get("level_price")) - live_price)))
        except Exception:
            continue
        if dist > max_atr_mult * float(atr):
            continue
        if not _directional(lvl.get("directional_bias", ""), direction):
            continue
        if best is None or dist < best[0]:
            best = (dist, lvl)
    if not best:
        return None
    lvl = best[1]
    price = float(lvl["level_price"])
    return {
        "status": "APPROACHING_LEVEL",
        "approaching": "support" if price < live_price else "resistance",
        "level_price": price,
        "distance_pts": best[0],
        "break_pct": float(lvl.get("break_pct", 0) or 0),
        "bounce_pct": float(lvl.get("bounce_pct", 0) or 0),
        "directional_bias": lvl.get("directional_bias", ""),
        "factors": lvl.get("factors", []),
    }


def _smart_strike(raw_chain, live_price, direction):
    """Choose a liquid available strike when the raw option-chain structure permits it.
    Falls back safely to ATM; never invents a strike that wasn't loaded.
    """
    rows = []
    def collect(x):
        if isinstance(x, dict):
            # common nested forms
            if any(k in x for k in ("strike", "strike_price", "strikePrice")):
                rows.append(x)
            for v in x.values():
                collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)
    collect(raw_chain)
    candidates = []
    for r in rows:
        try:
            strike = float(r.get("strike", r.get("strike_price", r.get("strikePrice"))))
        except Exception:
            continue
        # Prefer the relevant option side if explicitly present.
        side = _norm(r.get("option_type", r.get("type", r.get("instrument_type", ""))))
        want = "ce" if direction == "BUY" else "pe"
        if side and want not in side and direction in ("BUY", "SELL"):
            continue
        try:
            oi = float(r.get("oi", r.get("open_interest", r.get("openInterest", 0))) or 0)
            vol = float(r.get("volume", r.get("total_volume", 0)) or 0)
            iv = float(r.get("iv", r.get("implied_volatility", 0)) or 0)
            spread = abs(float(r.get("ask", 0) or 0) - float(r.get("bid", 0) or 0))
        except Exception:
            oi = vol = iv = spread = 0
        distance = abs(strike - live_price)
        # Liquidity score; ATM proximity is useful, but not at the expense of
        # a very illiquid contract.
        score = math.log1p(max(oi, 0)) * 1.5 + math.log1p(max(vol, 0)) - distance / max(live_price, 1) * 1000 - spread * 0.2
        candidates.append((score, strike))
    if not candidates:
        return _round_to_strike(live_price)
    return int(min(candidates, key=lambda z: -z[0])[1])


def generate_trade_decision(live_price, level_prediction, atr, max_pain=None,
                             signal_code=0, ml_agrees=False, banknifty_correlation_note=None,
                             breadth_advances=None, breadth_declines=None,
                             global_avg_change=None, live_vix=None, india_news_sentiment=None,
                             level_ladder=None, sr_context=None, sniper_bias=None, is_choppy=False,
                             dashboard_context=None):
    ctx = dashboard_context or {}
    required = (
        "Market Status", "Data Freshness", "Nifty Spot Price", "RAW Option Chain (all loaded strikes)",
        "RAW Full NIFTY 50 Breadth/Stock Data", "RAW Global Market Table", "RAW Global News/Sentiment Table",
        "RAW Multi-Timeframe Structure", "RAW SMC Zones / FVG / OB / Sweeps", "RAW Liquidity Map",
        "RAW S/R Ladder", "RAW Comprehensive S/R Zones", "RAW Regime Engine", "RAW CPR/Pivots/Opening Range",
        "RAW Order Flow", "RAW ML Results", "RAW Backtest Report", "RAW Monte Carlo Report",
        "RAW Model Drift Report", "RAW Position Sizing", "RAW Risk Engine", "RAW Sniper Setup", "RAW Hybrid AI Analysis")
    audit = {"sections_received": len(ctx), "required_sections": len(required),
             "missing_sections": [k for k in required if k not in ctx], "complete": all(k in ctx for k in required)}

    if signal_code == 0:
        return {"has_setup": False, "reason": "No directional signal right now.", "context_audit": audit}
    if not audit["complete"]:
        return {"has_setup": False, "reason": "Master dashboard context is incomplete; fresh trade blocked until live modules are available.", "context_audit": audit}
    freshness = _norm(ctx.get("Data Freshness"))
    if "stale" in freshness or "frozen" in freshness:
        return {"has_setup": False, "reason": "Dashboard data is STALE/FROZEN.", "context_audit": audit}
    if is_choppy:
        return {"has_setup": False, "reason": "Market is choppy/range-bound; waiting for cleaner structure.", "context_audit": audit}

    direction = "BUY" if signal_code == 1 else "SELL"
    sr_context = sr_context or ctx.get("RAW Comprehensive S/R Zones")
    if isinstance(sr_context, dict):
        loc = str(sr_context.get("location", "UNKNOWN")).upper()
        conf = sr_context.get("confirmation") or {}
        if loc in ("AT_RESISTANCE", "NEAR_RESISTANCE") and direction == "BUY" and not conf.get("breakout_confirmed"):
            return {"has_setup": False, "reason": "BUY blocked at resistance until breakout confirmation.", "context_audit": audit}
        if loc in ("AT_SUPPORT", "NEAR_SUPPORT") and direction == "SELL" and not conf.get("breakdown_confirmed"):
            return {"has_setup": False, "reason": "SELL blocked at support until breakdown confirmation.", "context_audit": audit}
        if loc == "MIDRANGE":
            return {"has_setup": False, "reason": "Price is mid-range; location is poor for a fresh entry.", "context_audit": audit}

    lp = level_prediction if isinstance(level_prediction, dict) and level_prediction.get("status") == "APPROACHING_LEVEL" else None
    if lp is None:
        lp = _find_best_ladder_level(level_ladder, live_price, atr, direction)
    if lp is None:
        return {"has_setup": False, "reason": "No nearby level with matching directional bias.", "context_audit": audit}
    if not _directional(lp.get("directional_bias", ""), direction):
        return {"has_setup": False, "reason": "Main signal and level bias disagree.", "context_audit": audit}

    factors_text = _norm(lp.get("factors", []))
    order_flow = _extract_order_flow(ctx)
    regime = _extract_regime(ctx)
    mtf_ok = _extract_mtf(ctx, direction)
    confirmation_ok = _extract_confirmation(ctx, lp, direction)
    breadth_ok = breadth_advances is not None and breadth_declines is not None and (float(breadth_advances)-float(breadth_declines)) * (1 if direction == "BUY" else -1) > 0
    global_ok = global_avg_change is not None and float(global_avg_change) * (1 if direction == "BUY" else -1) > 0.1
    bank_ok = not (banknifty_correlation_note and "divergence warning" in _norm(banknifty_correlation_note))
    news = _norm(india_news_sentiment)
    news_ok = _directional(news, direction)
    sniper_ok = _directional(sniper_bias, direction)
    vwap_ok = (("above a rising vwap" in factors_text) if direction == "BUY" else ("below a falling vwap" in factors_text))
    oi_ok = (("heavy put oi" in factors_text) if direction == "BUY" else ("heavy call oi" in factors_text))
    liquidity_ok = "liquidity sweep" in factors_text or "sweep" in _norm(ctx.get("RAW Liquidity Map"))
    smc_ok = any(x in (factors_text + " " + _norm(ctx.get("RAW SMC Zones / FVG / OB / Sweeps"))) for x in ("order block", "fvg", "fair value gap", "displacement"))
    ml_ok = bool(ml_agrees) or _directional(ctx.get("RAW ML Results"), direction)
    regime_ok = (regime == ("BULLISH" if direction == "BUY" else "BEARISH"))
    flow_ok = order_flow == (1 if direction == "BUY" else -1)
    low_vix = live_vix is not None and float(live_vix) < 15
    max_pain_ok = max_pain is not None and ((direction == "BUY" and live_price > max_pain) or (direction == "SELL" and live_price < max_pain))
    level_pct = float(lp.get("bounce_pct" if lp.get("approaching") == "support" else "break_pct", 0) or 0)
    level_ok = level_pct >= 60

    flags = {
        "main_signal_aligned": True, "level_pct_ge_65": level_ok, "ml_agrees": ml_ok,
        "banknifty_no_divergence": bank_ok, "breadth_aligned": breadth_ok, "global_aligned": global_ok,
        "oi_aligned": oi_ok, "fvg_ob_confluence": smc_ok, "vwap_aligned": vwap_ok,
        "htf_1h_aligned": _directional(ctx.get("RAW Multi-Timeframe Structure"), direction),
        "htf_15min_aligned": mtf_ok, "round_number_level": "round number" in factors_text,
        "liquidity_sweep": liquidity_ok, "low_vix": low_vix, "news_sentiment_aligned": news_ok,
        "sniper_setup_aligned": sniper_ok, "away_from_max_pain": max_pain_ok,
        "order_flow_aligned": flow_ok, "regime_aligned": regime_ok, "mtf_aligned": mtf_ok,
        "entry_confirmation": confirmation_ok,
    }

    # Weighted score: critical structure is mandatory; secondary indicators
    # add evidence. This permits some indicators to be neutral without
    # forcing the engine into either overtrading or zero-trade behavior.
    weights = {
        "main_signal_aligned": 12, "level_pct_ge_65": 8, "ml_agrees": 7, "banknifty_no_divergence": 5,
        "breadth_aligned": 5, "global_aligned": 3, "oi_aligned": 7, "fvg_ob_confluence": 7,
        "vwap_aligned": 6, "htf_1h_aligned": 8, "htf_15min_aligned": 6, "round_number_level": 2,
        "liquidity_sweep": 8, "low_vix": 2, "news_sentiment_aligned": 2, "sniper_setup_aligned": 5,
        "away_from_max_pain": 2, "order_flow_aligned": 8, "regime_aligned": 8, "mtf_aligned": 7,
        "entry_confirmation": 10,
    }
    score = sum(weights[k] for k, v in flags.items() if v)
    negative = 0
    if order_flow and order_flow != (1 if direction == "BUY" else -1): negative += 8
    if regime in ("BULLISH", "BEARISH") and not regime_ok: negative += 10
    if not mtf_ok: negative += 4
    if not bank_ok: negative += 10
    score -= negative

    # Hard safety gates: no amount of indicator confluence can override these.
    critical = flags["main_signal_aligned"] and flags["level_pct_ge_65"]
    if not critical:
        return {"has_setup": False, "reason": f"Level confirmation is weak ({level_pct:.1f}%). Waiting for a better entry.", "score": round(score,1), "factor_flags": flags, "context_audit": audit}
    if not confirmation_ok and score < 76:
        return {"has_setup": False, "reason": "Entry trigger is not confirmed yet; waiting for rejection/close/displacement confirmation.", "score": round(score,1), "factor_flags": flags, "context_audit": audit}
    if score < 70:
        return {"has_setup": False, "reason": f"Confluence score {score:.1f}/100 is below the trade threshold.", "score": round(score,1), "factor_flags": flags, "context_audit": audit}

    sl_distance = max(float(atr or 0) * 1.2, 12.0)
    min_rr = 1.6
    target = None
    same_side = (level_ladder or {}).get("resistances" if direction == "BUY" else "supports", []) if isinstance(level_ladder, dict) else []
    for lvl in same_side:
        try:
            p = float(lvl["level_price"]); dist = abs(p - live_price)
            if dist >= min_rr * sl_distance and ((direction == "BUY" and p > live_price) or (direction == "SELL" and p < live_price)):
                target = round(p,2); break
        except Exception:
            continue
    if target is None:
        target = round(live_price + (min_rr * sl_distance),2) if direction == "BUY" else round(live_price - (min_rr * sl_distance),2)
    stop = round(live_price - sl_distance,2) if direction == "BUY" else round(live_price + sl_distance,2)
    rr = abs(target-live_price)/sl_distance
    if rr < min_rr:
        return {"has_setup": False, "reason": "Risk/reward is below the minimum threshold.", "context_audit": audit}

    strike = _smart_strike(ctx.get("RAW Option Chain (all loaded strikes)"), live_price, direction)
    confidence, used_learning, learned_count = trade_learning.compute_confidence(flags, regime=regime)
    # Score contributes to confidence only within a bounded range; it is not
    # allowed to manufacture a high probability from a cosmetic formula.
    confidence = min(trade_learning.CONFIDENCE_CEILING, max(trade_learning.CONFIDENCE_FLOOR, round((confidence + score)/2,1)))
    track = trade_learning.get_overall_track_record()
    note = f"Weighted confluence {score:.1f}/100; RR {rr:.2f}. "
    note += f"Historical learning used for {learned_count} factor(s)." if used_learning else "Insufficient resolved history for adaptive factor learning."
    if track.get("sample_size",0): note += f" Overall resolved track record: {track['sample_size']} trades, {track['win_rate']}% wins."

    flags["_meta_regime"] = regime
    flags["_meta_score"] = round(score,2)
    return {
        "has_setup": True, "direction": direction, "strike": strike,
        "option_type": "CE" if direction == "BUY" else "PE", "underlying_entry": round(float(live_price),2),
        "stop_loss": stop, "target": target, "confidence_pct": confidence, "confidence_note": note,
        "factor_flags": flags, "factors_true": sum(bool(v) for k,v in flags.items() if not k.startswith("_")),
        "factors_total": len([k for k in flags if not k.startswith("_")]), "score": round(score,1), "risk_reward": round(rr,2),
        "level_price": lp.get("level_price"), "level_pct": level_pct, "context_audit": audit,
    }
