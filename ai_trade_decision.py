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


def generate_trade_decision(live_price, level_prediction, atr, max_pain=None,
                             signal_code=0, ml_agrees=False, banknifty_correlation_note=None,
                             breadth_advances=None, breadth_declines=None,
                             global_avg_change=None, live_vix=None, india_news_sentiment=None,
                             level_ladder=None, sniper_bias=None, is_choppy=False):
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

    "has_setup" only becomes True when at least
    trade_learning.MIN_FACTORS_TRUE_TO_QUALIFY factors are aligned -- this
    is the discipline check that stops trades from being flagged at
    every single refresh regardless of quality.
    """
    if signal_code == 0 or level_prediction is None or level_prediction.get('status') != 'APPROACHING_LEVEL':
        return {"has_setup": False, "reason": "No directional signal + key-level approach lined up right now."}

    if is_choppy:
        return {"has_setup": False, "reason": "Market Structure is Choppy/Range-bound right now -- "
                                               "confluence signals are unreliable in this regime, staying flat."}

    direction = "BUY" if signal_code == 1 else "SELL"
    lp = level_prediction
    bias_lower = lp['directional_bias'].lower()
    level_confirms = ("bullish" in bias_lower and direction == "BUY") or \
                      ("bearish" in bias_lower and direction == "SELL")
    if not level_confirms:
        return {"has_setup": False, "reason": "Main signal and nearest level's read disagree -- staying flat."}

    confirming_pct = (
        (lp['bounce_pct'] if lp['approaching'] == 'support' else lp['break_pct'])
        if direction == "BUY" else
        (lp['break_pct'] if lp['approaching'] == 'support' else lp['bounce_pct'])
    )

    factors_text = " ".join(lp.get('factors', [])).lower()
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
    }

    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    if factors_true < trade_learning.MIN_FACTORS_TRUE_TO_QUALIFY:
        return {"has_setup": False,
                "reason": f"Only {factors_true}/{factors_total} checklist factors aligned -- "
                          f"not enough real confluence to call this a genuine setup yet."}

    strike = _round_to_strike(live_price)
    option_type = "CE" if direction == "BUY" else "PE"
    underlying_entry = live_price
    sl_distance = max(atr * 1.2, 12.0)

    # -----------------------------------------------------------------
    # NEW — use the FULL round-number Ladder Calculator (every 50pt level,
    # both directions), not just the single nearest level:
    #   1. Extra confluence factor: do at least 2 of the next 3 ladder
    #      levels in this SAME direction also read favorably? (i.e. is
    #      this a run of aligned levels, not a one-off single reading)
    #   2. Smarter target: instead of a flat ATR multiple, target the
    #      NEXT ladder level in this direction -- an actual real
    #      support/resistance rung, not an arbitrary distance.
    # -----------------------------------------------------------------
    ladder_confluence_aligned = False
    target = None
    if level_ladder:
        same_side_levels = level_ladder.get('resistances' if direction == "BUY" else 'supports') or []
        agreeing = 0
        for lvl in same_side_levels[:3]:
            lvl_bias = lvl.get('directional_bias', '').lower()
            if ("bullish" in lvl_bias and direction == "BUY") or ("bearish" in lvl_bias and direction == "SELL"):
                agreeing += 1
        ladder_confluence_aligned = agreeing >= 2
        if same_side_levels:
            target = same_side_levels[0]['level_price']  # nearest ladder level in this direction

    factor_flags["ladder_confluence_aligned"] = ladder_confluence_aligned
    factors_true = sum(1 for v in factor_flags.values() if v)
    factors_total = len(factor_flags)

    if target is None:
        tgt_distance = max(atr * 2.4, 24.0)
        target = round(underlying_entry + tgt_distance, 2) if direction == "BUY" else round(underlying_entry - tgt_distance, 2)
    if direction == "BUY":
        stop_loss = round(underlying_entry - sl_distance, 2)
    else:
        stop_loss = round(underlying_entry + sl_distance, 2)

    confidence_pct, used_learning, learned_count = trade_learning.compute_confidence(factor_flags)
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
        "level_price": lp['level_price'], "level_pct": confirming_pct
    }
