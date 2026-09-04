"""
next_move_predictor.py — NEW, purely additive module.

Direct answer to one specific ask: "agle 5 minute me market kis side
zyada chance se jayega" -- ONE clear line, not 6 different sections to
read and mentally combine yourself.

IMPORTANT HONESTY NOTE (read before wiring a "100% accurate" claim
anywhere): no model, tool, or person can call 5-minute index direction
with guaranteed accuracy -- anyone claiming that is lying to you. What
this module actually does is take every live signal this dashboard
already computes (main confluence engine, validated ML model, level
approach read, compression/lean detector, sniper VWAP+CPR bias, Bank
Nifty divergence check, breadth, global markets, VIX, max-pain magnet)
and combine them into ONE weighted lean with an honest confidence
number -- so you get a single clear read instead of having to eyeball
6 sections yourself. It is a probability lean, not a guarantee, and it
says so on-screen every time.
"""


def predict_next_5min_direction(ai_analysis, ml_results, level_prediction, pre_move, sniper,
                                 banknifty_correlation_note, breadth_status, avg_market_change,
                                 live_vix, max_pain, live_price, is_choppy=False,
                                 india_news_sentiment=None, candle_pattern=None, smc_event=None):
    """
    Returns:
      {
        "direction": "UP" | "DOWN" | "SIDEWAYS",
        "confidence_pct": float,       # 0-100, how one-sided the votes are (post-classification)
        "raw_up_pct": float,           # 0-100 BEFORE classification/flip -- >50 = up lean. Used
                                        # by app.py to smooth across refreshes so the direction
                                        # doesn't flicker from small tick-to-tick noise.
        "votes_up": float, "votes_down": float, "votes_total_weight": float,
        "reasoning": [str, ...],       # every signal that voted, and which way
        "caution": str | None,         # e.g. choppy-regime or divergence warning
      }
    """
    votes = []  # (label, direction: +1/-1/0, weight)

    # 1. Main confluence engine (heaviest single vote -- it already blends
    #    technical + OI + sentiment + breadth internally)
    bias_text = (ai_analysis or {}).get('bias_text', '') or ''
    if "BULLISH" in bias_text.upper():
        votes.append(("Main Confluence Engine: Bullish", 1, 3.0))
    elif "BEARISH" in bias_text.upper():
        votes.append(("Main Confluence Engine: Bearish", -1, 3.0))
    else:
        votes.append(("Main Confluence Engine: Neutral/Choppy", 0, 3.0))

    # 2. Validated ML model
    if ml_results and ml_results.get("model_ready"):
        sig = ml_results.get("latest_signal", 0)
        if sig == 1:
            votes.append(("ML Model: Up", 1, 3.0))
        elif sig == -1:
            votes.append(("ML Model: Down", -1, 3.0))
        else:
            votes.append(("ML Model: Flat", 0, 3.0))

    # 3. Level-approach early warning (only counts if actually approaching a level)
    if level_prediction and level_prediction.get('status') == 'APPROACHING_LEVEL':
        lp_bias = (level_prediction.get('directional_bias') or '').lower()
        if "bullish" in lp_bias:
            votes.append((f"Level Approach ({level_prediction.get('level_type')} ₹{level_prediction.get('level_price'):,.2f}): Bullish read", 1, 2.0))
        elif "bearish" in lp_bias:
            votes.append((f"Level Approach ({level_prediction.get('level_type')} ₹{level_prediction.get('level_price'):,.2f}): Bearish read", -1, 2.0))

    # 4. Compression/pre-move lean (only counts if a real squeeze is active)
    if pre_move and pre_move.get('squeeze_active'):
        if pre_move.get('lean') == "BULLISH":
            votes.append((f"Compression Lean: Bullish ({pre_move.get('lean_score_pct')}%)", 1, 2.0))
        elif pre_move.get('lean') == "BEARISH":
            votes.append((f"Compression Lean: Bearish ({pre_move.get('lean_score_pct')}%)", -1, 2.0))

    # 5. Sniper Setup (VWAP + CPR institutional bias)
    sniper_bias = (sniper or {}).get('setup_bias') if sniper else None
    if sniper_bias == "Bullish":
        votes.append(("Sniper Setup (VWAP+CPR): Bullish", 1, 1.5))
    elif sniper_bias == "Bearish":
        votes.append(("Sniper Setup (VWAP+CPR): Bearish", -1, 1.5))

    # 6. Bank Nifty divergence -- this is a WARNING against the current move,
    #    not a fresh directional vote, so it subtracts confidence from
    #    whichever way the tally is already leaning rather than voting itself.
    banknifty_divergence = bool(banknifty_correlation_note and "DIVERGENCE WARNING" in banknifty_correlation_note.upper())

    # 7. Market breadth (heavyweights)
    bstat = (breadth_status or '').upper()
    if "BULLISH HEAVYWEIGHTS" in bstat:
        votes.append(("Market Breadth: Bullish heavyweights", 1, 1.0))
    elif "BEARISH HEAVYWEIGHTS" in bstat:
        votes.append(("Market Breadth: Bearish heavyweights", -1, 1.0))

    # 8. Global markets overnight/live average change
    if avg_market_change is not None:
        if avg_market_change > 0.1:
            votes.append((f"Global Markets: +{avg_market_change:.2f}% avg -- supportive", 1, 1.0))
        elif avg_market_change < -0.1:
            votes.append((f"Global Markets: {avg_market_change:.2f}% avg -- weighing down", -1, 1.0))

    # 9. Max Pain magnet -- price tends to drift TOWARD max pain near expiry,
    #    so it's a pull in that direction, not away from it.
    if max_pain is not None and live_price is not None:
        if live_price > max_pain:
            votes.append((f"Max Pain (₹{max_pain:,.2f}) magnet pulling DOWN toward it", -1, 0.5))
        elif live_price < max_pain:
            votes.append((f"Max Pain (₹{max_pain:,.2f}) magnet pulling UP toward it", 1, 0.5))

    # 10. India news sentiment
    news_up = (india_news_sentiment or '').upper()
    if "BULLISH" in news_up:
        votes.append(("India News Sentiment: Bullish", 1, 0.5))
    elif "BEARISH" in news_up:
        votes.append(("India News Sentiment: Bearish", -1, 0.5))

    # 11. Latest candlestick pattern -- ALWAYS available every refresh
    #     (unlike level-approach/squeeze which only fire in specific
    #     regimes), so this keeps the tally grounded in real price-action
    #     on every single candle, not just when a special setup exists.
    if candle_pattern and candle_pattern.get('bias') in ('BULLISH', 'BEARISH'):
        w = 1.0 * float(candle_pattern.get('strength', 0.5))
        if candle_pattern['bias'] == 'BULLISH':
            votes.append((f"Candle Pattern: {candle_pattern.get('pattern', 'Bullish')}", 1, w))
        else:
            votes.append((f"Candle Pattern: {candle_pattern.get('pattern', 'Bearish')}", -1, w))

    # 12. Market structure (BOS/CHoCH) -- also ALWAYS available every refresh
    smc_up = (smc_event or '').upper()
    if "BOS" in smc_up and "BULLISH" in smc_up:
        votes.append((f"Market Structure: {smc_event}", 1, 1.0))
    elif "CHOCH" in smc_up and "BEARISH" in smc_up:
        votes.append((f"Market Structure: {smc_event}", -1, 1.0))

    votes_up = sum(w for _, d, w in votes if d == 1)
    votes_down = sum(w for _, d, w in votes if d == -1)
    total_weight = sum(w for _, _, w in votes)

    net = votes_up - votes_down
    if total_weight <= 0:
        direction, confidence_pct, raw_up_pct = "SIDEWAYS", 50.0, 50.0
    else:
        raw_up_pct = round(50.0 + (net / total_weight) * 50.0, 1)
        raw_up_pct = max(1.0, min(99.0, raw_up_pct))
        confidence_pct = raw_up_pct
        if confidence_pct >= 58:
            direction = "UP"
        elif confidence_pct <= 42:
            direction = "DOWN"
            confidence_pct = round(100 - confidence_pct, 1)  # express as confidence in DOWN, not distance from 50
        else:
            direction = "SIDEWAYS"

    caution_parts = []
    if is_choppy:
        caution_parts.append("Volatility abhi compressed/choppy hai -- short-term direction reads is regime me kam reliable hote hain.")
        confidence_pct = min(confidence_pct, 60.0)
    if banknifty_divergence:
        caution_parts.append(banknifty_correlation_note)
        confidence_pct = min(confidence_pct, 60.0)
    if live_vix is not None and live_vix > 16:
        caution_parts.append(f"India VIX {live_vix:.2f} elevated hai -- whipsaw ka risk zyada.")

    reasoning = [f"{label}" for label, _, _ in votes]

    return {
        "direction": direction,
        "confidence_pct": confidence_pct,
        "raw_up_pct": raw_up_pct,
        "votes_up": round(votes_up, 2),
        "votes_down": round(votes_down, 2),
        "votes_total_weight": round(total_weight, 2),
        "reasoning": reasoning,
        "caution": " | ".join(caution_parts) if caution_parts else None,
    }


def smooth_and_classify(raw_up_pct, previous_smoothed, alpha=0.35, up_enter=58, down_enter=42):
    """
    Exponentially smooths raw_up_pct across refreshes and classifies with
    hysteresis, so the direction doesn't flicker UP/DOWN/SIDEWAYS on every
    30-second refresh from small tick-to-tick noise -- it only changes once
    the underlying lean has genuinely, persistently shifted.

    `previous_smoothed` should be whatever this function returned as
    "smoothed" last time (pass None on the very first call). Caller (app.py)
    is responsible for persisting it in st.session_state across reruns.

    Returns: {"smoothed": float, "direction": "UP"|"DOWN"|"SIDEWAYS", "confidence_pct": float}
    """
    if previous_smoothed is None:
        smoothed = raw_up_pct
    else:
        smoothed = round(alpha * raw_up_pct + (1 - alpha) * previous_smoothed, 2)

    if smoothed >= up_enter:
        direction, confidence_pct = "UP", smoothed
    elif smoothed <= down_enter:
        direction, confidence_pct = "DOWN", round(100 - smoothed, 1)
    else:
        direction, confidence_pct = "SIDEWAYS", round(100 - abs(smoothed - 50) * 2, 1)

    return {"smoothed": smoothed, "direction": direction, "confidence_pct": confidence_pct}
