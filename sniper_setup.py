import pandas as pd


def calculate_pdh_pdl_cpr(df):
    """
    Previous Day High (PDH), Previous Day Low (PDL), and Central Pivot
    Range (CPR: Pivot / Top Central / Bottom Central) -- the "Sniper Setup"
    framework's required daily reference levels.

    Uses the SAME 5-min candle dataframe already fetched by market_data.py
    (last 5 trading days, real Upstox/Yahoo data) -- no extra API call
    needed. Groups candles by calendar date and uses the last fully
    completed prior trading day (never the still-forming "today" candle).
    """
    if df is None or df.empty or len(df) < 10:
        return None
    try:
        dates = pd.Series(df.index.date, index=df.index)
        unique_dates = sorted(dates.unique())
        if len(unique_dates) < 2:
            return None
        prev_day = unique_dates[-2]
        prev_day_df = df[dates == prev_day]
        if prev_day_df.empty:
            return None

        pdh = float(prev_day_df['High'].max())
        pdl = float(prev_day_df['Low'].min())
        pdc = float(prev_day_df['Close'].iloc[-1])

        pivot = (pdh + pdl + pdc) / 3
        bc = (pdh + pdl) / 2
        tc = (pivot - bc) + pivot
        top, bottom = max(tc, bc), min(tc, bc)

        return {
            "pdh": round(pdh, 2), "pdl": round(pdl, 2), "pdc": round(pdc, 2),
            "pivot": round(pivot, 2), "cpr_top": round(top, 2), "cpr_bottom": round(bottom, 2),
            "cpr_width_pct": round(((top - bottom) / pivot) * 100, 3) if pivot else None,
        }
    except Exception:
        return None


def _nearest_zone(zones, live_price, direction):
    """Finds the nearest Order Block / FVG zone above or below live_price."""
    candidates = []
    for z in (zones or []):
        try:
            price = float(z.get("Price"))
        except Exception:
            continue
        if direction == "above" and price > live_price:
            candidates.append((price - live_price, z))
        elif direction == "below" and price < live_price:
            candidates.append((live_price - price, z))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _option_oi_at_strike(df_option_chain, strike_ref):
    """Finds the option chain row nearest to a reference price, rounded to
    the nearest 50-point Nifty strike step."""
    if df_option_chain is None or df_option_chain.empty or strike_ref is None:
        return None
    try:
        target_strike = round(strike_ref / 50) * 50
        tmp = df_option_chain.copy()
        tmp['dist'] = (tmp['Strike'] - target_strike).abs()
        return tmp.sort_values('dist').iloc[0]
    except Exception:
        return None


def generate_sniper_setup(live_price, live_vwap, cpr_info, fvg, ob, df_option_chain, atr, candle_pattern=None,
                           poc_level=None, volume_is_real=None):
    """
    Combines PDH/PDL/CPR + VWAP + Order Blocks/FVG + real candlestick
    price-action pattern + live Option Chain OI (Rule: heavy Call OI at
    resistance = reversal short; heavy Put OI at support = reversal long
    -- "The 90% Rule") into the Sniper Setup output format: Setup Bias /
    Key Levels / Price Action Confluence / Data Validation / Execution
    Trigger / Trade Plan.

    Every input here is real data already flowing through the app (live
    Upstox/Yahoo price, real VWAP, real FVG/OB from smart_money.py, real
    candlestick pattern shape, real option chain OI) -- nothing here is
    invented.
    """
    if live_price is None:
        return None

    # 1) Setup Bias -- VWAP position + CPR position must agree for a real signal
    vwap_bias = "Bullish" if (live_vwap and live_price > live_vwap) else ("Bearish" if live_vwap else None)

    cpr_bias, cpr_note = None, "CPR data unavailable (need at least 2 trading days of candles)"
    if cpr_info:
        if live_price > cpr_info["cpr_top"]:
            cpr_bias = "Bullish"
            cpr_note = f"Price is ABOVE CPR (Pivot ₹{cpr_info['pivot']}) — trending day bias"
        elif live_price < cpr_info["cpr_bottom"]:
            cpr_bias = "Bearish"
            cpr_note = f"Price is BELOW CPR (Pivot ₹{cpr_info['pivot']}) — trending day bias"
        else:
            cpr_bias = "Neutral"
            cpr_note = f"Price is INSIDE CPR (₹{cpr_info['cpr_bottom']}–₹{cpr_info['cpr_top']}) — range-bound/choppy day likely"

    if vwap_bias and cpr_bias and vwap_bias == cpr_bias:
        setup_bias = vwap_bias
    elif cpr_bias == "Neutral" and vwap_bias:
        setup_bias = vwap_bias
    elif vwap_bias and not cpr_bias:
        setup_bias = vwap_bias
    else:
        setup_bias = "Neutral"  # VWAP and CPR disagree -- honestly no confluence, don't force a bias

    # 2) Key Levels
    pdh_pdl_text = (f"PDH ₹{cpr_info['pdh']} | PDL ₹{cpr_info['pdl']} | CPR ₹{cpr_info['cpr_bottom']}–₹{cpr_info['cpr_top']}"
                     if cpr_info else "PDH/PDL/CPR unavailable")
    vwap_text = f"VWAP ₹{live_vwap:,.2f} ({vwap_bias or 'N/A'})" if live_vwap else "VWAP unavailable"

    # 3) Price Action Confluence -- nearest Order Block/FVG in the bias
    # direction, PLUS whether the actual candlestick shape agrees with it.
    poi, poi_desc = None, "No nearby Order Block/FVG found in the bias direction"
    if setup_bias == "Bullish":
        poi = _nearest_zone(ob, live_price, "below") or _nearest_zone(fvg, live_price, "below")
    elif setup_bias == "Bearish":
        poi = _nearest_zone(ob, live_price, "above") or _nearest_zone(fvg, live_price, "above")
    if poi:
        poi_desc = f"{poi.get('Type')} at ₹{poi.get('Price')}"

    pattern_conflict = False
    if candle_pattern and candle_pattern.get("bias") != "NEUTRAL":
        if setup_bias == "Neutral":
            pass  # no directional bias yet, nothing to compare against
        elif candle_pattern["bias"] == setup_bias.upper():
            poi_desc += f" — CONFIRMED by live candle: {candle_pattern['pattern']}"
        else:
            poi_desc += f" — ⚠️ CONFLICTS with live candle: {candle_pattern['pattern']} is printing the opposite side"
            pattern_conflict = True
    elif candle_pattern:
        poi_desc += f" | Latest candle: {candle_pattern['pattern']}"

    # 4) Option Chain OI Validation -- "The 90% Rule"
    oi_validation, oi_confirms = "Option chain data unavailable", False
    strike_ref = poi.get('Price') if poi else (cpr_info['pivot'] if cpr_info else live_price)
    row = _option_oi_at_strike(df_option_chain, strike_ref)
    if row is not None:
        call_oi = row.get('Call OI', 0) or 0
        put_oi = row.get('Put OI', 0) or 0
        strike = row.get('Strike')
        if setup_bias == "Bearish" and call_oi > put_oi * 1.3:
            oi_validation = f"Heavy CALL WRITING at ₹{strike} (Call OI {call_oi:,.0f} vs Put OI {put_oi:,.0f}) — confirms resistance/reversal"
            oi_confirms = True
        elif setup_bias == "Bullish" and put_oi > call_oi * 1.3:
            oi_validation = f"Heavy PUT WRITING at ₹{strike} (Put OI {put_oi:,.0f} vs Call OI {call_oi:,.0f}) — confirms support/bounce"
            oi_confirms = True
        else:
            oi_validation = f"At ₹{strike}: Call OI {call_oi:,.0f} vs Put OI {put_oi:,.0f} — no strong one-sided writing, confluence weak"

    # 5) Execution Trigger -- now a REAL check against the live POC/Volume
    # Profile level (was previously a static reminder sentence that never
    # actually looked at POC or volume data). Uses an ATR-based "zone"
    # around POC: if price is genuinely close to POC right now, that IS
    # the order-flow confirmation zone to watch for a candle-close +
    # retest before entering; if price is still far from POC, there's
    # honestly no confirmation zone reached yet.
    if poc_level is not None and atr:
        poc_distance = abs(live_price - poc_level)
        near_poc = poc_distance <= (atr * 0.5)
        volume_note = ("real Nifty Futures volume" if volume_is_real
                        else "price-based proxy — no real volume available right now, treat this loosely")
        if near_poc:
            execution_trigger = (
                f"✅ Price (₹{live_price:,.2f}) is right at the POC/max-volume zone (₹{poc_level:,.2f}, "
                f"basis: {volume_note}) — this IS the zone to watch for a candle-close + retest confirmation "
                f"before entering. Don't chase a first touch even here."
            )
        else:
            execution_trigger = (
                f"⏳ Price (₹{live_price:,.2f}) is still {poc_distance:.1f} pts away from the POC/max-volume "
                f"zone (₹{poc_level:,.2f}, basis: {volume_note}) — no order-flow confirmation zone reached yet. "
                f"Wait for price to actually approach POC before treating any level as tradeable."
            )
    else:
        execution_trigger = ("Wait for Order Flow/Volume Profile confirmation at POC before entry — don't chase "
                              "the level on first touch. (POC data unavailable right now.)")

    # 6) Trade Plan -- only produced when EVERY layer agrees (real confluence,
    # including the actual candle shape), otherwise honestly reported as
    # no-trade rather than forcing a plan.
    trade_plan = None
    if setup_bias in ("Bullish", "Bearish") and oi_confirms and atr and not pattern_conflict:
        if setup_bias == "Bullish":
            entry = live_price
            sl = (poi.get('Price') - atr * 0.5) if poi else (live_price - atr)
            target = cpr_info['pdh'] if (cpr_info and cpr_info['pdh'] > live_price) else (live_price + atr * 2)
        else:
            entry = live_price
            sl = (poi.get('Price') + atr * 0.5) if poi else (live_price + atr)
            target = cpr_info['pdl'] if (cpr_info and cpr_info['pdl'] < live_price) else (live_price - atr * 2)
        trade_plan = {"entry": round(entry, 2), "target": round(target, 2), "sl": round(sl, 2)}

    return {
        "setup_bias": setup_bias,
        "key_levels": f"{pdh_pdl_text} | {vwap_text}",
        "cpr_note": cpr_note,
        "price_action_confluence": poi_desc,
        "candle_pattern": candle_pattern,
        "pattern_conflict": pattern_conflict,
        "oi_validation": oi_validation,
        "oi_confirms": oi_confirms,
        "execution_trigger": execution_trigger,
        "trade_plan": trade_plan,
    }
