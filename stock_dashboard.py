from app_logging import get_logger
logger = get_logger(__name__)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _money(x):
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return "N/A"


def _pct(x):
    try:
        return f"{float(x):+.2f}%"
    except Exception:
        return "N/A"


def _safe_df(rows, cols=None):
    if not rows:
        return pd.DataFrame(columns=cols or [])
    return pd.DataFrame(rows)


def _option_calc_df(option_df):
    """Adapt the selected-stock option schema to the common option helpers."""
    if option_df is None or option_df.empty:
        return pd.DataFrame(columns=["Strike", "Call OI", "Put OI", "PCR", "IV (%)", "Type"])
    d = option_df.copy()
    if "CE OI" in d.columns:
        d["Call OI"] = pd.to_numeric(d["CE OI"], errors="coerce").fillna(0)
    if "PE OI" in d.columns:
        d["Put OI"] = pd.to_numeric(d["PE OI"], errors="coerce").fillna(0)
    return d


def _render_chart(r):
    df = r.get("df")
    if df is None or df.empty:
        st.info("Stock-specific chart data unavailable.")
        return
    d = df.tail(180).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=r["symbol"]
    ))
    for col, name in [("EMA_20", "EMA 20"), ("EMA_50", "EMA 50"), ("VWAP", "VWAP")]:
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=name))
    if "POC_Level" in d.columns:
        fig.add_hline(y=float(d["POC_Level"].iloc[-1]), line_dash="dot", annotation_text="POC")
    sr = r.get("sr") or {}
    ns = sr.get("nearest_support")
    nr = sr.get("nearest_resistance")
    if isinstance(ns, dict):
        ns = ns.get("price") or ns.get("low")
    if isinstance(nr, dict):
        nr = nr.get("price") or nr.get("high")
    if ns is not None:
        fig.add_hline(y=float(ns), line_dash="dash", annotation_text="Support")
    if nr is not None:
        fig.add_hline(y=float(nr), line_dash="dash", annotation_text="Resistance")
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        title=f"{r['symbol']} — Institutional Price / Structure Chart"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_stock_dashboard(r):
    """Full-screen selected-stock dashboard.

    This intentionally mirrors the NIFTY dashboard's section order, cards,
    expanders and terminology, while every asset-specific number comes from
    the selected Upstox equity. NIFTY price/options/futures/breadth are never
    substituted for missing stock data.
    """
    symbol = r["symbol"]
    ai = r.get("ai") or {}
    ml = r.get("ml") or {}
    quote = r.get("quote") or {}
    opt = r.get("option_result") or {}
    option_df = r.get("option_df")
    fund = r.get("fundamentals") or {}
    regime = r.get("regime") or {}
    sr = r.get("sr") or {}
    mtf = r.get("mtf") or {}
    smc = r.get("smc") or {}
    liquidity = r.get("liquidity") or {}
    pivots = r.get("pivots") or {}
    orb = r.get("opening_range") or {}
    lp = r.get("level_prediction") or {}
    ladder = r.get("ladder") or {}
    df = r.get("df")
    live = float(r.get("live_price") or 0)

    # Same visual language as the NIFTY dashboard, but with a selected-stock header.
    st.markdown("""
    <style>
    .stock-mode-banner { padding: 10px 14px; border-radius: 8px; margin: 4px 0 14px 0;
        border: 1px solid rgba(80,180,120,.35); background: rgba(80,180,120,.08); }
    .stock-mode-banner b { font-size: 1.05rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"# ⚡ {symbol} Institutional AI Trading Dashboard (Pro Edition)")
    st.markdown(
        f'<div class="stock-mode-banner">🟢 <b>STOCK MODE ACTIVE — {symbol}</b> | '
        f'Upstox Live Equity | Instrument: <code>{r.get("instrument_key")}</code> | '
        f'Last candle: {r.get("last_candle")}</div>', unsafe_allow_html=True)

    if not r.get("data_quality", {}).get("nifty_data_used", True):
        st.success("🔒 DATA ISOLATION ON — is screen ke asset-specific calculations mein NIFTY price/options/futures/depth/breadth ko substitute nahi kiya gaya.")

    # ------------------------------------------------------------------
    # TOP CARDS — same shape as NIFTY
    # ------------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Real-Time Live {symbol} Price (LTP)", _money(live))
    c2.metric("Institutional Confluence Signal", r.get("final_signal", "NO TRADE"))
    c3.metric("AI Confluence Score (live heuristic)", f"{ai.get('confidence_pct', 0)}%")

    if r.get("data_quality", {}).get("price_volume"):
        st.caption(f"Data source: {r['data_quality']['price_volume']} | All asset-specific modules use {symbol} data only.")

    # Chart is deliberately stock-only.
    _render_chart(r)

    # ------------------------------------------------------------------
    # EARLY WARNING — exact NIFTY section concept
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Early Warning: Support/Resistance Approach Predictor (ICT)")
    st.caption("Ye level ko TOUCH hone se pehle fire hota hai — selected stock ke support/resistance ke paas pahunchte hi break ya bounce probability dikhata hai.")
    if lp.get("status") == "APPROACHING_LEVEL":
        k = lp.get("level_type", "Level")
        level = lp.get("level_price")
        dist = lp.get("distance_pts")
        w1, w2, w3 = st.columns(3)
        w1.metric(f"Approaching {k}", _money(level), delta=f"{dist:.2f} pts away" if dist is not None else None)
        w2.metric("Break-Through Probability", f"{lp.get('break_pct', 50)}%")
        w3.metric("Bounce / Reject Probability", f"{lp.get('bounce_pct', 50)}%")
        strength = lp.get("level_strength_pct")
        if strength is not None:
            label = "STRONG" if strength >= 75 else ("MODERATE" if strength >= 55 else "WEAK")
            st.caption(f"🛡️ **Structural Level Strength: {strength:.1f}% ({label})** | Break/Hold % is a live model estimate, not a guarantee.")
        bias = str(lp.get("directional_bias", ""))
        if "🟢" in bias:
            st.success(f"**Early Read:** {bias}")
        else:
            st.error(f"**Early Read:** {bias or 'Neutral'}")
        with st.expander("🔍 Confluence Factors (ICT + Momentum + Volume)", expanded=False):
            for factor in lp.get("factors", []):
                st.write(f"- {factor}")
    else:
        st.info(f"Price abhi key approach-zone me nahi hai. Nearest Support: {_money(lp.get('nearest_support'))} | Nearest Resistance: {_money(lp.get('nearest_resistance'))}")

    # ------------------------------------------------------------------
    # PHASE 1 — Market Regime + CPR/Pivot + Liquidity
    # ------------------------------------------------------------------
    with st.expander("🧭 Market Regime + CPR/Pivot + Liquidity Map (Phase 1)", expanded=False):
        st.markdown(f"**Regime:** {regime.get('primary','N/A')} | **Volatility:** {regime.get('volatility','N/A')} | **Structure:** {regime.get('structure','N/A')}")
        if regime.get("adx") is not None:
            st.caption(f"ADX {regime.get('adx')} (+DI {regime.get('plus_di')} / -DI {regime.get('minus_di')}) | ATR percentile {regime.get('atr_percentile')} | VIX: market-wide context only")
        if r.get("regime_guidance"):
            st.write(r["regime_guidance"])
        st.markdown("---")
        if pivots.get("status") == "OK":
            st.markdown(f"**Today's Pivot:** {_money(pivots.get('pivot'))} | **CPR:** {_money(pivots.get('bc'))} – {_money(pivots.get('tc'))} ({pivots.get('cpr_label','N/A')}) | **Position:** {r.get('cpr_position','N/A')}")
            st.caption(f"R1 {_money(pivots.get('r1'))} | R2 {_money(pivots.get('r2'))} | R3 {_money(pivots.get('r3'))} · S1 {_money(pivots.get('s1'))} | S2 {_money(pivots.get('s2'))} | S3 {_money(pivots.get('s3'))} · PDH {_money(pivots.get('pdh'))} | PDL {_money(pivots.get('pdl'))} | PDC {_money(pivots.get('pdc'))}")
        else:
            st.caption("Pivots: not enough selected-stock prior-day data yet.")
        if orb.get("status") == "OK":
            st.markdown(f"**Opening Range ({orb.get('orb_minutes')}m):** {_money(orb.get('orb_low'))} – {_money(orb.get('orb_high'))} → {orb.get('read')}")
        st.markdown("---")
        if liquidity.get("levels"):
            nearest = min(liquidity["levels"], key=lambda x: x.get("distance_pts", 1e9))
            st.markdown(f"**Liquidity:** Nearest target {nearest.get('name')} at {_money(nearest.get('price'))} ({nearest.get('distance_pts')} pts {nearest.get('side')})")
            for lv in liquidity["levels"][:6]:
                st.write(f"- {lv.get('name')}: {_money(lv.get('price'))} ({lv.get('distance_pts')} pts {lv.get('side')})")
        else:
            st.caption("Selected-stock liquidity map unavailable.")
        if liquidity.get("equal_highs"):
            st.caption("Equal Highs: " + ", ".join(f"{_money(e.get('level'))} ({e.get('touches')}x)" for e in liquidity["equal_highs"][:3]))
        if liquidity.get("equal_lows"):
            st.caption("Equal Lows: " + ", ".join(f"{_money(e.get('level'))} ({e.get('touches')}x)" for e in liquidity["equal_lows"][:3]))

    # ------------------------------------------------------------------
    # PHASE 2 — options + calibration
    # ------------------------------------------------------------------
    with st.expander("🔬 Deep Option Chain + Stock Context + Model Calibration (Phase 2)", expanded=False):
        st.markdown("**IV Skew & Percentile**")
        if r.get("iv_skew"):
            x = r["iv_skew"]
            st.write(f"Downside ({_money(x.get('downside_strike'))}) IV: {x.get('downside_iv')}% | Upside ({_money(x.get('upside_strike'))}) IV: {x.get('upside_iv')}% | Skew: {x.get('skew')}")
            st.caption(x.get("read", ""))
        else:
            st.caption("Selected-stock IV skew unavailable — no live stock option IV surface available.")
        if r.get("iv_percentile"):
            x = r["iv_percentile"]
            st.write(f"Current stock option IV proxy: {x.get('current_iv')}% | Percentile: {x.get('percentile')}%")
            st.caption(x.get("read", ""))
        else:
            st.caption("IV percentile unavailable. No market-wide VIX value is substituted as a stock-specific IV percentile.")
        st.markdown("---")
        st.markdown("**Stock Option Chain / OI / PCR**")
        if opt.get("status") == "AVAILABLE" and option_df is not None and not option_df.empty:
            st.success(f"Live {symbol} option chain: {opt.get('expiry','N/A')} | Source: {opt.get('source','Upstox')}")
            st.dataframe(option_df, hide_index=True, use_container_width=True)
            st.caption(f"Stock option Max Pain: {_money(r.get('max_pain'))} | Stock PCR: {r.get('stock_pcr','N/A')}")
        else:
            st.warning(f"{symbol} ke liye live option-chain available nahi hai. NIFTY option data substitute nahi kiya gaya.")
        st.markdown("---")
        st.markdown("**Model Confidence Calibration**")
        st.write(f"Raw ensemble confidence: {ml.get('latest_confidence', 0.0)*100:.1f}%")
        if ml.get("calibrated_confidence") is not None:
            st.write(f"Calibrated confidence: {ml['calibrated_confidence']*100:.1f}%")
        st.caption(ml.get("calibration_note", "Stock-specific walk-forward model calibration."))

    # ------------------------------------------------------------------
    # PHASE 3 — stock order flow + stock historical ML/backtest
    # ------------------------------------------------------------------
    with st.expander("📉 Stock Order Flow + Historical ML + Drift Monitor (Phase 3)", expanded=False):
        st.markdown(f"**Order Book Imbalance (Live, {symbol} Equity Depth)**")
        if quote.get("status") == "AVAILABLE":
            st.write(f"{quote.get('pressure','N/A')} — Imbalance: {quote.get('imbalance_pct',0):+.1f}%")
            st.caption(f"Resting Buy Qty: {quote.get('total_buy_qty',0):,.0f} | Resting Sell Qty: {quote.get('total_sell_qty',0):,.0f} | Best Bid: {_money(quote.get('best_bid'))} | Best Ask: {_money(quote.get('best_ask'))} | Spread: {quote.get('spread','N/A')}")
            st.caption("Ye selected stock ka real top-5 Upstox equity depth hai; NIFTY futures depth use nahi hota.")
        else:
            st.warning("Selected-stock live depth unavailable; flow score fabricated nahi kiya gaya.")

        st.markdown("---")
        st.markdown(f"**Historical ML Walk-Forward Test ({symbol} OHLCV)**")
        if ml:
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Model Ready", "YES" if ml.get("model_ready") else "NO")
            m2.metric("OOS Signal", {1:"BUY",-1:"SELL",0:"FLAT"}.get(ml.get("latest_signal"),"FLAT"))
            m3.metric("Win Rate", ml.get("Win Rate","N/A"))
            m4.metric("Calibrated Confidence", f"{ml['calibrated_confidence']*100:.1f}%" if ml.get("calibrated_confidence") is not None else "N/A")
            st.caption(ml.get("calibration_note", ""))
        else:
            st.caption("Stock-specific ML history unavailable.")
        st.markdown("---")
        st.markdown("**Backtest / Monte Carlo / Drift**")
        st.info("In modules ke liye selected-stock historical trades alag se log hone chahiye. NIFTY ke logged trades ko stock performance ke roop mein display nahi kiya jaata.")

    # ------------------------------------------------------------------
    # PHASE 4 — MTF + position sizing + risk
    # ------------------------------------------------------------------
    with st.expander("🧮 Multi-Timeframe Structure + Position Sizing + Risk Engine (Phase 4)", expanded=False):
        st.markdown(f"**Multi-Timeframe Structure ({symbol} — real candles)**")
        rows=[]
        for lbl in ["1D","4H","1H","30M","15M","5M"]:
            x=mtf.get("timeframes",{}).get(lbl,{})
            if x.get("status")=="OK":
                rows.append({"Timeframe":lbl,"Trend":x.get("trend"),"Structure":x.get("structure"),"Last Close":x.get("last_close")})
            else:
                rows.append({"Timeframe":lbl,"Trend":"DATA UNAVAILABLE","Structure":"—","Last Close":"—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.write(f"**Alignment:** {mtf.get('alignment','N/A')}")
        st.markdown("---")
        st.markdown("**Position Sizing Calculator**")
        pc1,pc2=st.columns(2)
        capital=pc1.number_input("Trading Capital (₹)",min_value=1000.0,value=float(st.session_state.get("risk_capital",100000.0)),step=1000.0,key="stock_risk_capital")
        riskpct=pc2.number_input("Risk per Trade (%)",min_value=0.1,max_value=10.0,value=float(st.session_state.get("risk_pct",1.0)),step=0.1,key="stock_risk_pct")
        entry=float(ai.get("entry_price",live)); sl=float(ai.get("stop_loss",live))
        distance=abs(entry-sl)
        qty=int((capital*riskpct/100)/distance) if distance>0 else 0
        st.write(f"Quantity: **{qty:,}** | SL Distance: **{distance:.2f}** pts | Approx risk: **₹{qty*distance:,.0f}**")
        st.caption("Equity quantity is calculated for the selected stock; NIFTY lot size is not used.")
        st.markdown("---")
        st.markdown("**Risk Engine**")
        st.write("Status: **STOCK-SPECIFIC CONTEXT**")
        st.caption("Portfolio/trade-log risk state is not borrowed from NIFTY trades.")

    # ------------------------------------------------------------------
    # Full S/R ladder
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📐 Full Support/Resistance Ladder Calculator (har 50-point level)", expanded=False):
        st.caption(f"Same ICT + price-action + volume + option-confluence ladder, but calculated from {symbol} candles and {symbol} options only.")
        lc, rc = st.columns(2)
        with rc:
            st.markdown(f"**🔴 Resistances above {_money(live)}**")
            rows=[{"Level":_money(x.get("level_price")),"Away":f"{x.get('distance_pts',0):.1f} pts","Break %":f"{x.get('break_pct',0)}%","Bounce %":f"{x.get('bounce_pct',0)}%","Read":str(x.get('directional_bias','')).replace(' 🟢','').replace(' 🔴','')} for x in ladder.get("resistances",[])]
            st.dataframe(_safe_df(rows),hide_index=True,use_container_width=True) if rows else st.caption("Ladder unavailable.")
        with lc:
            st.markdown(f"**🟢 Supports below {_money(live)}**")
            rows=[{"Level":_money(x.get("level_price")),"Away":f"{x.get('distance_pts',0):.1f} pts","Break %":f"{x.get('break_pct',0)}%","Bounce %":f"{x.get('bounce_pct',0)}%","Read":str(x.get('directional_bias','')).replace(' 🟢','').replace(' 🔴','')} for x in ladder.get("supports",[])]
            st.dataframe(_safe_df(rows),hide_index=True,use_container_width=True) if rows else st.caption("Ladder unavailable.")

    # ------------------------------------------------------------------
    # AI decision engine — same prominent section
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧠 AI Trade Decision Engine (Self-Learning)")
    st.caption(f"Complete confluence is calculated for **{symbol}**. NIFTY-specific price/option/breadth/futures inputs are excluded from the asset score.")
    a1,a2,a3,a4=st.columns(4)
    a1.metric("Bias",ai.get("bias_text","N/A"))
    a2.metric("Confidence",f"{ai.get('confidence_pct',0)}%")
    a3.metric("ML Signal",{1:"BUY",-1:"SELL",0:"FLAT"}.get(ml.get("latest_signal"),"FLAT"))
    a4.metric("Final Decision",r.get("final_signal","NO TRADE"))
    st.info(
        f"**Technical:** {ai.get('tech_score','N/A')} | **SMC:** {ai.get('smc_score','N/A')} | "
        f"**Macro:** {ai.get('macro_score','N/A')} | **Stock Order Flow:** {ai.get('flow_score','N/A')} | "
        f"**Action:** {ai.get('signal_type','NO TRADE')}"
    )
    tm=ai.get("tech_metrics") or {}
    st.dataframe(pd.DataFrame([
        ["RSI",tm.get("rsi")],["EMA 20",_money(tm.get("ema_20"))],["EMA 50",_money(tm.get("ema_50"))],
        ["VWAP",_money(tm.get("vwap"))],["ATR",tm.get("atr")],["PCR",tm.get("avg_pcr","N/A")]
    ],columns=["Technical Factor",f"{symbol} Read"]),hide_index=True,use_container_width=True)

    st.markdown("### 💰 Execution Plan")
    e1,e2,e3,e4=st.columns(4)
    e1.metric("Entry",_money(ai.get("entry_price"))); e2.metric("Stop Loss",_money(ai.get("stop_loss"))); e3.metric("Target 1",_money(ai.get("target_1"))); e4.metric("Target 2",_money(ai.get("target_2")))
    st.caption(f"Invalidation: {_money(ai.get('invalidation_level'))} | Risk/Reward: {ai.get('risk_reward_ratio','N/A')} | Entry type: {ai.get('entry_type','N/A')}")

    # ------------------------------------------------------------------
    # Institutional Order Flow chart + SMC / ICT — same visual section family
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"📊 Institutional Order Flow Chart ({symbol} — VWAP & Volume Profile)")
    st.caption(f"Price, VWAP and volume profile are computed from {symbol} equity candles only.")
    if df is not None and not df.empty:
        vol=d.tail(100)[["Close","Volume"]].copy()
        st.dataframe(vol.tail(20),hide_index=False,use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Sniper Setup (SMC + Option Chain OI + PDH/PDL/CPR Confluence)")
    st.markdown(f"**Setup Bias:** {ai.get('bias_text','N/A')}")
    st.markdown(f"**Key Levels (Sniper Zone):** PDH {_money(pivots.get('pdh'))} | PDL {_money(pivots.get('pdl'))} | CPR {_money(pivots.get('bc'))}–{_money(pivots.get('tc'))} | VWAP {_money(tm.get('vwap'))}")
    st.markdown(f"**Price Action Confluence:** {smc.get('candle','N/A')} | Structure: {smc.get('event','N/A')}")
    st.markdown(f"**Option OI:** {r.get('stock_pcr','N/A')} PCR | Max Pain: {_money(r.get('max_pain'))}")
    st.markdown(f"**Trade Plan:** {r.get('final_signal','NO TRADE')} | Entry {_money(ai.get('entry_price'))} | Target {_money(ai.get('target_1'))} | SL {_money(ai.get('stop_loss'))}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏛️ Institutional F&O Footprint & Max Pain Analytics")
    if option_df is not None and not option_df.empty:
        st.write(f"Selected stock PCR: **{r.get('stock_pcr','N/A')}** | Max Pain: **{_money(r.get('max_pain'))}**")
        st.dataframe(option_df,hide_index=True,use_container_width=True)
    else:
        st.caption("Selected-stock F&O footprint unavailable. No NIFTY option chain is displayed here.")

    # ------------------------------------------------------------------
    # Technical + SMC details
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧱 Advanced Technical Indicators Table")
    if df is not None and not df.empty:
        last=df.iloc[-1]
        indicators_rows=[]
        for col in ["Close","EMA_20","EMA_50","RSI","MACD","MACD_Signal","MACD_Hist","ATR","VWAP","POC_Level"]:
            if col in df.columns:
                indicators_rows.append({"Indicator":col,"Value":last.get(col)})
        st.dataframe(pd.DataFrame(indicators_rows),hide_index=True,use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏦 Smart Money Concepts & Market Structure (BOS / CHoCH)")
    st.markdown(f"**Market Event:** {smc.get('event','N/A')}")
    st.markdown(f"**Latest Candle:** {smc.get('candle','N/A')}")
    with st.expander("📌 Fair Value Gaps (FVG)",expanded=False):
        st.dataframe(_safe_df(smc.get("fvg") or []),hide_index=True,use_container_width=True) if smc.get("fvg") else st.caption("No selected-stock FVG detected.")
    with st.expander("🧱 Order Blocks (OB)",expanded=False):
        st.dataframe(_safe_df(smc.get("order_blocks") or []),hide_index=True,use_container_width=True) if smc.get("order_blocks") else st.caption("No selected-stock order block detected.")
    with st.expander("🌊 Liquidity Sweeps",expanded=False):
        st.dataframe(_safe_df(smc.get("liquidity_sweeps") or []),hide_index=True,use_container_width=True) if smc.get("liquidity_sweeps") else st.caption("No selected-stock liquidity sweep detected.")

    # ------------------------------------------------------------------
    # Fundamentals + news: selected company only
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"🏢 {symbol} Fundamentals + Company Research")
    if fund.get("status") == "AVAILABLE":
        st.caption(f"Source: {fund.get('source','N/A')} | Sector: {fund.get('sector',fund.get('industry','N/A'))} | ISIN: {r.get('isin','N/A')}")
        ratio_map=fund.get("ratio_map") or {}
        if ratio_map:
            st.dataframe(pd.DataFrame([{"Ratio":k,"Value":v.get('value') if isinstance(v,dict) else v} for k,v in list(ratio_map.items())[:30]]),hide_index=True,use_container_width=True)
        for key,label in [("company_profile","Company Profile"),("income_statement","Income Statement"),("balance_sheet","Balance Sheet"),("cash_flow","Cash Flow"),("shareholdings","Shareholding")]:
            val=fund.get(key)
            if val:
                with st.expander(label,expanded=False):
                    if isinstance(val,dict):
                        st.json(val)
                    elif isinstance(val,list):
                        st.dataframe(pd.DataFrame(val),hide_index=True,use_container_width=True)
    else:
        st.warning(f"{symbol} stock-specific fundamentals unavailable. NIFTY/company substitute nahi dikhaya gaya.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"📰 {symbol} Stock-specific News")
    if r.get("news"):
        for n in r["news"][:10]:
            st.write(f"• **{n.get('title','')}** — {n.get('published','')} {(' | '+n.get('source','')) if n.get('source') else ''}")
    else:
        st.caption("No stock-specific news feed available.")

    # ------------------------------------------------------------------
    # Audit: makes the isolation explicit rather than silently mixing data.
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔍 Data Quality / Source Audit")
    for k,v in (r.get("data_quality") or {}).items():
        st.write(f"**{k}:** {v}")
