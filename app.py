import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Import custom modules
import database
import upstox_auth
import market_data
import option_chain
import global_news
import global_markets
import smart_money
import market_breadth
import index_correlation
import ml_engine
import support_resistance
import market_status
from hybrid_ai_engine import HybridAIEngine
from alert_manager import AlertManager
from ai_chat import render_ai_chat

# Page Configuration
st.set_page_config(page_title="Nifty 50 Real-Time AI Predictor", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 1rem !important;
        }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.1rem !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ GEMINI_API_KEY not found. Please set it in your secrets.toml file or Streamlit Cloud dashboard.")
    st.stop()

ai_engine = HybridAIEngine(api_key=GEMINI_API_KEY)

try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    alert_sys = AlertManager(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
except Exception:
    alert_sys = None

# Initialize the trades database (creates table if it doesn't exist yet)
database.init_db()

st.title("⚡ Nifty 50 Institutional AI Trading Dashboard (Pro Edition)")
st.caption("⚠️ Research/paper-trading tool. Signals are model outputs, not financial advice. "
           "See 'Real Performance Tracking' below for actual measured accuracy before trusting any signal.")
st.markdown("---")

# -------------------------------------------------------------
# LIVE DASHBOARD -- runs as an independent st.fragment that refreshes
# itself every 30s WITHOUT rerunning the rest of the script. This is
# what fixes two real bugs: (1) any audio the chat was playing used to
# get cut off every 30s because the whole page (including the <audio>
# element) was being torn down and rebuilt by the old autorefresh; and
# (2) text you were mid-typing into the chat box used to get wiped if
# a refresh landed while you were typing, because that chat box lived
# inside the same auto-refreshing scope. Moving the chat OUTSIDE this
# fragment (see bottom of file) means it now only re-renders when you
# actually interact with it -- never on the 30s timer.
# -------------------------------------------------------------
@st.fragment(run_every=30)
def _live_dashboard():
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None

    # Step 1: Login & Get Broker Token
    access_token = upstox_auth.get_upstox_access_token()

    # Step 2: Fetch Live Market Data
    df, model, feature_cols, live_price = market_data.fetch_live_market_data(access_token)

    if df is not None and not df.empty:

        # -------------------------------------------------------------
        # 🕒 NEW: REAL MARKET STATE + DATA FRESHNESS CHECK
        # Fixes the exact "Snapshot vs Market Status disconnect" bug —
        # the dashboard used to have no idea whether NSE was actually
        # open, and no way to tell fresh data from a frozen/cached feed.
        # Shown FIRST, before anything else, so it's the first thing you
        # see. Purely informational — doesn't change any fetch/score logic.
        # -------------------------------------------------------------
        mkt_status = market_status.get_market_status()
        last_candle_time = df.index[-1] if len(df.index) > 0 else None
        data_freshness = market_status.check_data_freshness(last_candle_time, mkt_status['state'])

        status_line = f"{mkt_status['label']} | IST Time: {mkt_status['now_ist'].strftime('%d-%b-%Y %H:%M:%S')}"
        if last_candle_time is not None:
            status_line += f" | Last Candle: {last_candle_time.strftime('%d-%b-%Y %H:%M:%S')}"

        if data_freshness['is_stale']:
            st.error(f"⚠️ **DATA STALE/FROZEN** — {data_freshness['note']} | {status_line}")
        elif mkt_status['state'] == 'OPEN':
            st.success(f"{status_line} ✅ Data looks fresh")
        else:
            st.info(f"{status_line} — market band hai, ye last traded price hai, live tick nahi.")

        # -------------------------------------------------------------
        # VWAP & POC CALCULATION (unchanged from original — was correct)
        # -------------------------------------------------------------
        try:
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                df['Cumulative_Volume'] = df['Volume'].cumsum()
                df['Cumulative_PV'] = (df['Typical_Price'] * df['Volume']).cumsum()
                df['VWAP'] = df['Cumulative_PV'] / df['Cumulative_Volume']
            else:
                df['VWAP'] = df['Typical_Price'].rolling(window=14, min_periods=1).mean()

            if 'Volume' in df.columns and df['Volume'].sum() > 0:
                vp_df = df[['Close', 'Volume']].dropna()
                vp_df['Price_Bin'] = pd.cut(vp_df['Close'], bins=50)
                volume_profile = vp_df.groupby('Price_Bin')['Volume'].sum().reset_index()
                volume_profile['Mid_Price'] = volume_profile['Price_Bin'].apply(lambda x: x.mid)
                poc_index = volume_profile['Volume'].idxmax()
                df['POC_Level'] = float(volume_profile.loc[poc_index, 'Mid_Price'])
            else:
                df['Rounded_Close'] = (df['Close'] / 5).round() * 5
                poc_level_proxy = df['Rounded_Close'].mode()[0]
                df['POC_Level'] = float(poc_level_proxy)
                df.drop('Rounded_Close', axis=1, inplace=True)
        except Exception:
            df['VWAP'] = df['EMA_20'] if 'EMA_20' in df.columns else df['Close']
            df['POC_Level'] = df['Close'].iloc[-1]

        # Fetch Supporting Data for Confluence Engine
        df_option_chain, oc_source = option_chain.generate_option_chain_data(live_price)
        fvg, ob, sweeps = smart_money.detect_smc_zones(df)
        market_structure = smart_money.detect_market_structure(df)
        smc_event = market_structure[0]["Market Event"] if market_structure else "Neutral Structure"

        df_heavyweights, advances, declines, breadth_ratio, breadth_status, heavyweights_debug = market_breadth.get_nifty_internal_breadth(access_token)
        df_full50, adv50, dec50, ratio50, breadth50_status, full50_debug = market_breadth.get_full_nifty50_breadth(access_token)
        max_pain = option_chain.calculate_max_pain(df_option_chain)
        fii_footprint, current_pcr = option_chain.get_fii_dii_fo_footprint(df_option_chain)

        # Bank Nifty / Sensex -- the biggest cross-index drivers of Nifty option
        # sentiment. Fetched Upstox-first (live) with Yahoo Finance fallback.
        nifty_change_pct = index_correlation.get_nifty_change_pct(live_price)
        index_correlation_data, banknifty_correlation_note = index_correlation.get_bank_nifty_sensex(
            access_token, nifty_change_pct
        )

        df_global_sentiment, top_headline = global_news.get_global_market_sentiment()
        df_global_markets = global_markets.get_global_market_indices()
        global_sentiment_score, avg_market_change = global_markets.get_global_market_summary(df_global_markets)
        live_vix = global_markets.get_live_vix(df_global_markets)  # NEW: real VIX, not hardcoded

        # -------------------------------------------------------------
        # REAL, WALK-FORWARD VALIDATED MODEL (replaces the unvalidated
        # retrain-every-30-seconds XGBoost signal from before)
        # -------------------------------------------------------------
        ml_results = ml_engine.train_and_backtest(df, live_vix=live_vix)

        # -------------------------------------------------------------
        # VOLATILITY REGIME FILTER (unchanged — was working correctly)
        # -------------------------------------------------------------
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

        avg_bb_width = df['BB_Width'].rolling(window=50).mean().iloc[-1]
        current_bb_width = df['BB_Width'].iloc[-1]
        is_choppy = current_bb_width < (avg_bb_width * 0.75) if pd.notna(avg_bb_width) and avg_bb_width > 0 else False

        # -------------------------------------------------------------
        # HYBRID AI ENGINE — dynamic ATR-based SL/Target (kept consistent
        # everywhere, instead of a hardcoded 50/100-point trade elsewhere)
        # -------------------------------------------------------------
        ai_analysis = ai_engine.analyze(
            live_price=live_price,
            df=df,
            df_option_chain=df_option_chain,
            smc_data=smc_event,
            sentiment_score=global_sentiment_score,
            global_avg_change=avg_market_change,
            fii_footprint=fii_footprint,
            breadth_status=breadth_status,
            is_choppy=is_choppy,
            banknifty_correlation_note=banknifty_correlation_note,
            full_breadth_status=breadth50_status,
            banknifty_sensex_data=index_correlation_data
        )

        # -------------------------------------------------------------
        # 🎯 NEW: EARLY WARNING — SUPPORT/RESISTANCE APPROACH PREDICTOR (ICT)
        # Predicts BEFORE price touches a support/resistance level whether it
        # will more likely BREAK through or BOUNCE off it, using ICT concepts
        # (liquidity sweep, order block/FVG confluence, premium/discount) plus
        # momentum + volume exhaustion reads. Purely additive — does NOT read
        # from or modify net_score / signal_code / entry-SL-target logic below.
        # -------------------------------------------------------------
        level_prediction = support_resistance.predict_level_reaction(
            df=df, live_price=live_price, atr=ai_analysis['tech_metrics'].get('atr', 15.0),
            fvg_list=fvg, ob_list=ob, trend_bias=ai_analysis.get('tech_score', 0.0)
        )
        # Attach it onto ai_analysis (new key only) so the Deep Reasoning Report
        # below can read + confirm/deny it too — doesn't touch net_score/signal.
        ai_analysis['level_prediction'] = level_prediction

        signal_code = 1 if "BULLISH" in ai_analysis['bias_text'] else (-1 if "BEARISH" in ai_analysis['bias_text'] else 0)
        # Require the validated ML model to agree before treating this as actionable
        if ml_results.get("model_ready") and signal_code != 0 and ml_results.get("latest_signal", 0) != signal_code:
            signal_code = 0  # ML model disagrees with confluence engine — stay flat rather than force a signal

        final_signal_text = ai_analysis['signal_type'] if signal_code != 0 else "NO TRADE (ML model disagreement or choppy)"

        entry_price = ai_analysis['entry_price']
        sl = ai_analysis['stop_loss']
        t1 = ai_analysis['target_1']
        t2 = ai_analysis['target_2']
        intraday_atr = ai_analysis['tech_metrics'].get('atr', 15.0) * 0.4

        # -------------------------------------------------------------
        # AUTOMATIC TRADE ENTRY — REMOVED (per user request).
        # The system no longer auto-logs a BUY/SELL paper trade to the DB
        # whenever confluence + ML agree. It now only DISPLAYS the signal
        # (Institutional Confluence Signal, Model Decision, Execution Plan
        # below) -- taking a trade, if you choose to, is entirely manual.
        #
        # Why this changed: previously, once a trade was auto-logged it kept
        # riding until its own Stop Loss or Target hit, even if the signal
        # flipped to "NO TRADE" a few refreshes later (confidence and signal
        # are recalculated fresh every 30s from the CURRENT candle, while an
        # already-open trade intentionally does not exit early just because
        # the live signal changed -- that's standard trade management, not a
        # bug). But it looked confusing on screen: an "ACTIVE TRADE" box
        # staying up while the signal above it said "NO TRADE." Removing
        # auto-entry avoids that entirely.
        #
        # Any trade already open in the database from BEFORE this change
        # will still be tracked to its SL/Target/time-exit by the line
        # below, so past trades resolve properly -- this only stops NEW
        # auto-entries from being created.
        # -------------------------------------------------------------
        database.check_and_update_open_trades(live_price)
        open_trade = database.check_open_position()

        if open_trade is not None:
            # Trailing stop-loss management on the DB-tracked open trade
            # (only applies to a trade opened before this change, until it
            # resolves)
            is_buy = "BUY" in open_trade["signal"].upper()
            entry = open_trade["entry_price"]
            if is_buy:
                if live_price >= entry + (1.0 * intraday_atr) and open_trade["stop_loss"] < entry:
                    database.update_stop_loss(open_trade["id"], entry)
                if live_price >= entry + (2.5 * intraday_atr) and open_trade["stop_loss"] < entry + intraday_atr:
                    database.update_stop_loss(open_trade["id"], entry + (1.0 * intraday_atr))
            else:
                if live_price <= entry - (1.0 * intraday_atr) and open_trade["stop_loss"] > entry:
                    database.update_stop_loss(open_trade["id"], entry)
                if live_price <= entry - (2.5 * intraday_atr) and open_trade["stop_loss"] > entry - intraday_atr:
                    database.update_stop_loss(open_trade["id"], entry - (1.0 * intraday_atr))

        perf = database.fetch_performance_metrics()

        # -------------------------------------------------------------
        # CORE METRICS — fake "Accuracy Score" renamed, real metrics added
        # -------------------------------------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Real-Time Live Nifty Price (LTP)", value=f"₹ {live_price:,.2f}")
        with col2:
            st.metric(label="Institutional Confluence Signal", value=final_signal_text)
        with col3:
            # RENAMED from "AI Model Accuracy Score" — this is a live confidence
            # heuristic based on current indicators, NOT historical accuracy.
            st.metric(label="AI Confluence Score (live heuristic)", value=f"{ai_analysis['confidence_pct']}%")

        if open_trade:
            st.warning(f"🛡️ **ACTIVE TRADE (DB-tracked, from before auto-entry was disabled)** | {open_trade['signal']} | "
                       f"Entry: ₹{open_trade['entry_price']:.2f} | Current SL: ₹{open_trade['stop_loss']:.2f} | Target: ₹{open_trade['target_1']:.2f}")
        else:
            st.info("ℹ️ No auto-trade is taken. Review the signal and Execution Plan below and decide manually.")

        # -------------------------------------------------------------
        # 🎯 NEW: EARLY WARNING — SUPPORT/RESISTANCE APPROACH PREDICTOR (ICT)
        # Fires BEFORE price touches a level, so it doesn't lag like the
        # confirmation signal above (which only confirms after a reaction has
        # already happened). Purely additive — doesn't change any section above.
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎯 Early Warning: Support/Resistance Approach Predictor (ICT)")
        st.caption("Ye level ko TOUCH hone se pehle fire hota hai — support/resistance ke paas pahunchte hi break ya "
                   "bounce hone ka % chance dikhata hai, taaki lagging confirmation signal ka wait na karna pade.")

        if level_prediction['status'] == 'APPROACHING_LEVEL':
            lp_kind = level_prediction['level_type']
            lp_level = level_prediction['level_price']
            lp_dist = level_prediction['distance_pts']
            lp_break = level_prediction['break_pct']
            lp_bounce = level_prediction['bounce_pct']
            lp_bias = level_prediction['directional_bias']

            lw1, lw2, lw3 = st.columns(3)
            lw1.metric(label=f"Approaching {lp_kind}", value=f"₹{lp_level:,.2f}", delta=f"{lp_dist:.1f} pts away")
            lw2.metric(label="Break-Through Probability", value=f"{lp_break}%")
            lw3.metric(label="Bounce / Reject Probability", value=f"{lp_bounce}%")

            if "🟢" in lp_bias:
                st.success(f"**Early Read:** {lp_bias} — {lp_kind} ₹{lp_level:,.2f} ke paas confluence factors is direction ko favor kar rahe hain.")
            else:
                st.error(f"**Early Read:** {lp_bias} — {lp_kind} ₹{lp_level:,.2f} ke paas confluence factors is direction ko favor kar rahe hain.")

            with st.expander("🔍 Confluence Factors (ICT + Momentum + Volume)", expanded=False):
                for f in level_prediction['factors']:
                    st.write(f"- {f}")

            # Actionable early entry — only shown once conviction crosses the
            # threshold, so you can enter WHILE the move is forming near the
            # level instead of waiting for the slower main confirmation signal.
            ee = level_prediction.get('early_entry')
            if ee and ee.get('entry_price') is not None:
                st.markdown(f"##### ⚡ Early Actionable Entry (Conviction: {ee['confidence_pct']}%)")
                ee1, ee2, ee3, ee4 = st.columns(4)
                ee1.metric("Action", ee['action'])
                ee2.metric("Entry", f"₹{ee['entry_price']:,.2f}")
                ee3.metric("Stop Loss", f"₹{ee['stop_loss']:,.2f}")
                ee4.metric("Target", f"₹{ee['target']:,.2f}")
                st.caption("Ye ek fast, ATR-based tactical entry hai jo abhi level ke paas conviction high hone par bana — "
                           "main Institutional Confluence Signal (upar) se independent hai aur usse pehle trigger ho sakta hai.")
            elif ee:
                st.caption(f"⏳ {ee['action']} — filhaal conviction {ee['confidence_pct']}% hai, "
                           f"{int(support_resistance.ENTRY_CONFIDENCE_THRESHOLD)}%+ hone tak koi actionable entry suggest nahi ki jaati.")
        else:
            ns = level_prediction.get('nearest_support')
            nr = level_prediction.get('nearest_resistance')
            ns_text = f"₹{ns:,.2f}" if ns is not None else "N/A"
            nr_text = f"₹{nr:,.2f}" if nr is not None else "N/A"
            st.info(f"Price abhi kisi key support/resistance ke approach-zone me nahi hai. "
                    f"Nearest Support: {ns_text} | Nearest Resistance: {nr_text}")

        # -------------------------------------------------------------
        # REAL PERFORMANCE TRACKING — the numbers that actually matter
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Real Performance Tracking (Measured, Not Assumed)")

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Walk-Forward ML Accuracy", ml_results["Accuracy"])
        rc2.metric("ML Sample Size", ml_results["Sample Size"])
        rc3.metric("Live DB Win Rate", f"{perf['win_rate']}%" if perf['win_rate'] is not None else "No resolved trades yet")
        rc4.metric("Live DB Sample Size", f"{perf['sample_size']} resolved trades")

        if perf['sample_size'] < 30:
            st.warning(f"⚠️ Only {perf['sample_size']} resolved trades so far. Win-rate numbers below ~30 trades "
                       f"are not statistically reliable — treat them as early signal, not proof.")
        st.caption(f"Wins: {perf['wins']} | Losses: {perf['losses']} | Time-exits (neither hit): {perf['time_exits']} "
                   f"| Simulated Net PnL (points, illustrative): {perf['total_pnl']}")

        # -------------------------------------------------------------
        # HYBRID AI ENGINE REASONING REPORT (unchanged logic)
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🦅 Multi-Factor Hybrid AI Deep Reasoning Report")

        with st.spinner("Analyzing Technicals, Option Chain, FII Footprint, Breadth, SMC & Global Macros..."):
            ai_report_text = ai_engine.generate_llm_reasoning(
                live_price=live_price, meta_summary=ai_analysis,
                sentiment_score=global_sentiment_score, global_avg_change=avg_market_change
            )
            st.markdown(ai_report_text)

        # -------------------------------------------------------------
        # INSTITUTIONAL CHART (unchanged — was correct)
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Institutional Order Flow Chart (VWAP & Volume Profile)")

        chart_df = df.tail(100)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart_df.index, open=chart_df['Open'], high=chart_df['High'],
            low=chart_df['Low'], close=chart_df['Close'], name='Nifty 50',
            increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
        ))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='#2962FF', width=1)))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['VWAP'], mode='lines', name='VWAP (Smart Money Avg)', line=dict(color='#E040FB', width=2.5, dash='dot')))
        fig.add_hline(y=df['POC_Level'].iloc[-1], line_width=2, line_dash="solid", line_color="#FFEA00",
                      annotation_text=f"POC Level (Max Vol): {df['POC_Level'].iloc[-1]:.2f}", annotation_position="top right")
        fig.update_layout(xaxis_title='Time', yaxis_title='Price (₹)', template='plotly_dark', height=600,
                           xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        live_vwap = float(df['VWAP'].iloc[-1])
        live_poc = float(df['POC_Level'].iloc[-1])
        vwap_status = "BULLISH 🟢" if live_price > live_vwap else "BEARISH 🔴"
        poc_status = "PRICE IS ABOVE MAX VOLUME 🟢" if live_price > live_poc else "PRICE IS BELOW MAX VOLUME 🔴"
        col_v1, col_v2 = st.columns(2)
        col_v1.info(f"**VWAP Trend:** Price is {vwap_status} (VWAP: ₹{live_vwap:,.2f})")
        col_v2.info(f"**Smart Money POC:** {poc_status} (POC: ₹{live_poc:,.2f})")

        # -------------------------------------------------------------
        # FII / DII FOOTPRINT & MAX PAIN (unchanged)
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🏛️ Institutional F&O Footprint & Max Pain Analytics")
        col_mp1, col_mp2 = st.columns(2)
        col_mp1.metric(label="🎯 Calculated Max Pain Strike", value=f"₹ {max_pain:,.2f}" if max_pain > 0 else "N/A")
        col_mp2.metric(label="📊 Option Chain PCR", value=f"{current_pcr:.2f}")

        if "BULLISH" in fii_footprint:
            st.success(f"**🏦 FII / DII Net F&O Bias:** {fii_footprint}")
        elif "BEARISH" in fii_footprint:
            st.error(f"**🏦 FII / DII Net F&O Bias:** {fii_footprint}")
        else:
            st.warning(f"**🏦 FII / DII Net F&O Bias:** {fii_footprint}")

        if max_pain > 0:
            if live_price > max_pain:
                st.info(f"**Max Pain Gravity Note:** Live price (₹{live_price:,.2f}) is currently **above** Max Pain (₹{max_pain:,.2f}). Expiry pull is upward unless heavy selling triggers.")
            else:
                st.info(f"**Max Pain Gravity Note:** Live price (₹{live_price:,.2f}) is currently **below** Max Pain (₹{max_pain:,.2f}). Expiry pull is downward.")

        # -------------------------------------------------------------
        # NIFTY INTERNAL BREADTH & HEAVYWEIGHTS TRACKER — combined section.
        # Full 50-stock breadth shown first (how many of the actual 50 are up
        # vs down), then the best-weighted heavyweights table underneath.
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Nifty Internal Breadth & Heavyweights Tracker")

        if breadth50_status.startswith("DATA UNAVAILABLE"):
            st.error(f"**Full Nifty 50 Breadth Status:** {breadth50_status}")
        else:
            col_f1, col_f2, col_f3 = st.columns(3)
            col_f1.metric(label="🟢 Nifty Advances (of 50)", value=adv50)
            col_f2.metric(label="🔴 Nifty Declines (of 50)", value=dec50)
            col_f3.metric(label="📊 Advance/Decline Ratio", value=ratio50)
            st.info(f"**Full Breadth Status:** {breadth50_status}")

        if "Upstox" not in breadth50_status:
            with st.expander("🔧 Why isn't this using Upstox Live? (debug)", expanded=False):
                st.code(full50_debug, language=None)

        st.markdown("**🏆 Nifty 50 Core Heavyweights Performance** (best-weighted stocks, ~65%+ of index weight)")

        if breadth_status.startswith("DATA UNAVAILABLE"):
            st.error(f"**Market Breadth Status:** {breadth_status}")
        else:
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric(label="🟢 Advancing (of 12)", value=advances)
            col_b2.metric(label="🔴 Declining (of 12)", value=declines)
            col_b3.metric(label="📊 Advance/Decline Ratio", value=breadth_ratio)
            st.info(f"**Market Breadth Status:** {breadth_status}")

        try:
            st.dataframe(df_heavyweights.style.background_gradient(subset=['Change (%)'], cmap='RdYlGn'),
                         use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_heavyweights, use_container_width=True, hide_index=True)

        if "Upstox" not in breadth_status:
            with st.expander("🔧 Why isn't this using Upstox Live? (debug)", expanded=False):
                st.code(heavyweights_debug, language=None)

        heavyweight_avg_change = df_heavyweights['Change (%)'].mean() if df_heavyweights['Change (%)'].notna().any() else 0.0
        if signal_code == 1 and heavyweight_avg_change < -0.2:
            st.error("🚨 **FAKE BREAKOUT WARNING:** Confluence is BUY, but core heavyweights are negative! High risk of fake breakout.")
        elif signal_code == -1 and heavyweight_avg_change > 0.2:
            st.warning("⚠️ **FAKE BREAKDOWN WARNING:** Confluence is SELL, but core heavyweights are positive!")
        elif not breadth_status.startswith("DATA UNAVAILABLE"):
            st.success("✅ **Breadth Confirmation:** Heavyweights are aligned with the confluence score.")

        # -------------------------------------------------------------
        # BANK NIFTY, SENSEX & CORRELATED INDICES — the biggest cross-market
        # movers that affect Nifty 50 option chain sentiment.
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🏦 Bank Nifty, Sensex & Correlated Indices")

        idx_cols = st.columns(len(index_correlation_data))
        for col, idx in zip(idx_cols, index_correlation_data):
            if idx["price"] is None:
                col.metric(label=idx["name"], value="N/A")
            else:
                col.metric(label=f"{idx['name']} ({idx['source']})", value=f"₹{idx['price']:,.2f}", delta=f"{idx['change_pct']}%")

        if banknifty_correlation_note:
            if "DIVERGENCE WARNING" in banknifty_correlation_note.upper():
                st.warning(f"**Bank Nifty ⇄ Nifty 50 Correlation:** {banknifty_correlation_note}")
            elif "CONFIRMED" in banknifty_correlation_note.upper():
                st.success(f"**Bank Nifty ⇄ Nifty 50 Correlation:** {banknifty_correlation_note}")
            else:
                st.info(f"**Bank Nifty ⇄ Nifty 50 Correlation:** {banknifty_correlation_note}")

        # -------------------------------------------------------------
        # DATA TABLES (unchanged)
        # -------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🧱 Advanced Technical Indicators Table")
        display_columns = ['Close', 'Volume', 'VWAP', 'POC_Level', 'EMA_20', 'EMA_50', 'MACD', 'ATR', 'RSI']
        available_cols = [col for col in display_columns if col in df.columns]
        with st.expander("📈 View Live Indicators Grid", expanded=False):
            st.dataframe(df[available_cols].tail(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⛓️ Nifty 50 Option Chain (Greeks & PCR)")
        if oc_source == "LIVE":
            st.success("🟢 **Source: LIVE** — real Open Interest and Greeks from your connected Upstox account.")
        else:
            st.warning("⚠️ **Source: SIMULATED** — no live Upstox login/data. Greeks below are calculated with real "
                        "Black-Scholes math (live spot price, live India VIX, real time-to-expiry), but Open Interest "
                        "is a modelled estimate, not real broker OI. Login with Upstox above for live real OI/PCR.")
        if not df_option_chain.empty:
            try:
                st.dataframe(df_option_chain.style.background_gradient(subset=['PCR'], cmap='RdYlGn'), use_container_width=True)
            except Exception:
                st.dataframe(df_option_chain, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🏦 Smart Money Concepts & Market Structure (BOS / CHoCH)")
        if market_structure:
            trigger_val = market_structure[0]["Trigger Level"]
            if "Bullish" in smc_event:
                st.success(f"**Market Structure Status:** {smc_event} | **Trigger Level:** {trigger_val}")
            elif "Bearish" in smc_event:
                st.error(f"**Market Structure Status:** {smc_event} | **Trigger Level:** {trigger_val}")
            else:
                st.info(f"**Market Structure Status:** {smc_event} | **Trigger Level:** {trigger_val}")

        col_smc1, col_smc2, col_smc3 = st.columns(3)
        with col_smc1:
            st.markdown("##### 📌 Fair Value Gaps (FVG)")
            if fvg:
                st.table(pd.DataFrame(fvg))
            else:
                st.info("No active FVG detected.")
        with col_smc2:
            st.markdown("##### 🧱 Order Blocks (OB)")
            if ob:
                st.table(pd.DataFrame(ob))
            else:
                st.info("No active Order Blocks detected.")
        with col_smc3:
            st.markdown("##### 🌊 Liquidity Sweeps")
            if sweeps:
                st.table(pd.DataFrame(sweeps))
            else:
                st.info("No active Sweeps detected.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🌐 Global Market Sentiment & Regional Live News")
        st.info(f"**⚡ World's Strongest News Highlight (Live Today):**\n\n*{top_headline}*")
        try:
            st.dataframe(df_global_sentiment.style.background_gradient(subset=['Positive News', 'Negative News'], cmap='Blues'), use_container_width=True)
        except Exception:
            st.dataframe(df_global_sentiment, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🌍 Global Major Stock Markets & Macro Live Tracker")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="🎯 Automatic Global Sentiment Score", value=global_sentiment_score)
        with col_m2:
            st.metric(label="📈 Average Global Change (%)", value=f"{avg_market_change}%")
        try:
            st.dataframe(df_global_markets, column_config={"Logo": st.column_config.ImageColumn("Flag / Icon", width="small")},
                         hide_index=True, use_container_width=True)
        except Exception:
            st.dataframe(df_global_markets, use_container_width=True)

    # -------------------------------------------------------------
    # Save this run's full context for the chat box to read. The chat is
    # rendered OUTSIDE this fragment (see bottom of file) so it is never
    # interrupted by this fragment's own 30s refresh cycle.
    # -------------------------------------------------------------
    st.session_state['dashboard_context'] = {
            "Market Status": status_line,
            "Data Freshness": (
                f"STALE/FROZEN -- {data_freshness['note']}" if data_freshness['is_stale'] else "Live / fresh"
            ),
            "Nifty Spot Price": f"₹{live_price:,.2f}",
            "Confluence Signal": final_signal_text,
            "AI Confluence Score": f"{ai_analysis['confidence_pct']}%",
            "Entry / SL / T1 / T2": f"Entry ₹{entry_price:,.2f} | SL ₹{sl:,.2f} | T1 ₹{t1:,.2f} | T2 ₹{t2:,.2f}",
            "Active DB Trade": (f"{open_trade['signal']} | Entry ₹{open_trade['entry_price']:.2f} | "
                                 f"SL ₹{open_trade['stop_loss']:.2f} | Target ₹{open_trade['target_1']:.2f}")
                                if open_trade else "None open right now",
            "Walk-Forward ML Accuracy": ml_results["Accuracy"],
            "Live DB Win Rate": f"{perf['win_rate']}%" if perf['win_rate'] is not None else "No resolved trades yet",
            "Live DB Sample Size": perf['sample_size'],
            "RSI": f"{ai_analysis['tech_metrics']['rsi']:.2f}",
            "MACD": f"{ai_analysis['tech_metrics']['macd']:.2f}",
            "VWAP": f"₹{live_vwap:,.2f} ({vwap_status})",
            "EMA 20 / EMA 50": f"₹{ai_analysis['tech_metrics']['ema_20']:,.2f} / ₹{ai_analysis['tech_metrics']['ema_50']:,.2f}",
            "ATR": f"{ai_analysis['tech_metrics']['atr']:.2f}",
            "Smart Money POC": f"₹{live_poc:,.2f} ({poc_status})",
            "Option Chain PCR": f"{current_pcr:.2f}",
            "Option Chain Source": oc_source,
            "Max Pain Strike": f"₹{max_pain:,.2f}" if max_pain > 0 else "N/A",
            "Max Pain Gravity": (
                f"Live price is ABOVE Max Pain -- expiry pull tends upward unless heavy selling shows up"
                if max_pain > 0 and live_price > max_pain else
                f"Live price is BELOW Max Pain -- expiry pull tends downward"
                if max_pain > 0 else "N/A"
            ),
            "FII / DII Footprint": fii_footprint,
            "Full Nifty 50 Breadth": breadth50_status,
            "Heavyweights (12-stock) Breadth": breadth_status,
            "Bank Nifty / Sensex": ", ".join(
                f"{idx['name']}: ₹{idx['price']:,.2f} ({idx['change_pct']}%)" if idx["price"] is not None else f"{idx['name']}: N/A"
                for idx in index_correlation_data
            ),
            "Bank Nifty ⇄ Nifty 50 Correlation": banknifty_correlation_note,
            "Smart Money Structure (BOS/CHoCH)": smc_event,
            "Volatility Regime": "Choppy / Range-bound" if is_choppy else "Active",
            "Global Sentiment": f"{global_sentiment_score} (avg change {avg_market_change}%)",
            "Top Global Headline": top_headline,
            "Early Warning (S/R Approach Predictor)": (
                f"Approaching {level_prediction['level_type']} ₹{level_prediction['level_price']:,.2f} "
                f"({level_prediction['distance_pts']:.1f} pts away) | Break: {level_prediction['break_pct']}% | "
                f"Bounce/Reject: {level_prediction['bounce_pct']}% | Read: {level_prediction['directional_bias']} | "
                f"Factors: {'; '.join(level_prediction['factors'])} | "
                f"Early Entry: " + (
                    f"{level_prediction['early_entry']['action']} @ ₹{level_prediction['early_entry']['entry_price']:,.2f} "
                    f"(SL ₹{level_prediction['early_entry']['stop_loss']:,.2f}, Target ₹{level_prediction['early_entry']['target']:,.2f})"
                    if level_prediction.get('early_entry') and level_prediction['early_entry'].get('entry_price') is not None
                    else f"{level_prediction['early_entry']['action']} (confidence {level_prediction['early_entry']['confidence_pct']}%)"
                    if level_prediction.get('early_entry') else "N/A"
                )
            ) if level_prediction['status'] == 'APPROACHING_LEVEL' else (
                f"Price not currently near a key level. Nearest Support: "
                f"{'₹' + format(level_prediction['nearest_support'], ',.2f') if level_prediction.get('nearest_support') is not None else 'N/A'} | "
                f"Nearest Resistance: "
                f"{'₹' + format(level_prediction['nearest_resistance'], ',.2f') if level_prediction.get('nearest_resistance') is not None else 'N/A'}"
            ),
        }


_live_dashboard()

# -------------------------------------------------------------
# INTERACTIVE AI CHAT -- deliberately OUTSIDE the fragment above, so the
# 30s live-data refresh never interrupts a reply being read aloud or text
# you're in the middle of typing. It reads the latest snapshot the
# fragment saved to session_state, which is fresh as of the last refresh.
# -------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
render_ai_chat(
    gemini_api_key=GEMINI_API_KEY,
    dashboard_context=st.session_state.get('dashboard_context', {})
)
