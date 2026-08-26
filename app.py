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
        rc3.metric("Live DB Win Rate", f"{perf['win_rate']}%" if perf['win_rate'] is not None else "N
