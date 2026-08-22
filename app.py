import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

# --- IMPORT CUSTOM MODULES ---
import database
import market_data
import global_markets
import global_news
import market_breadth
import indicators
import ml_engine
from hybrid_ai_engine import HybridAIEngine
from alert_manager import AlertManager

# Optional safe imports for structural modules if available
try:
    import option_chain
except ImportError:
    option_chain = None

try:
    import smart_money
except ImportError:
    smart_money = None

try:
    import upstox_auth
except ImportError:
    upstox_auth = None

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nifty 50 Real-Time Institutional AI Terminal",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    </style>
""", unsafe_allow_html=True)

# Initialize Database
database.init_db()

# Auto-refresh every 30 seconds for live tick data
st_autorefresh(interval=30000, key="nifty_live_refresh")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🦅 Institutional Controls")
gemini_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
fii_footprint_sel = st.sidebar.selectbox("FII / DII Footprint", ["Neutral 🟡", "Bullish 🟢 (Net Buyers)", "Bearish 🔴 (Net Sellers)"])
is_choppy_toggle = st.sidebar.checkbox("Force Choppy / Sideways Guard ⚠️", value=False)

st.sidebar.markdown("---")
st.sidebar.info("System Status: Live Data Sync Active 🟢")

# --- FETCH LIVE DATA ACROSS MODULES ---
with st.spinner("Fetching real-time feeds & executing quantitative models..."):
    # 1. Nifty Intraday Data & Indicators
    df_raw = market_data.fetch_nifty_data()
    if not df_raw.empty:
        df = indicators.calculate_technical_indicators(df_raw)
        live_price = float(df['Close'].iloc[-1])
        vwap_val = float(df['VWAP'].iloc[-1]) if 'VWAP' in df.columns else live_price
    else:
        live_price, vwap_val = 24500.0, 24500.0
        df = pd.DataFrame()

    # 2. Global Markets & Macro VIX
    df_global_markets = global_markets.get_global_market_indices()
    global_sentiment_score, global_avg_change = global_markets.get_global_market_summary(df_global_markets)
    live_vix = global_markets.get_live_vix(df_global_markets)

    # 3. Global Regional News Sentiment
    df_region_sentiment, top_world_headline = global_news.get_global_market_sentiment()

    # 4. Market Breadth & Heavyweights
    df_breadth, total_adv, total_dec = market_breadth.get_real_market_breadth()
    breadth_status = "Strong Bullish 🟢" if total_adv > 30 else ("Strong Bearish 🔴" if total_dec > 30 else "Neutral 🟡")

    # 5. Option Chain & SMC Hooks
    df_oc = option_chain.get_option_chain_data() if option_chain and hasattr(option_chain, 'get_option_chain_data') else None
    smc_event = smart_money.detect_smc_structure(df) if smart_money and hasattr(smart_money, 'detect_smc_structure') else "BOS - Bullish Continuation"

    # 6. Hybrid AI Engine Execution
    ai_engine = HybridAIEngine(api_key=gemini_key if gemini_key else None)
    analysis = ai_engine.analyze(
        live_price=live_price,
        df=df,
        df_option_chain=df_oc,
        smc_data=smc_event,
        sentiment_score=global_sentiment_score,
        global_avg_change=global_avg_change,
        fii_footprint=fii_footprint_sel,
        breadth_status=breadth_status,
        is_choppy=is_choppy_toggle
    )

# --- MAIN DASHBOARD INTERFACE ---
st.title("🦅 Nifty 50 Real-Time Institutional AI Predictor & Terminal")

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Nifty Spot Price", f"₹{live_price:,.2f}", f"{global_avg_change}% Macro")
col2.metric("AI Conviction Bias", analysis['bias_text'], f"{analysis['confidence_pct']}% Conf")
col3.metric("Live India VIX", f"{live_vix:.2f}", "Volatility")
col4.metric("Market Breadth", f"Adv: {total_adv} / Dec: {total_dec}", breadth_status)
col5.metric("Active Signal", analysis['signal_type'])

st.markdown("---")

# Tabs for Organized Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Chart & AI Execution", 
    "🌍 Global Markets & VIX", 
    "📰 Global News Sentiment", 
    "📊 Market Breadth & Heavyweights", 
    "🧠 ML Backtest & Trade Logs"
])

with tab1:
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader("📉 Nifty Intraday Price Action & Technicals")
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Nifty 5m"
            ))
            if 'VWAP' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=1.5), name="VWAP"))
            if 'EMA_20' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='blue', width=1), name="EMA 20"))
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient intraday data for charting.")

    with col_r:
        st.subheader("🎯 Institutional Trade Plan")
        st.info(f"**Action:** {analysis['signal_type']}")
        st.write(f"**Entry Strategy:** {analysis['entry_type']}")
        st.write(f"**Execution Level:** ₹{analysis['entry_price']}")
        st.write(f"**Stop Loss:** ₹{analysis['stop_loss']} ({analysis['sl_points']} pts risk)")
        st.write(f"**Target 1:** ₹{analysis['target_1']}")
        st.write(f"**Target 2:** ₹{analysis['target_2']}")
        st.write(f"**Risk-Reward:** {analysis['risk_reward_ratio']}")
        
        # Trade Logger Button
        if st.button("📥 Log Active Setup to Database"):
            t_id, msg = database.log_entry_safe(analysis['signal_type'], analysis['entry_price'], analysis['stop_loss'], analysis['target_1'])
            if t_id:
                AlertManager.trigger_alert(f"Trade successfully logged with ID #{t_id}", "success")
            else:
                AlertManager.trigger_alert(msg, "warning")

    st.markdown("---")
    st.subheader("🦅 Deep Quant AI Reasoning & Desk Report")
    llm_report = ai_engine.generate_llm_reasoning(live_price, analysis, global_sentiment_score, global_avg_change)
    st.markdown(llm_report)

with tab2:
    st.subheader("🌐 Real-Time Global Markets & Macro Indicators")
    st.dataframe(df_global_markets, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("📰 Global Regional News & Sentiment Breakdown")
    st.markdown(f"🔥 **Top World Headline:** {top_world_headline}")
    st.dataframe(df_region_sentiment, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("📊 Market Breadth & Heavyweights Tracker")
    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        st.metric("Total Market Internal Breadth Score", f"{total_adv} Advances / {total_dec} Declines")
    with col_b2:
        st.write(f"**Current Trend Health:** {breadth_status}")
    st.dataframe(df_breadth, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("🧠 Machine Learning Ensemble Walk-Forward Backtest")
    
    # ML Engine Integration Call
    ml_metrics = ml_engine.train_and_backtest(df, live_vix=live_vix)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Ensemble Accuracy", ml_metrics.get("Accuracy", "N/A"))
    m_col2.metric("Walk-Forward Win Rate", ml_metrics.get("Win Rate", "N/A"))
    m_col3.metric("Simulated Net PnL", ml_metrics.get("Net PnL", "N/A"))
    m_col4.metric("Sample Size", ml_metrics.get("Sample Size", "0 Trades"))

    if ml_metrics.get("model_ready"):
        sig_map = {1: "Bullish Signal 🟢", -1: "Bearish Signal 🔴", 0: "No Edge / Sideways ⚪"}
        st.info(f"🤖 **Live Ensemble Model State:** Ready | **Current Signal:** {sig_map.get(ml_metrics.get('latest_signal'), 'Neutral')} | **Model Confidence:** {ml_metrics.get('latest_confidence') * 100:.1f}%")
    else:
        st.warning("🤖 **Live Ensemble Model State:** Collecting more historical features for robust threshold validation.")

    st.markdown("---")
    st.subheader("🗄️ Database Performance & Trade Log History")
    wins, losses, win_rate = database.fetch_performance_metrics()
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("Total Winning Trades", wins)
    d_col2.metric("Total Losing Trades", losses)
    d_col3.metric("Overall Win Rate %", f"{win_rate}%")
