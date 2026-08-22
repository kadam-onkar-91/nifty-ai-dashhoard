import streamlit as st
from streamlit_autorefresh import st_autorefresh
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
import indicators
import lm_engine  # 🧠 Integrated Machine Learning Engine
from hybrid_ai_engine import HybridAIEngine
from alert_manager import AlertManager  # 🔔 Telegram Alert Manager

# Page Configuration
st.set_page_config(page_title="Nifty 50 Real-Time AI Predictor", page_icon="⚡", layout="wide")

# -------------------------------------------------------------
# 📱 MOBILE-RESPONSIVE CSS
# -------------------------------------------------------------
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

# Secure Gemini API Key Configuration
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error("⚠️ GEMINI_API_KEY not found. Please set it in your secrets.toml file or Streamlit Cloud dashboard.")
    st.stop()

ai_engine = HybridAIEngine(api_key=GEMINI_API_KEY)

# Initialize Telegram Alert System securely using Secrets
try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    alert_sys = AlertManager(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
except Exception as e:
    alert_sys = None

# Auto-refresh setup: 30 seconds (30000 ms)
st_autorefresh(interval=30000, key="datarefresh")

st.title("⚡ Nifty 50 Institutional AI Trading Dashboard (Pro Edition)")
st.markdown("---")

# Initialize Session State
if 'access_token' not in st.session_state:
    st.session_state.access_token = None

if 'active_trade' not in st.session_state:
    st.session_state.active_trade = None
    st.session_state.entry_price = 0.0
    st.session_state.direction = None
    st.session_state.trailing_sl = 0.0

# Step 1: Login & Get Broker Token
access_token = upstox_auth.get_upstox_access_token()
st.session_state['access_token'] = access_token

# Step 2: Fetch Live Market Data & XGBoost Model from market_data.py
df, ml_model, feature_cols, live_price = market_data.fetch_live_market_data(access_token)

if df is not None and not df.empty:
    
    # -------------------------------------------------------------
    # 🚀 CALCULATE TECHNICAL INDICATORS & VWAP / POC
    # -------------------------------------------------------------
    df = indicators.calculate_technical_indicators(df)
    
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
    df_option_chain = option_chain.generate_option_chain_data(live_price)
    fvg, ob, sweeps = smart_money.detect_smc_zones(df)
    market_structure = smart_money.detect_market_structure(df)
    smc_event = market_structure[0]["Market Event"] if market_structure else "Neutral Structure"
    
    df_heavyweights, advances, declines, breadth_ratio, breadth_status = market_breadth.get_nifty_internal_breadth(access_token)
    max_pain = option_chain.calculate_max_pain(df_option_chain)
    fii_footprint, current_pcr = option_chain.get_fii_dii_fo_footprint(df_option_chain)
    
    df_global_sentiment, top_headline = global_news.get_global_market_sentiment()
    df_global_markets = global_markets.get_global_market_indices()
    global_sentiment_score, avg_market_change = global_markets.get_global_market_summary(df_global_markets)

    # -------------------------------------------------------------
    # 🧠 MACHINE LEARNING ENSEMBLE ENGINE (lm_engine integration)
    # -------------------------------------------------------------
    ml_results = lm_engine.train_and_backtest(df)
    ml_signal = ml_results.get("latest_signal", 0)
    
    live_vwap = float(df['VWAP'].iloc[-1])
    live_poc = float(df['POC_Level'].iloc[-1])
    heavyweight_avg_change = df_heavyweights['Change (%)'].mean() if not df_heavyweights.empty else 0.0

    # Calculate Confluence Score (Base 50%)
    confluence_score = 50
    if live_price > live_vwap: confluence_score += 10
    else: confluence_score -= 10

    if live_price > live_poc: confluence_score += 10
    else: confluence_score -= 10

    if heavyweight_avg_change > 0.1: confluence_score += 15
    elif heavyweight_avg_change < -0.1: confluence_score -= 15

    if current_pcr > 1.02: confluence_score += 10
    elif current_pcr < 0.98: confluence_score -= 10

    if ml_signal == 1: confluence_score += 15
    elif ml_signal == -1: confluence_score -= 15

    confluence_score = max(10, min(95, confluence_score))

    # Actionable Signal determination
    if confluence_score >= 68:
        final_signal_text = "STRONG BUY 🟢 (High Conviction)"
        signal_code = 1
    elif confluence_score >= 55:
        final_signal_text = "MODERATE BUY 🟢"
        signal_code = 1
    elif confluence_score <= 32:
        final_signal_text = "STRONG SELL 🔴 (High Conviction)"
        signal_code = -1
    elif confluence_score <= 45:
        final_signal_text = "MODERATE SELL 🔴"
        signal_code = -1
    else:
        final_signal_text = "NO TRADE / CHOPPY MARKET ⚠️ (Wait)"
        signal_code = 0

    # Volatility Regime Filter
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['EMA_20'].replace(0, 1)
    avg_bb_width = df['BB_Width'].rolling(window=50).mean().iloc[-1]
    current_bb_width = df['BB_Width'].iloc[-1]
    is_choppy = current_bb_width < (avg_bb_width * 0.75) if not np.isnan(avg_bb_width) else False

    if is_choppy:
        confluence_score = 50
        final_signal_text = "CHOPPY MARKET ⚠️ (No Trade Zone - Low Volatility)"
        signal_code = 0

    daily_atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else (live_price * 0.01)
    intraday_atr = daily_atr * 0.4

    # Trailing Stop Loss Management
    if signal_code != 0 and st.session_state.active_trade is None:
        st.session_state.active_trade = True
        st.session_state.entry_price = live_price
        st.session_state.direction = signal_code
        st.session_state.trailing_sl = live_price - (1.5 * intraday_atr) if signal_code == 1 else live_price + (1.5 * intraday_atr)

    if st.session_state.active_trade:
        direction = st.session_state.direction
        entry = st.session_state.entry_price
        if direction == 1:
            if live_price >= entry + (1.0 * intraday_atr): st.session_state.trailing_sl = entry
            if live_price >= entry + (2.5 * intraday_atr): st.session_state.trailing_sl = entry + (1.0 * intraday_atr)
        elif direction == -1:
            if live_price <= entry - (1.0 * intraday_atr): st.session_state.trailing_sl = entry
            if live_price <= entry - (2.5 * intraday_atr): st.session_state.trailing_sl = entry - (1.0 * intraday_atr)

    if signal_code == 0:
        st.session_state.active_trade = None

    sl = st.session_state.trailing_sl if st.session_state.active_trade else (live_price - (1.0 * intraday_atr) if signal_code >= 0 else live_price + (1.0 * intraday_atr))
    t1 = live_price + (2.0 * intraday_atr) if signal_code >= 0 else live_price - (2.0 * intraday_atr)
    t2 = live_price + (4.0 * intraday_atr) if signal_code >= 0 else live_price - (4.0 * intraday_atr)

    try:
        database.log_prediction_to_db(live_price, signal_code, sl, t1, t2)
    except Exception:
        pass

    if alert_sys and signal_code != 0 and confluence_score >= 68:
        alert_sys.send_trade_alert(
            signal_type=final_signal_text,
            confidence=confluence_score,
            price=live_price,
            sentiment=f"Global Score: {global_sentiment_score}",
            logic="Institutional Confluence Score >= 68 + Volatility Filter Passed"
        )

    # Core Metrics Display
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label="Real-Time Nifty Price", value=f"₹ {live_price:,.2f}")
    with col2: st.metric(label="Institutional Signal", value=final_signal_text)
    with col3: st.metric(label="Confluence Score", value=f"{confluence_score}%")
    with col4: st.metric(label="ML Win Rate", value=ml_results.get("Win Rate", "N/A"))

    if st.session_state.active_trade:
        st.warning(f"🛡️ **ACTIVE TRADE** | Entry: ₹{st.session_state.entry_price:.2f} | **Trailing SL: ₹{st.session_state.trailing_sl:.2f}**")
    else:
        st.info("ℹ️ System is monitoring high-probability setups...")

    # Hybrid AI Reasoning Report
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🦅 Multi-Factor Hybrid AI Deep Reasoning Report")
    with st.spinner("Analyzing Technicals, Option Chain, FII Footprint, Breadth, SMC & Global Macros..."):
        ai_analysis = ai_engine.analyze(
            live_price=live_price, df=df, df_option_chain=df_option_chain,
            smc_data=smc_event, sentiment_score=global_sentiment_score,
            global_avg_change=avg_market_change, fii_footprint=fii_footprint,
            breadth_status=breadth_status, is_choppy=is_choppy
        )
        ai_report_text = ai_engine.generate_llm_reasoning(
            live_price=live_price, meta_summary=ai_analysis,
            sentiment_score=global_sentiment_score, global_avg_change=avg_market_change
        )
        st.markdown(ai_report_text)

    # Order Flow Chart
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
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['VWAP'], mode='lines', name='VWAP', line=dict(color='#E040FB', width=2.5, dash='dot')))
    fig.add_hline(y=live_poc, line_width=2, line_dash="solid", line_color="#FFEA00", annotation_text=f"POC: {live_poc:.2f}", annotation_position="top right")
    fig.update_layout(xaxis_title='Time', yaxis_title='Price (₹)', template='plotly_dark', height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # FII Footprint & Max Pain
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏛️ Institutional F&O Footprint & Max Pain Analytics")
    col_mp1, col_mp2 = st.columns(2)
    col_mp1.metric(label="🎯 Max Pain Strike", value=f"₹ {max_pain:,.2f}" if max_pain > 0 else "N/A")
    col_mp2.metric(label="📊 Option Chain PCR", value=f"{current_pcr:.2f}")
    if "BULLISH" in fii_footprint: st.success(f"**🏦 FII / DII Bias:** {fii_footprint}")
    elif "BEARISH" in fii_footprint: st.error(f"**🏦 FII / DII Bias:** {fii_footprint}")
    else: st.warning(f"**🏦 FII / DII Bias:** {fii_footprint}")

    # Market Breadth & Heavyweights
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Nifty Internal Breadth & Heavyweights Tracker")
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric(label="🟢 Advances", value=advances)
    col_b2.metric(label="🔴 Declines", value=declines)
    col_b3.metric(label="📊 A/D Ratio", value=breadth_ratio)
    st.info(f"**Market Breadth Status:** {breadth_status}")
    
    st.markdown("##### 🏋️ Heavyweights Performance")
    try:
        st.dataframe(df_heavyweights.style.background_gradient(subset=['Change (%)'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(df_heavyweights, use_container_width=True, hide_index=True)

    # Data Tables & Option Chain
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⛓️ Nifty 50 Option Chain (Greeks & PCR)")
    if not df_option_chain.empty:
        try: st.dataframe(df_option_chain.style.background_gradient(subset=['PCR'], cmap='RdYlGn'), use_container_width=True)
        except Exception: st.dataframe(df_option_chain, use_container_width=True)

    # Global Markets
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌍 Global Major Stock Markets & Macro Live Tracker")
    col_m1, col_m2 = st.columns(2)
    with col_m1: st.metric(label="🎯 Global Sentiment Score", value=global_sentiment_score)
    with col_m2: st.metric(label="📈 Avg Global Change (%)", value=f"{avg_market_change}%")
    try:
        st.dataframe(df_global_markets, column_config={"Logo": st.column_config.ImageColumn("Flag", width="small")}, hide_index=True, use_container_width=True)
    except Exception:
        st.dataframe(df_global_markets, use_container_width=True)
else:
    st.warning("⚠️ Live market data is currently unavailable. Please check your Upstox API credentials or internet connection.")
