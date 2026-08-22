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
from hybrid_ai_engine import HybridAIEngine
from alert_manager import AlertManager  # 🔔 Added Telegram Alert Manager

# Page Configuration
st.set_page_config(page_title="Nifty 50 Real-Time AI Predictor", page_icon="⚡", layout="wide")

# -------------------------------------------------------------
# 📱 MOBILE-RESPONSIVE CSS (Desktop untouched, Mobile optimized)
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
# -------------------------------------------------------------

# Secure Gemini API Key Configuration using Streamlit Secrets
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

# Initialize Session State for Dynamic Trailing Stop Loss (TSL)
if 'active_trade' not in st.session_state:
    st.session_state.active_trade = None
    st.session_state.entry_price = 0.0
    st.session_state.direction = None
    st.session_state.trailing_sl = 0.0

# Step 1: Login & Get Broker Token
access_token = upstox_auth.get_upstox_access_token()

# Step 2: Fetch Live Market Data
df, model, feature_cols, live_price = market_data.fetch_live_market_data(access_token)

if df is not None and not df.empty:
    
    # -------------------------------------------------------------
    # 🚀 ADVANCED FEATURE: VWAP & POC CALCULATION
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
            
    except Exception as e:
        df['VWAP'] = df['EMA_20'] if 'EMA_20' in df.columns else df['Close']
        df['POC_Level'] = df['Close'].iloc[-1]
    # -------------------------------------------------------------

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
    # 🎯 INSTITUTIONAL MULTI-FACTOR CONFLUENCE & ACCURACY ENGINE
    # -------------------------------------------------------------
    latest_data = df[feature_cols].tail(1)
    ml_signal = int(model.predict(latest_data)[0])
    
    live_vwap = float(df['VWAP'].iloc[-1])
    live_poc = float(df['POC_Level'].iloc[-1])
    heavyweight_avg_change = df_heavyweights['Change (%)'].mean()

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
    else: confluence_score -= 15

    confluence_score = max(10, min(95, confluence_score))

    # Determine Initial Actionable Signal based on Confluence
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

    # -------------------------------------------------------------
    # 🌪️ NEW: ADVANCED VOLATILITY REGIME FILTER (CHOOPY MARKET BLOCKER)
    # -------------------------------------------------------------
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    
    avg_bb_width = df['BB_Width'].rolling(window=50).mean().iloc[-1]
    current_bb_width = df['BB_Width'].iloc[-1]

    # Agar Volatility bahut low hai (Market Soya hai / Rangebound hai)
    is_choppy = current_bb_width < (avg_bb_width * 0.75)

    if is_choppy:
        confluence_score = 50
        final_signal_text = "CHOPPY MARKET ⚠️ (No Trade Zone - Low Volatility)"
        signal_code = 0
    # -------------------------------------------------------------

    daily_atr = float(df['ATR'].iloc[-1])
    intraday_atr = daily_atr * 0.4

    # -------------------------------------------------------------
    # 🛡️ DYNAMIC ATR-BASED TRAILING STOP LOSS (TSL) ENGINE
    # -------------------------------------------------------------
    if signal_code != 0 and st.session_state.active_trade is None:
        st.session_state.active_trade = True
        st.session_state.entry_price = live_price
        st.session_state.direction = signal_code
        st.session_state.trailing_sl = live_price - (1.5 * intraday_atr) if signal_code == 1 else live_price + (1.5 * intraday_atr)

    if st.session_state.active_trade:
        direction = st.session_state.direction
        entry = st.session_state.entry_price
        
        if direction == 1:  # BUY Side
            if live_price >= entry + (1.0 * intraday_atr):
                st.session_state.trailing_sl = entry  # Breakeven Protection
            if live_price >= entry + (2.5 * intraday_atr):
                st.session_state.trailing_sl = entry + (1.0 * intraday_atr)  # Profit Lock-in
                
        elif direction == -1:  # SELL Side
            if live_price <= entry - (1.0 * intraday_atr):
                st.session_state.trailing_sl = entry  # Breakeven Protection
            if live_price <= entry - (2.5 * intraday_atr):
                st.session_state.trailing_sl = entry - (1.0 * intraday_atr)  # Profit Lock-in

    if signal_code == 0:
        st.session_state.active_trade = None

    sl = st.session_state.trailing_sl if st.session_state.active_trade else (live_price - (1.0 * intraday_atr) if signal_code >= 0 else live_price + (1.0 * intraday_atr))
    t1 = live_price + (2.0 * intraday_atr) if signal_code >= 0 else live_price - (2.0 * intraday_atr)
    t2 = live_price + (4.0 * intraday_atr) if signal_code >= 0 else live_price - (4.0 * intraday_atr)

    # Save to Database
    database.log_prediction_to_db(live_price, signal_code, sl, t1, t2)

    # 🔔 Send Telegram Alert on High Conviction Signals (Confidence >= 68)
    if alert_sys and signal_code != 0 and confluence_score >= 68:
        alert_sys.send_trade_alert(
            signal_type=final_signal_text,
            confidence=confluence_score,
            price=live_price,
            sentiment=f"Global Score: {global_sentiment_score}%",
            logic="Institutional Confluence Score >= 68 + Volatility Filter Passed"
        )

    # Step 4: Show Core Metrics with Confluence Accuracy Score
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="Real-Time Live Nifty Price (LTP)", value=f"₹ {live_price:,.2f}")
    with col2: st.metric(label="Institutional Confluence Signal", value=final_signal_text)
    with col3: st.metric(label="AI Model Accuracy Score", value=f"{confluence_score}%")

    # Display Dynamic Active Trade Management Banner
    if st.session_state.active_trade:
        st.warning(f"🛡️ **ACTIVE TRADE MANAGEMENT** | Entry: ₹{st.session_state.entry_price:.2f} | **Dynamic Trailing SL: ₹{st.session_state.trailing_sl:.2f}**")
    else:
        st.info("ℹ️ System is scanning for high-probability entry setup with Volatility Regime Protection...")
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # 🦅 HYBRID AI ENGINE INTEGRATION & REASONING REPORT (UPDATED)
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🦅 Multi-Factor Hybrid AI Deep Reasoning Report")
    
    with st.spinner("Analyzing Technicals, Option Chain, FII Footprint, Breadth, SMC & Global Macros..."):
        ai_analysis = ai_engine.analyze(
            live_price=live_price,
            df=df,
            df_option_chain=df_option_chain,
            smc_data=smc_event,
            sentiment_score=global_sentiment_score,
            global_avg_change=avg_market_change,
            fii_footprint=fii_footprint,
            breadth_status=breadth_status,
            is_choppy=is_choppy
        )

        ai_report_text = ai_engine.generate_llm_reasoning(
            live_price=live_price,
            meta_summary=ai_analysis,
            sentiment_score=global_sentiment_score,
            global_avg_change=avg_market_change
        )

        st.markdown(ai_report_text)

    # -------------------------------------------------------------
    # 📊 ADVANCED INSTITUTIONAL CHART (VWAP + POC)
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
    fig.add_hline(y=live_poc, line_width=2, line_dash="solid", line_color="#FFEA00", annotation_text=f"POC Level (Max Vol): {live_poc:.2f}", annotation_position="top right")

    fig.update_layout(xaxis_title='Time', yaxis_title='Price (₹)', template='plotly_dark', height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    vwap_status = "BULLISH 🟢" if live_price > live_vwap else "BEARISH 🔴"
    poc_status = "PRICE IS ABOVE MAX VOLUME 🟢" if live_price > live_poc else "PRICE IS BELOW MAX VOLUME 🔴"
    
    col_v1, col_v2 = st.columns(2)
    col_v1.info(f"**VWAP Trend:** Price is {vwap_status} (VWAP: ₹{live_vwap:,.2f})")
    col_v2.info(f"**Smart Money POC:** {poc_status} (POC: ₹{live_poc:,.2f})")

    # -------------------------------------------------------------
    # 🏛️ FII / DII FOOTPRINT & MAX PAIN ANALYTICS
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
    # 📈 NIFTY INTERNAL BREADTH & FAKE BREAKOUT DETECTOR
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Nifty Internal Breadth & Heavyweights Tracker")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric(label="🟢 Nifty Advances", value=advances)
    col_b2.metric(label="🔴 Nifty Declines", value=declines)
    col_b3.metric(label="📊 Advance / Decline Ratio", value=breadth_ratio)
    
    st.info(f"**Market Breadth Status:** {breadth_status}")
    
    st.markdown("##### 🏋️ Nifty 50 Core Heavyweights Performance")
    try:
        st.dataframe(df_heavyweights.style.background_gradient(subset=['Change (%)'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(df_heavyweights, use_container_width=True, hide_index=True)
        
    if signal_code == 1 and heavyweight_avg_change < -0.2:
        st.error("🚨 **FAKE BREAKOUT WARNING:** Institutional Confluence is **BUY**, but core heavyweights are negative! High risk of fake breakout.")
    elif signal_code == -1 and heavyweight_avg_change > 0.2:
        st.warning("⚠️ **FAKE BREAKDOWN WARNING:** Institutional Confluence is **SELL**, but core heavyweights are supporting the market positively!")
    else:
        st.success("✅ **Breadth Confirmation:** Heavyweights are perfectly aligned with the institutional confluence score.")
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # 🧱 DATA TABLES & SUPPORTING METRICS
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧱 Advanced Technical Indicators Table")
    display_columns = ['Close', 'Volume', 'VWAP', 'POC_Level', 'EMA_20', 'EMA_50', 'MACD', 'ATR', 'RSI']
    available_cols = [col for col in display_columns if col in df.columns]
    
    with st.expander("📈 View Live Indicators Grid", expanded=False):
        st.dataframe(df[available_cols].tail(10), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⛓️ Nifty 50 Option Chain (Greeks & PCR)")
    if not df_option_chain.empty:
        try: st.dataframe(df_option_chain.style.background_gradient(subset=['PCR'], cmap='RdYlGn'), use_container_width=True)
        except Exception: st.dataframe(df_option_chain, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏦 Smart Money Concepts & Market Structure (BOS / CHoCH)")
    
    if market_structure:
        trigger_val = market_structure[0]["Trigger Level"]
        if "Bullish" in smc_event: st.success(f"**Market Structure Status:** {smc_event} | **Trigger Level:** {trigger_val}")
        elif "Bearish" in smc_event: st.error(f"**Market Structure Status:** {smc_event} | **Trigger Level:** {trigger_val}")
        else: st.info(f"**Market Structure Status:** {smc_event} | **Trigger Level:** {trigger_val}")

    col_smc1, col_smc2, col_smc3 = st.columns(3)
    with col_smc1:
        st.markdown("##### 📌 Fair Value Gaps (FVG)")
        if fvg: st.table(pd.DataFrame(fvg))
        else: st.info("No active FVG detected.")
    with col_smc2:
        st.markdown("##### 🧱 Order Blocks (OB)")
        if ob: st.table(pd.DataFrame(ob))
        else: st.info("No active Order Blocks detected.")
    with col_smc3:
        st.markdown("##### 🌊 Liquidity Sweeps")
        if sweeps: st.table(pd.DataFrame(sweeps))
        else: st.info("No active Sweeps detected.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌐 Global Market Sentiment & Regional Live News")
    st.info(f"**⚡ World's Strongest News Highlight (Live Today):**\n\n*{top_headline}*")
    try: st.dataframe(df_global_sentiment.style.background_gradient(subset=['Positive News', 'Negative News'], cmap='Blues'), use_container_width=True)
    except Exception: st.dataframe(df_global_sentiment, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🌍 Global Major Stock Markets & Macro Live Tracker")
    col_m1, col_m2 = st.columns(2)
    with col_m1: st.metric(label="🎯 Automatic Global Sentiment Score", value=global_sentiment_score)
    with col_m2: st.metric(label="📈 Average Global Change (%)", value=f"{avg_market_change}%")
    
    try: st.dataframe(df_global_markets, column_config={"Logo": st.column_config.ImageColumn("Flag / Icon", width="small")}, hide_index=True, use_container_width=True)
    except Exception: st.dataframe(df_global_markets, use_container_width=True)
