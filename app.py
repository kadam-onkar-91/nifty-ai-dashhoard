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
import index_correlation
import ml_engine
from hybrid_ai_engine import HybridAIEngine
from alert_manager import AlertManager

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

st_autorefresh(interval=30000, key="datarefresh")

st.title("⚡ Nifty 50 Institutional AI Trading Dashboard (Pro Edition)")
st.caption("⚠️ Research/paper-trading tool. Signals are model outputs, not financial advice.")
st.markdown("---")

if 'access_token' not in st.session_state:
    st.session_state.access_token = None

# Step 1: Login & Get Broker Token
access_token = upstox_auth.get_upstox_access_token()

# Step 2: Fetch Live Market Data
df, model, feature_cols, live_price = market_data.fetch_live_market_data(access_token)

if df is not None and not df.empty:

    # -------------------------------------------------------------
    # VWAP & POC CALCULATION
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

    nifty_change_pct = index_correlation.get_nifty_change_pct(access_token, live_price)
    index_correlation_data, banknifty_correlation_note = index_correlation.get_bank_nifty_sensex(
        access_token, nifty_change_pct
    )

    df_global_sentiment, top_headline = global_news.get_global_market_sentiment()
    df_global_markets = global_markets.get_global_market_indices()
    global_sentiment_score, avg_market_change = global_markets.get_global_market_summary(df_global_markets)
    live_vix = global_markets.get_live_vix(df_global_markets)

    # -------------------------------------------------------------
    # REAL, WALK-FORWARD VALIDATED MODEL
    # -------------------------------------------------------------
    ml_results = ml_engine.train_and_backtest(df, live_vix=live_vix)

    # -------------------------------------------------------------
    # VOLATILITY REGIME FILTER
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
    # HYBRID AI ENGINE 
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

    signal_code = 1 if "BULLISH" in ai_analysis['bias_text'] else (-1 if "BEARISH" in ai_analysis['bias_text'] else 0)
    if ml_results.get("model_ready") and signal_code != 0 and ml_results.get("latest_signal", 0) != signal_code:
        signal_code = 0  

    final_signal_text = ai_analysis['signal_type'] if signal_code != 0 else "NO TRADE (ML model disagreement or choppy)"

    entry_price = ai_analysis['entry_price']
    sl = ai_analysis['stop_loss']
    t1 = ai_analysis['target_1']
    t2 = ai_analysis['target_2']
    intraday_atr = ai_analysis['tech_metrics'].get('atr', 15.0) * 0.4

    # -------------------------------------------------------------
    # DATABASE-DRIVEN TRADE STATE
    # -------------------------------------------------------------
    database.check_and_update_open_trades(live_price)
    open_trade = database.check_open_position()

    if open_trade is None and signal_code != 0:
        signal_label = "BUY" if signal_code == 1 else "SELL"
        trade_id, log_msg = database.log_entry_safe(signal_label, live_price, sl, t1, t2)
        
        # Telegram Alert (Confidence >= 45%)
        if trade_id and alert_sys and ai_analysis['confidence_pct'] >= 45:
            alert_sys.send_trade_alert(
                signal_type=final_signal_text, confidence=ai_analysis['confidence_pct'],
                price=live_price, sentiment=f"Global Score: {global_sentiment_score}",
                logic="Confluence + Walk-Forward ML agreement"
            )
        open_trade = database.check_open_position()
    elif open_trade is not None:
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
    # CORE METRICS 
    # -------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Real-Time Live Nifty Price (LTP)", value=f"₹ {live_price:,.2f}")
    with col2:
        st.metric(label="Institutional Confluence Signal", value=final_signal_text)
    with col3:
        st.metric(label="AI Confluence Score (live heuristic)", value=f"{ai_analysis['confidence_pct']}%")

    # (Wahan ka status alert/info message yahan se puri tarah hata diya gaya hai)

    # -------------------------------------------------------------
    # HYBRID AI ENGINE REASONING REPORT
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
    # INSTITUTIONAL CHART 
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
    # FII / DII FOOTPRINT & MAX PAIN
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
    # NIFTY INTERNAL BREADTH & HEAVYWEIGHTS TRACKER
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
    # BANK NIFTY, SENSEX & CORRELATED INDICES
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
    # DATA TABLES 
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
