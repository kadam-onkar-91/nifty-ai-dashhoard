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
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY_1") or st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ GEMINI_API_KEY not found. Please set GEMINI_API_KEY_1 in your secrets.toml file or Streamlit Cloud dashboard.")
    st.stop()

ai_engine = HybridAIEngine(api_key=GEMINI_API_KEY)

try:
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
    alert_sys = AlertManager(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
except Exception:
    alert_sys = None

# Initialize the trades database
database.init_db()

st.title("⚡ Nifty 50 Institutional AI Trading Dashboard (Pro Edition)")
st.caption("⚠️ Research/paper-trading tool. Signals are model outputs, not financial advice.")
st.markdown("---")

@st.fragment(run_every=30)
def _live_dashboard():
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None

    access_token = upstox_auth.get_upstox_access_token()
    df, model, feature_cols, live_price = market_data.fetch_live_market_data(access_token)

    if df is not None and not df.empty:
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
            st.info(f"{status_line} — market band hai, ye last traded price hai.")

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

        df_option_chain, oc_source = option_chain.generate_option_chain_data(live_price)
        fvg, ob, sweeps = smart_money.detect_smc_zones(df)
        market_structure = smart_money.detect_market_structure(df)
        smc_event = market_structure[0]["Market Event"] if market_structure else "Neutral Structure"

        df_heavyweights, advances, declines, breadth_ratio, breadth_status, heavyweights_debug = market_breadth.get_nifty_internal_breadth(access_token)
        df_full50, adv50, dec50, ratio50, breadth50_status, full50_debug = market_breadth.get_full_nifty50_breadth(access_token)
        max_pain = option_chain.calculate_max_pain(df_option_chain)
        fii_footprint, current_pcr = option_chain.get_fii_dii_fo_footprint(df_option_chain)

        nifty_change_pct = index_correlation.get_nifty_change_pct(live_price)
        index_correlation_data, banknifty_correlation_note = index_correlation.get_bank_nifty_sensex(
            access_token, nifty_change_pct
        )

        df_global_sentiment, top_headline = global_news.get_global_market_sentiment()
        df_global_markets = global_markets.get_global_market_indices()
        global_sentiment_score, avg_market_change = global_markets.get_global_market_summary(df_global_markets)
        live_vix = global_markets.get_live_vix(df_global_markets)

        ml_results = ml_engine.train_and_backtest(df, live_vix=live_vix)

        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

        avg_bb_width = df['BB_Width'].rolling(window=50).mean().iloc[-1]
        current_bb_width = df['BB_Width'].iloc[-1]
        is_choppy = current_bb_width < (avg_bb_width * 0.75) if pd.notna(avg_bb_width) and avg_bb_width > 0 else False

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

        level_prediction = support_resistance.predict_level_reaction(
            df=df, live_price=live_price, atr=ai_analysis['tech_metrics'].get('atr', 15.0),
            fvg_list=fvg, ob_list=ob, trend_bias=ai_analysis.get('tech_score', 0.0)
        )
        ai_analysis['level_prediction'] = level_prediction

        signal_code = 1 if "BULLISH" in ai_analysis['bias_text'] else (-1 if "BEARISH" in ai_analysis['bias_text'] else 0)
        if ml_results.get("model_ready") and signal_code != 0 and ml_results.get("latest_signal", 0) != signal_code:
            signal_code = 0

        final_signal_text = ai_analysis['signal_type'] if signal_code != 0 else "NO TRADE (ML model disagreement or choppy)"
        intraday_atr = ai_analysis['tech_metrics'].get('atr', 15.0) * 0.4

        database.check_and_update_open_trades(live_price)
        open_trade = database.check_open_position()

        entry_price = open_trade["entry_price"] if open_trade else None
        sl = open_trade["stop_loss"] if open_trade else None
        t1 = open_trade.get("target_1") if open_trade else None
        t2 = open_trade.get("target_2") if open_trade else None

        perf = database.fetch_performance_metrics()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Real-Time Live Nifty Price (LTP)", value=f"₹ {live_price:,.2f}")
        with col2:
            st.metric(label="Institutional Confluence Signal", value=final_signal_text)
        with col3:
            st.metric(label="AI Confluence Score (live heuristic)", value=f"{ai_analysis['confidence_pct']}%")

        if open_trade:
            st.warning(f"🛡️ **ACTIVE TRADE** | {open_trade['signal']} | "
                       f"Entry: ₹{open_trade['entry_price']:.2f} | Current SL: ₹{open_trade['stop_loss']:.2f}")
        else:
            st.info("ℹ️ No auto-trade is taken. Review the signal and Execution Plan below.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Real Performance Tracking")

        # Safe metric extraction avoiding KeyErrors
        win_rate_val = perf.get('win_rate') if isinstance(perf, dict) else None
        win_rate_str = f"{win_rate_val}%" if win_rate_val is not None else "N/A"
        total_trades_val = perf.get('total_trades', 0) if isinstance(perf, dict) else 0

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Walk-Forward ML Accuracy", ml_results.get("Accuracy", "N/A"))
        rc2.metric("ML Sample Size", ml_results.get("Sample Size", 0))
        rc3.metric("Live DB Win Rate", win_rate_str)
        rc4.metric("Total Closed Trades", total_trades_val)

        dashboard_context = {
            "Live Nifty Price": f"₹{live_price:,.2f}",
            "Institutional Signal": final_signal_text,
            "AI Confidence Score": f"{ai_analysis['confidence_pct']}%",
            "Market Trend Bias": ai_analysis['bias_text'],
            "SMC Market Structure": smc_event,
            "VWAP": f"₹{ai_analysis['tech_metrics'].get('vwap', 0):,.2f}",
            "POC (Volume Profile)": f"₹{ai_analysis['tech_metrics'].get('poc', 0):,.2f}",
            "PCR (Put-Call Ratio)": current_pcr,
            "Max Pain": max_pain,
            "FII/DII Sentiment": fii_footprint,
            "India VIX": live_vix,
            "Global Market Sentiment Score": global_sentiment_score,
            "Bank Nifty/Sensex Correlation Note": banknifty_correlation_note,
            "Nifty Heavyweights Breadth": breadth_status,
            "Full Nifty 50 Breadth": breadth50_status,
            "Suggested Entry": f"₹{entry_price:,.2f}" if entry_price else "N/A",
            "Suggested Stop Loss": f"₹{sl:,.2f}" if sl else "N/A",
            "Suggested Target 1": f"₹{t1:,.2f}" if t1 else "N/A",
            "Suggested Target 2": f"₹{t2:,.2f}" if t2 else "N/A",
        }

        st.session_state.current_dashboard_context = dashboard_context
    else:
        st.warning("⚠️ Live market data fetch nahi ho paya. Upstox token ya network check karein.")

_live_dashboard()

st.markdown("---")

ctx_to_pass = st.session_state.get("current_dashboard_context", {})
render_ai_chat(gemini_api_key=GEMINI_API_KEY, dashboard_context=ctx_to_pass)
