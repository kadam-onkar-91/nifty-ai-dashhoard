import pandas as pd
import numpy as np

# Naya Google GenAI Package Import
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

class HybridAIEngine:
    """
    Institutional Grade Hybrid AI Logic Engine for Nifty 50 Real-Time Trading.
    Upgraded to ingest FII Footprint, Internal Breadth, and Volatility Regimes.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        if self.api_key and HAS_GENAI:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                pass

    def calculate_technical_score(self, live_price, df, df_option_chain):
        if df is None or df.empty:
            return 0.0, {'rsi': 50, 'macd': 0, 'ema_20': live_price, 'ema_50': live_price, 'atr': 15.0, 'avg_pcr': 1.0}

        latest = df.iloc[-1]
        rsi = float(latest.get('RSI', 50))
        macd = float(latest.get('MACD', 0))
        ema_20 = float(latest.get('EMA_20', live_price))
        ema_50 = float(latest.get('EMA_50', live_price))
        atr = float(latest.get('ATR', 15.0))

        score = 0.0

        if rsi > 60: score += 0.25
        elif rsi < 40: score -= 0.25
        elif rsi > 50: score += 0.1
        else: score -= 0.1

        if live_price > ema_20 > ema_50: score += 0.35
        elif live_price < ema_20 < ema_50: score -= 0.35
        elif live_price > ema_20: score += 0.15
        elif live_price < ema_20: score -= 0.15

        if macd > 0: score += 0.15
        else: score -= 0.15

        avg_pcr = 1.0
        if df_option_chain is not None and not df_option_chain.empty and 'PCR' in df_option_chain.columns:
            try: avg_pcr = float(df_option_chain['PCR'].mean())
            except Exception: avg_pcr = 1.0

        if avg_pcr > 1.3: score += 0.25
        elif avg_pcr > 1.1: score += 0.10
        elif avg_pcr < 0.7: score -= 0.25
        elif avg_pcr < 0.9: score -= 0.10

        tech_score = max(-1.0, min(1.0, score))
        metrics = {'rsi': rsi, 'macd': macd, 'ema_20': ema_20, 'ema_50': ema_50, 'atr': atr if atr > 0 else 15.0, 'avg_pcr': avg_pcr}
        return tech_score, metrics

    def calculate_smc_score(self, smc_data):
        if not smc_data: return 0.0, "Neutral Structure"
        event = str(smc_data).lower()
        score = 0.0
        if 'bos - bullish' in event or 'bullish continuation' in event: score += 0.80
        elif 'choch - bullish' in event or 'bullish reversal' in event: score += 0.90
        elif 'bos - bearish' in event or 'bearish continuation' in event: score -= 0.80
        elif 'choch - bearish' in event or 'bearish reversal' in event: score -= 0.90
        elif 'bullish' in event: score += 0.40
        elif 'bearish' in event: score -= 0.40
        return max(-1.0, min(1.0, score)), smc_data

    def calculate_macro_score(self, global_avg_change, sentiment_score):
        score = 0.0
        if global_avg_change <= -0.8: score -= 0.60
        elif global_avg_change <= -0.3: score -= 0.30
        elif global_avg_change >= 0.8: score += 0.60
        elif global_avg_change >= 0.3: score += 0.30

        sent = str(sentiment_score).lower()
        if 'strong bearish' in sent: score -= 0.40
        elif 'bearish' in sent: score -= 0.20
        elif 'strong bullish' in sent: score += 0.40
        elif 'bullish' in sent: score += 0.20
        return max(-1.0, min(1.0, score))

    def calculate_institutional_flow_score(self, fii_footprint, breadth_status, is_choppy):
        score = 0.0
        fii_str = str(fii_footprint).upper()
        breadth_str = str(breadth_status).upper()

        # FII Footprint scoring
        if 'BULLISH' in fii_str: score += 0.50
        elif 'BEARISH' in fii_str: score -= 0.50

        # Internal Breadth scoring
        if 'STRONG' in breadth_str or 'POSITIVE' in breadth_str: score += 0.30
        elif 'WEAK' in breadth_str or 'NEGATIVE' in breadth_str: score -= 0.30

        # Choppy market penalty / override factor
        if is_choppy:
            score *= 0.2  # Dampen score heavily if market is choppy/sideways

        return max(-1.0, min(1.0, score))

    def analyze(self, live_price, df, df_option_chain, smc_data, sentiment_score, global_avg_change, fii_footprint="Neutral", breadth_status="Neutral", is_choppy=False):
        tech_score, tech_metrics = self.calculate_technical_score(live_price, df, df_option_chain)
        smc_score, smc_desc = self.calculate_smc_score(smc_data)
        macro_score = self.calculate_macro_score(global_avg_change, sentiment_score)
        flow_score = self.calculate_institutional_flow_score(fii_footprint, breadth_status, is_choppy)

        # Weighted final score including all new institutional metrics
        net_score = (tech_score * 0.25) + (smc_score * 0.25) + (macro_score * 0.15) + (flow_score * 0.35)
        
        if is_choppy:
            net_score = 0.0 # Force neutral if choppy volatility filter triggers

        confidence_pct = round(abs(net_score) * 100, 1)

        if is_choppy:
            bias_text, signal_type, entry_type = "CHOPPY / NO TRADE ⚠️", "NO TRADE ZONE (Low Volatility)", "Wait for Volatility Expansion"
        elif net_score >= 0.25:
            bias_text, signal_type, entry_type = "BULLISH 🟢", "BUY (CALL)", "Buy-on-Dip / Breakout Hold"
        elif net_score <= -0.25:
            bias_text, signal_type, entry_type = "BEARISH 🔴", "SELL (PUT)", "Sell-on-Rally / Breakdown Retest"
        else:
            bias_text, signal_type, entry_type = "SIDEWAYS / CONSOLIDATION 🟡", "NO TRADE / RANGE BOUND", "Wait for Clear Level Breakout"

        atr = tech_metrics.get('atr', 15.0)
        sl_points = max(18.0, round(atr * 0.85, 2))
        t1_points = round(sl_points * 1.5, 2)
        t2_points = round(sl_points * 2.5, 2)

        if "BEARISH" in bias_text:
            entry_price, stop_loss, target_1, target_2 = live_price, live_price + sl_points, live_price - t1_points, live_price - t2_points
            invalidation_level, breakdown_level = live_price + (sl_points * 1.2), live_price - (atr * 0.5)
        elif "BULLISH" in bias_text:
            entry_price, stop_loss, target_1, target_2 = live_price, live_price - sl_points, live_price + t1_points, live_price + t2_points
            invalidation_level, breakdown_level = live_price - (sl_points * 1.2), live_price + (atr * 0.5)
        else:
            entry_price, stop_loss, target_1, target_2 = live_price, live_price - sl_points, live_price + (sl_points * 1.5), live_price - (sl_points * 1.5)
            invalidation_level, breakdown_level = live_price + (atr * 1.5), live_price + (atr * 1.0)

        return {
            'net_score': round(net_score, 2), 'confidence_pct': confidence_pct, 'bias_text': bias_text,
            'signal_type': signal_type, 'entry_type': entry_type, 'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2), 'target_1': round(target_1, 2), 'target_2': round(target_2, 2),
            'sl_points': sl_points, 't1_points': t1_points, 't2_points': t2_points,
            'invalidation_level': round(invalidation_level, 2), 'breakdown_level': round(breakdown_level, 2),
            'risk_reward_ratio': "1 : 2.0", 'tech_score': tech_score, 'smc_score': smc_score,
            'macro_score': macro_score, 'flow_score': flow_score, 'tech_metrics': tech_metrics, 
            'smc_desc': smc_desc, 'fii_footprint': fii_footprint, 'breadth_status': breadth_status, 'is_choppy': is_choppy
        }

    def generate_llm_reasoning(self, live_price, meta_summary, sentiment_score, global_avg_change):
        m = meta_summary
        tm = m['tech_metrics']

        # Fully updated prompt containing ALL features (FII, Breadth, Volatility, SMC, PCR, etc.)
        if self.client and HAS_GENAI:
            try:
                prompt = f"""
You are a top-tier Institutional Quant & Hedge Fund Lead Trader in India analyzing Nifty 50 intraday.
Write a deep, analytical, confident, and structured trade reasoning report in conversational Hinglish (Hindi + English mix).
You MUST explicitly consider and mention ALL the following dimensions in your response:

Live Market Context & Institutional Metrics:
- Spot Price: ₹{live_price}
- Calculated Directional Bias: {m['bias_text']}
- Meta Score Confidence: {m['confidence_pct']}%
- Signal Type: {m['signal_type']}
- Entry Price: ₹{m['entry_price']}
- Stop Loss: ₹{m['stop_loss']}
- Target 1: ₹{m['target_1']} | Target 2: ₹{m['target_2']}
- Invalidation Level: ₹{m['invalidation_level']}
- Technicals: RSI = {tm['rsi']:.2f}, EMA 20 = {tm['ema_20']:.2f}, EMA 50 = {tm['ema_50']:.2f}, ATR = {tm['atr']:.2f}
- Option Chain Avg PCR: {tm['avg_pcr']:.2f}
- Smart Money Concepts (SMC Event): {m['smc_desc']}
- FII / DII F&O Footprint: {m['fii_footprint']}
- Nifty Internal Breadth Status: {m['breadth_status']}
- Volatility Regime / Choppy Status: {"Choppy / Sideways" if m['is_choppy'] else "Normal Active Volatility"}
- Global Macro Avg Change: {global_avg_change}% | Sentiment: {sentiment_score}

Start exact tone like: "Haan. Maine chart + options chain + FII footprint + internal breadth + global news data sab read kar liya hai..."
Explicitly explain how FII data and Internal Breadth align or conflict with technicals. If the market is choppy, emphasize why capital protection matters. NO financial disclaimer.
"""
                response = self.client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt
                )
                return response.text
            except Exception as e:
                pass 

        # High-Quality Quant Fallback Report (Agar API kaam na kare)
        pcr_desc = "Put writers heavy hain (Bullish support)" if tm['avg_pcr'] > 1.1 else "Call writers aggressive hain (Resistance)" if tm['avg_pcr'] < 0.9 else "PCR Neutral zone mein hai"
        macro_desc = "Global market pressure heavy hai" if global_avg_change < -0.4 else "Global markets supportive hain" if global_avg_change > 0.4 else "Global markets flat hain"

        return f"""
### 🦅 Quant AI Deep Reasoning & Execution Plan (All-Feature Integrated)

**Haan. Maine chart + Options Chain + FII Footprint + Internal Breadth + SMC Order Blocks + Global News data sab read kar liya hai.**
Abhi Nifty Spot **₹{live_price:,.2f}** par hai. EMA 20: **₹{tm['ema_20']:,.2f}** | RSI: **{tm['rsi']:.2f}** | Average PCR: **{tm['avg_pcr']:.2f}** | FII Bias: **{m['fii_footprint']}**.

---

### 📊 Model Decision: **{m['bias_text']}** (Signal Strength: {m['confidence_pct']}%)

#### 🎯 Primary Execution Levels (Spot):
* **Action:** {m['signal_type']}
* **Entry Strategy:** {m['entry_type']} (At **₹{m['entry_price']:,.2f}**)
* **Stop Loss (SL):** **₹{m['stop_loss']:,.2f}** (Risk: {m['sl_points']} pts)
* **Target 1 (T1):** **₹{m['target_1']:,.2f}** (Reward: {m['t1_points']} pts)
* **Target 2 (T2):** **₹{m['target_2']:,.2f}** (Reward: {m['t2_points']} pts)
* **Risk-to-Reward Ratio:** {m['risk_reward_ratio']}

---

### 🧠 Institutional Flow & Multi-Factor Logic:
1. **FII / DII F&O Footprint:** {m['fii_footprint']}
2. **Nifty Internal Breadth:** {m['breadth_status']} (Heavyweights backing)
3. **Global Macro vs Domestic:** {macro_desc} (Global Avg Change: **{global_avg_change}%**, Sentiment: **{sentiment_score}**).
4. **Smart Money Structure:** Current SMC Event = **"{m['smc_desc']}"** with Volatility Regime: {"Choppy ⚠️" if m['is_choppy'] else "Active 🟢"}.

⚠️ **Trap Warning & Confirmation Rule:**
Directly jump mat karna. 5-minute candle ko **₹{m['entry_price']:,.2f}** level ke pass close hone do. Retest confirmation ke baad hi position build karna.

🛑 **Invalidation Rule (Trade Cancellation):**
Agar price **₹{m['invalidation_level']:,.2f}** ke opposite side cross kar deta hai, to ye bias CANCEL ho jayega aur position immediate cut/square-off karni hai.
"""