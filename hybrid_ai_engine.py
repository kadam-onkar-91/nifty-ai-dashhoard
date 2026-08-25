import pandas as pd
import numpy as np

# Naya Google GenAI Package Import (google-genai SDK)
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

class HybridAIEngine:
    """
    Institutional Grade Hybrid AI Logic Engine for Nifty 50 Real-Time Trading.
    Upgraded with Multi-Factor Confluence, Fake Breakout Guards, and Advanced Risk Architecture.
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
            return 0.0, {'rsi': 50, 'macd': 0, 'ema_20': live_price, 'ema_50': live_price, 'vwap': live_price, 'atr': 15.0, 'avg_pcr': 1.0}

        latest = df.iloc[-1]
        rsi = float(latest.get('RSI', 50))
        macd = float(latest.get('MACD', 0))
        ema_20 = float(latest.get('EMA_20', live_price))
        ema_50 = float(latest.get('EMA_50', live_price))
        vwap = float(latest.get('VWAP', live_price))
        atr = float(latest.get('ATR', 15.0))

        score = 0.0

        # RSI Momentum Confluence
        if rsi > 60: score += 0.25
        elif rsi < 40: score -= 0.25
        elif rsi > 50: score += 0.1
        else: score -= 0.1

        # Trend & VWAP Confirmation
        if live_price > vwap and live_price > ema_20 > ema_50: score += 0.40
        elif live_price < vwap and live_price < ema_20 < ema_50: score -= 0.40
        elif live_price > ema_20: score += 0.15
        elif live_price < ema_20: score -= 0.15

        # MACD Histogram Momentum
        if macd > 0: score += 0.15
        else: score -= 0.15

        # Option Chain PCR Confluence
        avg_pcr = 1.0
        if df_option_chain is not None and not df_option_chain.empty and 'PCR' in df_option_chain.columns:
            try: avg_pcr = float(df_option_chain['PCR'].mean())
            except Exception: avg_pcr = 1.0

        if avg_pcr > 1.25: score += 0.25
        elif avg_pcr > 1.05: score += 0.10
        elif avg_pcr < 0.75: score -= 0.25
        elif avg_pcr < 0.95: score -= 0.10

        tech_score = max(-1.0, min(1.0, score))
        metrics = {
            'rsi': rsi, 'macd': macd, 'ema_20': ema_20, 'ema_50': ema_50, 
            'vwap': vwap, 'atr': atr if atr > 0 else 15.0, 'avg_pcr': avg_pcr
        }
        return tech_score, metrics

    def calculate_smc_score(self, smc_data):
        if not smc_data: return 0.0, "Neutral Structure"
        event = str(smc_data).lower()
        score = 0.0
        if 'bos - bullish' in event or 'bullish continuation' in event: score += 0.85
        elif 'choch - bullish' in event or 'bullish reversal' in event: score += 0.95
        elif 'bos - bearish' in event or 'bearish continuation' in event: score -= 0.85
        elif 'choch - bearish' in event or 'bearish reversal' in event: score -= 0.95
        elif 'bullish' in event: score += 0.45
        elif 'bearish' in event: score -= 0.45
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
        if 'STRONG' in breadth_str or 'POSITIVE' in breadth_str: score += 0.35
        elif 'WEAK' in breadth_str or 'NEGATIVE' in breadth_str: score -= 0.35

        # Choppy market penalty / override factor
        if is_choppy:
            score *= 0.15  # Heavy damping to eliminate false signals in chop zones

        return max(-1.0, min(1.0, score))

    def analyze(self, live_price, df, df_option_chain, smc_data, sentiment_score, global_avg_change, fii_footprint="Neutral", breadth_status="Neutral", is_choppy=False):
        tech_score, tech_metrics = self.calculate_technical_score(live_price, df, df_option_chain)
        smc_score, smc_desc = self.calculate_smc_score(smc_data)
        macro_score = self.calculate_macro_score(global_avg_change, sentiment_score)
        flow_score = self.calculate_institutional_flow_score(fii_footprint, breadth_status, is_choppy)

        # Weighted institutional model matrix
        net_score = (tech_score * 0.20) + (smc_score * 0.25) + (macro_score * 0.15) + (flow_score * 0.40)
        
        if is_choppy:
            net_score = 0.0 # Strict Volatility Filter Lock

        confidence_pct = round(abs(net_score) * 100, 1)

        if is_choppy:
            bias_text, signal_type, entry_type = "CHOPPY / NO TRADE ⚠️", "NO TRADE ZONE (Low Volatility)", "Wait for Volatility Expansion"
        elif net_score >= 0.22:
            bias_text, signal_type, entry_type = "BULLISH 🟢", "STRONG BUY (CALL)", "Buy-on-Dip / Breakout Hold with Retest"
        elif net_score <= -0.22:
            bias_text, signal_type, entry_type = "BEARISH 🔴", "STRONG SELL (PUT)", "Sell-on-Rally / Breakdown Retest"
        else:
            bias_text, signal_type, entry_type = "SIDEWAYS / CONSOLIDATION 🟡", "NO TRADE / RANGE BOUND", "Wait for Level Breakout"

        atr = tech_metrics.get('atr', 15.0)
        sl_points = max(16.0, round(atr * 0.75, 2))
        t1_points = round(sl_points * 1.5, 2)
        t2_points = round(sl_points * 2.8, 2)

        if "BEARISH" in bias_text:
            entry_price, stop_loss, target_1, target_2 = live_price, live_price + sl_points, live_price - t1_points, live_price - t2_points
            invalidation_level, breakdown_level = live_price + (sl_points * 1.15), live_price - (atr * 0.4)
        elif "BULLISH" in bias_text:
            entry_price, stop_loss, target_1, target_2 = live_price, live_price - sl_points, live_price + t1_points, live_price + t2_points
            invalidation_level, breakdown_level = live_price - (sl_points * 1.15), live_price + (atr * 0.4)
        else:
            entry_price, stop_loss, target_1, target_2 = live_price, live_price - sl_points, live_price + (sl_points * 1.5), live_price - (sl_points * 1.5)
            invalidation_level, breakdown_level = live_price + (atr * 1.5), live_price + (atr * 1.0)

        return {
            'net_score': round(net_score, 2), 'confidence_pct': confidence_pct, 'bias_text': bias_text,
            'signal_type': signal_type, 'entry_type': entry_type, 'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2), 'target_1': round(target_1, 2), 'target_2': round(target_2, 2),
            'sl_points': sl_points, 't1_points': t1_points, 't2_points': t2_points,
            'invalidation_level': round(invalidation_level, 2), 'breakdown_level': round(breakdown_level, 2),
            'risk_reward_ratio': "1 : 2.5", 'tech_score': tech_score, 'smc_score': smc_score,
            'macro_score': macro_score, 'flow_score': flow_score, 'tech_metrics': tech_metrics, 
            'smc_desc': smc_desc, 'fii_footprint': fii_footprint, 'breadth_status': breadth_status, 'is_choppy': is_choppy
        }

    def generate_llm_reasoning(self, live_price, meta_summary, sentiment_score, global_avg_change):
        m = meta_summary
        tm = m['tech_metrics']

        if self.client and HAS_GENAI:
            try:
                prompt = f"""
You are an Elite Institutional Quant Trader & Hedge Fund Lead in India analyzing Nifty 50 real-time order flow.
Write a deep, razor-sharp, highly confident institutional trade reasoning report in conversational Hinglish (Hindi + English mix).
You MUST explicitly synthesize ALL the following live parameters in your breakdown:

Live Market Data & Quant Metrics:
- Spot Price: ₹{live_price}
- Institutional Bias: {m['bias_text']} (Conviction Confidence: {m['confidence_pct']}%)
- Recommended Action: {m['signal_type']}
- Exact Entry Level: ₹{m['entry_price']}
- Stop Loss: ₹{m['stop_loss']} ({m['sl_points']} pts risk)
- Targets: T1 = ₹{m['target_1']}, T2 = ₹{m['target_2']}
- Invalidation / Cancellation Level: ₹{m['invalidation_level']}
- Technical Indicators: RSI = {tm['rsi']:.2f}, VWAP = ₹{tm['vwap']:.2f}, EMA 20 = {tm['ema_20']:.2f}, ATR = {tm['atr']:.2f}
- Derivatives & PCR: Option Chain Avg PCR = {tm['avg_pcr']:.2f}
- Smart Money Concepts (SMC): {m['smc_desc']}
- Institutional Footprint: FII Bias = {m['fii_footprint']} | Internal Breadth = {m['breadth_status']}
- Volatility Status: {"Choppy / Rangebound (No Trade Guard Active)" if m['is_choppy'] else "Active Volatility Expansion"}
- Global Macro: Avg Change = {global_avg_change}% | Sentiment = {sentiment_score}

Mandatory Tone & Structure:
1. Start exact tone: "Haan. Maine chart + options chain + FII footprint + internal breadth + global news data sab read kar liya hai..."
2. Explicitly analyze how FII data and internal breadth align or conflict with the technical price action.
3. Clearly mention the exact entry trigger, fake breakout warning, and strict invalidation rule to protect capital.
NO financial disclaimers. Make it read like a professional desk note from a top proprietary trading desk.
"""
                response = self.client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt
                )
                return response.text
            except Exception as e:
                pass 

        # Professional Quant Fallback Report
        return f"""
### 🦅 Institutional Quant AI Deep Reasoning & Execution Plan

**Haan. Maine chart + Options Chain + FII Footprint + Internal Breadth + SMC Order Blocks + Global News data sab read kar liya hai.**
Nifty Spot **₹{live_price:,.2f}** par trade kar raha hai. VWAP: **₹{tm['vwap']:,.2f}** | EMA 20: **₹{tm['ema_20']:,.2f}** | RSI: **{tm['rsi']:.2f}** | PCR: **{tm['avg_pcr']:.2f}** | FII Footprint: **{m['fii_footprint']}**.

---

### 📊 Model Decision: **{m['bias_text']}** (Confidence Score: {m['confidence_pct']}%)

#### 🎯 Primary Execution Plan:
* **Actionable Setup:** {m['signal_type']}
* **Execution Strategy:** {m['entry_type']} at **₹{m['entry_price']:,.2f}**
* **Stop Loss (SL):** **₹{m['stop_loss']:,.2f}** (Risk: {m['sl_points']} pts)
* **Target 1 (T1):** **₹{m['target_1']:,.2f}**
* **Target 2 (T2):** **₹{m['target_2']:,.2f}**
* **Risk-to-Reward:** {m['risk_reward_ratio']}

---

### 🧠 Multi-Factor Confluence Breakdown:
1. **FII / DII Footprint & Breadth:** FII bias is **{m['fii_footprint']}** and internal breadth is **{m['breadth_status']}**, confirming institutional participation.
2. **Smart Money Structure:** Current SMC Event = **"{m['smc_desc']}"** with Volatility State: {"Choppy ⚠️" if m['is_choppy'] else "Active 🟢"}.
3. **Global Macro Context:** Global sentiment is **{sentiment_score}** with an average change of **{global_avg_change}%**.

⚠️ **Trap Warning & Confirmation Rule:**
Do not enter blindly on a direct touch. Wait for a candle confirmation close near **₹{m['entry_price']:,.2f}** followed by a valid retest before executing.

🛑 **Invalidation Rule:**
Agar price **₹{m['invalidation_level']:,.2f}** ke opposite side sustain kar jata hai, toh ye poora setup cancel ho jayega aur position immediate cut karni hai.
"""
