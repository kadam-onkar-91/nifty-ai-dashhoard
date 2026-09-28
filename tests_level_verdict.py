"""
tests_level_verdict.py -- v17 regression tests for the OI bounce-or-break behaviour.
Run:  python -m unittest tests_level_verdict
Covers: bounce only when data leans bounce; break trade only AFTER a confirmed break and only when data leans break;
no entry before the break; drought counter uses market minutes; bad-form penalty expires.
"""
import sys, types, unittest
for _m in ("feedparser", "streamlit", "yfinance", "upstox_client", "supabase", "plotly"):
    try:
        __import__(_m)
    except Exception:
        sys.modules[_m] = types.ModuleType(_m)
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, collections
import trade_learning, ai_trade_decision, indicators
trade_learning.init_db()
def make_df(closes, start="2026-09-24 09:15"):
    idx = pd.date_range(start, periods=len(closes), freq="5min")
    c = np.array(closes); o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 3, "Low": np.minimum(o, c) - 3, "Close": c, "Volume": 1000.0}, index=idx)
    df["EMA_20"] = df.Close.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df.Close.ewm(span=50, adjust=False).mean()
    tr = pd.concat([df.High - df.Low, (df.High - df.Close.shift()).abs(), (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean().fillna(10)
    df["RSI"] = indicators.calculate_rsi(df, 14).fillna(50)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = indicators.calculate_macd(df)
    df["VWAP"] = df.Close.expanding().mean()
    return df


def run(mode, data="bounce", verbose=True):
    """mode: 'support_bounce' | 'resistance_reject' | 'break_down_confirmed' | 'break_down_unconfirmed'"""
    ai_trade_decision._outside_entry_window = lambda: False
    rng = np.random.default_rng(11)
    closes = [22850 + 10*np.sin(i/6) + rng.normal(0, 3) for i in range(40)]
    x = closes[-1]
    for _ in range(26):
        x += -1.5 + rng.normal(0, 4); closes.append(x)          # fall toward ~22,810
    df = make_df(closes)
    last = df.index[-1]
    if mode.startswith("break"):
        rng2 = np.random.default_rng(5)
        closes = [22750, 22770, 22795] + [22826 + 4*np.sin(i/3) + rng2.normal(0, 2.5) for i in range(57)]
        closes += [22818, 22815, 22813, 22801, 22797, 22800, 22798, 22802]
        df = make_df(closes)
    if mode in ("support_bounce",):
        # rejection candle at 22,808: long lower wick, bullish close
        df.iloc[-1, df.columns.get_loc("Open")] = 22807.0; df.iloc[-1, df.columns.get_loc("Close")] = 22813.0
        df.iloc[-1, df.columns.get_loc("Low")] = 22800.0;  df.iloc[-1, df.columns.get_loc("High")] = 22814.0
        live = 22813.0
    else:
        live = float(df.Close.iloc[-1])
    atr = float(df.ATR.iloc[-1])
    put_txt = "Heavy Put OI writing near this support (Put OI 4,717,804 vs Call OI 2,803,100)"
    ladder = {"supports": [{"level_price": 22808.0, "factors": [put_txt], "bounce_pct": 70, "break_pct": 30, "directional_bias": "bullish", "distance_pts": abs(live-22808)}],
              "resistances": []}
    conf = {}
    if mode == "support_bounce": conf["support_rejection_confirmed"] = True
    if mode == "break_down_confirmed": conf.update({"breakdown_confirmed": True, "breakdown_retest_confirmed": True, "volume_expansion": True})
    sr = {"sr_engine": "REAL_SR_V2", "confirmation": conf, "entry_timing": {},
          "zones": [{"price": 22808.0, "actionable": True, "grade": "A", "sources": ["Option Chain OI wall", "Hourly swing low"]}]}
    bullish = data == "bounce"
    ctx = {"Data Freshness": "Live / fresh",
           "RAW Regime Engine": "{'primary': '%s', 'structure': 'X'}" % ("BULLISH" if bullish else "BEARISH"),
           "RAW Multi-Timeframe Structure": ("bullish bullish bullish" if bullish else "bearish bearish bearish"),
           "RAW Order Flow": ("BUYING PRESSURE" if bullish else "SELLING PRESSURE")}
    out = ai_trade_decision.generate_trade_decision(
        live_price=live, level_prediction=None, atr=atr, max_pain=(22900 if bullish else 22700),
        signal_code=(1 if bullish else -1), ml_agrees=True, breadth_advances=(35 if bullish else 12), breadth_declines=(15 if bullish else 38),
        global_avg_change=(0.4 if bullish else -0.4), india_news_sentiment=("BULLISH" if bullish else "BEARISH"),
        nifty50_news_sentiment=("BULLISH" if bullish else "BEARISH"), sniper_bias=("BULLISH" if bullish else "BEARISH"),
        level_ladder=ladder, sr_context=sr, df=df, pcr=(1.3 if bullish else 0.7), oc_source="LIVE", dashboard_context=ctx)
    return out, atr



class LevelVerdictTests(unittest.TestCase):
    def test_bounce_confirmed_and_data_agrees_takes_buy(self):
        out, _ = run("support_bounce", "bounce")
        self.assertTrue(out["has_setup"]); self.assertEqual(out["direction"], "BUY")
        self.assertEqual(out["level_verdict"]["state"], "BOUNCE_CONFIRMED")

    def test_bounce_candle_but_data_leans_break_is_skipped(self):
        out, _ = run("support_bounce", "break")
        self.assertFalse(out["has_setup"])

    def test_no_sell_before_break_is_confirmed(self):
        out, _ = run("break_down_unconfirmed", "break")
        self.assertFalse(out["has_setup"])

    def test_sell_after_confirmed_break_of_oi_wall_when_data_agrees(self):
        out, _ = run("break_down_confirmed", "break")
        self.assertTrue(out["has_setup"]); self.assertEqual(out["direction"], "SELL")
        self.assertEqual(out["playbook"], "BREAKDOWN_SELL")

    def test_confirmed_break_but_data_leans_bounce_is_skipped(self):
        out, _ = run("break_down_confirmed", "bounce")
        self.assertFalse(out["has_setup"])

    def test_market_minutes_ignore_nights_and_weekends(self):
        from datetime import datetime
        f = ai_trade_decision._market_minutes_between
        # Fri 15:00 -> Mon 09:45  = 30 min (Fri) + 30 min (Mon)
        self.assertAlmostEqual(f(datetime(2026, 9, 25, 15, 0), datetime(2026, 9, 28, 9, 45)), 60.0)
        self.assertEqual(f(datetime(2026, 9, 26, 10, 0), datetime(2026, 9, 27, 14, 0)), 0.0)   # Sat->Sun

    def test_bad_form_penalty_expires(self):
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(hours=60)).strftime("%Y-%m-%d %H:%M:%S")
        rows = [{"status": "LOSS", "probe": False, "timestamp": old, "exit_timestamp": old}] * 8 + \
               [{"status": "WIN", "probe": False, "timestamp": old, "exit_timestamp": old}] * 0
        orig = trade_learning.get_recent_setups
        trade_learning.get_recent_setups = lambda limit=40: rows
        try:
            perf = trade_learning.recent_performance()
        finally:
            trade_learning.get_recent_setups = orig
        self.assertFalse(perf["bad"]); self.assertTrue(perf["stale_bad"])


if __name__ == "__main__":
    unittest.main()
