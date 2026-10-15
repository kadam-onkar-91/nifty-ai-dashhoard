"""v47 tests: no entry INTO a rejecting ceiling/floor (break not confirmed); breadth ticker handling.  python -m unittest tests_v47"""
import unittest
import numpy as np
import pandas as pd
import entry_quality as eq


def _df(rows, start="2026-10-15 09:15"):
    idx = pd.date_range(start, periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=0.0)


def _rally_then_press(final_close=23041.5):
    rows, p = [], 22700.0
    for i in range(40):                       # steady rally, 6-pt candles
        o, c = p, p + 6.0
        rows.append((o, c + 2, o - 2, c)); p = c
    # a 23,000 base, then 4 candles that poke 23,050 and get rejected (long upper wicks)
    rows += [(23030, 23050, 23028, 23036), (23036, 23050, 23032, 23040), (23040, 23049, 23034, 23038), (23038, 23050, 23033, 23039)]
    rows += [(23039, 23042, 23036, final_close)]    # forming candle
    return _df(rows)


class Pressing(unittest.TestCase):
    def test_buy_under_rejecting_ceiling_is_flagged(self):
        r = eq.assess_entry(_rally_then_press(), "BUY", atr=12.0)
        self.assertIsNotNone(r.get("pressing"))
        self.assertEqual(r["pressing"]["level"], 23050.0)
        self.assertGreaterEqual(r["pressing"]["rejections"], 2)

    def test_bounce_side_is_not_flagged(self):
        # SELL from the same ceiling is a rejection trade, not pressing into a floor
        r = eq.assess_entry(_rally_then_press(), "SELL", atr=12.0)
        self.assertIsNone(r.get("pressing"))

    def test_after_the_break_closes_it_is_not_flagged(self):
        df = _rally_then_press()
        rows = df[["Open", "High", "Low", "Close"]].values.tolist()
        rows[-3] = [23039, 23062, 23037, 23058]      # two closed candles close ABOVE 23,050
        rows[-2] = [23058, 23066, 23054, 23061]
        rows[-1] = [23061, 23064, 23057, 23062]
        r = eq.assess_entry(_df(rows), "BUY", atr=12.0)
        self.assertIsNone(r.get("pressing"))

    def test_clean_trend_is_not_flagged(self):
        rows, p = [], 22700.0
        for i in range(50):
            o, c = p, p + 5.0
            rows.append((o, c + 1, o - 1, c)); p = c
        r = eq.assess_entry(_df(rows), "BUY", atr=10.0)
        self.assertIsNone(r.get("pressing"))

    def test_engine_blocks_it(self):
        import ai_trade_decision as A
        self.assertTrue(A.PRESSING_LEVEL_BLOCK)
        self.assertEqual(A.classify_block("Pressing into level: x"), "Pressing into level (break not closed yet)")


class Persist(unittest.TestCase):
    def test_persist_store_exists_and_is_dict(self):
        import ai_trade_decision as A
        self.assertIsInstance(A.PERSIST, dict)

    def test_app_no_longer_keeps_review_state_in_session(self):
        src = open("app.py", encoding="utf-8").read()
        for k in ("_fl_carry", "_fl_plans", "_fl_plan_budget", "_gemini_trade_review_cache"):
            self.assertNotIn(f"st.session_state.get('{k}'", src)
            self.assertNotIn(f"st.session_state.setdefault('{k}'", src)


class BreadthTickers(unittest.TestCase):
    def test_delisted_ticker_replaced_and_dead_skip(self):
        try:
            import market_breadth as mb
        except Exception:
            self.skipTest("streamlit/yfinance not installed")
        self.assertNotIn("TATAMOTORS.NS", mb.NIFTY50_YF)
        self.assertIn("TMPV.NS", mb.NIFTY50_YF)
        mb._YF_DEAD.clear()
        mb._YF_DEAD["LTIM.NS"] = __import__("time").time()
        self.assertNotIn("LTIM.NS", mb._yf_alive(mb.NIFTY50_YF))
        mb._YF_DEAD.clear()


if __name__ == "__main__":
    unittest.main()
