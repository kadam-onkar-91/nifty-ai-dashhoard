"""v41 tests: persistent gate log + the 'why no strategy' explanation.  python -m unittest tests_v41_visibility"""
import os, sys, tempfile, types, unittest
from datetime import datetime, timedelta

os.environ["GATE_LOG_DB"] = os.path.join(tempfile.mkdtemp(), "gate_test.db")
try:
    import streamlit  # noqa: F401
except Exception:
    sys.modules["streamlit"] = types.ModuleType("streamlit")

import gate_log
import ai_trade_decision as atd


class GateLogTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(gate_log.DB_PATH):
            os.remove(gate_log.DB_PATH)

    def test_one_row_per_candle_and_gate(self):
        gate_log.record("2026-10-09 09:20:00", 22300, "No strategy formed", "r")
        gate_log.record("2026-10-09 09:20:00", 22301, "No strategy formed", "r")      # same candle+gate -> ignored
        gate_log.record("2026-10-09 09:20:00", 22301, "Win-chance floor", "r")        # other gate same candle -> kept
        self.assertEqual(len(gate_log.rows_for_day("2026-10-09")), 2)

    def test_coverage_reports_late_start_and_hole(self):
        base = datetime(2026, 10, 9, 12, 0)
        for i in range(0, 4):
            gate_log.record(base + timedelta(minutes=5 * i), 22400, "No strategy formed")
        for i in range(10, 13):                       # 30 min hole in the middle
            gate_log.record(base + timedelta(minutes=5 * i), 22400, "No strategy formed")
        cov = gate_log.coverage("2026-10-09")
        self.assertEqual(cov["first"], base)
        self.assertGreaterEqual(cov["missed_start_min"], 15)          # nothing between 09:15 and 12:00
        self.assertEqual(len(cov["gaps"]), 1)
        note = gate_log.coverage_note("2026-10-09")
        self.assertIn("12:00", note)
        self.assertIn("chal hi nahi", note)

    def test_full_coverage_has_no_note(self):
        base = datetime(2026, 10, 9, 9, 15)
        for i in range(0, 20):
            gate_log.record(base + timedelta(minutes=5 * i), 22400, "No strategy formed")
        self.assertEqual(gate_log.coverage_note("2026-10-09"), "")

    def test_never_raises_on_bad_input(self):
        gate_log.record(None, None, None)
        self.assertEqual(gate_log.rows_for_day("1999-01-01"), [])


class NoStrategyDetailTests(unittest.TestCase):
    def test_non_upstox_source_is_named(self):
        d = atd._no_strategy_detail({"market_data_source": "YAHOO_FALLBACK_NON_TRADABLE", "live_upstox": False})
        self.assertIn("YAHOO_FALLBACK_NON_TRADABLE", d)
        self.assertIn("Upstox", d)

    def test_live_block_is_explained(self):
        sr = {"market_data_source": "UPSTOX_LIVE", "live_upstox": True, "buy_live_count": 0, "sell_live_count": 0,
              "buy_score": 0, "sell_score": 0,
              "ranked_active": [{"live_valid": False, "live_note": "BUY blocked: price already dropped 1.4 ATR off the high - wait"}]}
        d = atd._no_strategy_detail(sr)
        self.assertIn("live price action ne roka", d)
        self.assertIn("BUY blocked", d)

    def test_message_still_classified_as_no_strategy(self):
        msg = "No named strategy is currently formed strongly enough. AI research continues." + atd._no_strategy_detail({})
        self.assertEqual(atd.classify_block(msg), "No strategy formed")

    def test_detail_never_raises(self):
        self.assertIsInstance(atd._no_strategy_detail(None), str)
        self.assertIsInstance(atd._no_strategy_detail({"ranked_active": "garbage"}), str)


if __name__ == "__main__":
    unittest.main()
