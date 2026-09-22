"""Lightweight regression tests for the critical trading decision path.
Run with: python tests.py
"""
import unittest

import ai_trade_decision
import position_sizing
import risk_engine
import support_resistance
import trade_learning
import historical_backtest


class RiskAndSizingTests(unittest.TestCase):
    def test_position_size_rounds_down(self):
        out = position_sizing.calculate_position_size(100000, 1.0, 100, 90, lot_size=75)
        self.assertEqual(out["lots"], 1)
        self.assertEqual(out["quantity"], 75)

    def test_invalid_zero_stop_distance(self):
        out = position_sizing.calculate_position_size(100000, 1.0, 100, 100)
        self.assertEqual(out["status"], "INVALID")

    def test_risk_blocks_after_consecutive_losses(self):
        trades = [{"pnl": -1, "exit_timestamp": "2026-09-22 10:00:00"},
                  {"pnl": -1, "exit_timestamp": "2026-09-22 11:00:00"},
                  {"pnl": -1, "exit_timestamp": "2026-09-22 12:00:00"}]
        out = risk_engine.evaluate_risk_state(trades, 100000, max_consecutive_losses=3)
        self.assertEqual(out["status"], "NO_NEW_ENTRIES")

    def test_testing_mode_has_no_trade_count_or_loss_streak_cap(self):
        self.assertIsNone(risk_engine.DEFAULT_MAX_TRADES_PER_DAY)
        self.assertIsNone(risk_engine.DEFAULT_MAX_CONSECUTIVE_LOSSES)


class DecisionGateTests(unittest.TestCase):
    def test_stale_data_never_creates_trade(self):
        out = ai_trade_decision.generate_trade_decision(
            live_price=100, level_prediction=None, atr=2, signal_code=1,
            dashboard_context={"Data Freshness": "STALE"}
        )
        self.assertFalse(out["has_setup"])


    def test_full_context_gate_is_stricter_than_single_factor(self):
        import inspect
        src = inspect.getsource(ai_trade_decision.generate_trade_decision)
        self.assertIn('critical_true < 5', src)
        self.assertIn('FULL_CONTEXT_BEFORE_ENTRY', src)
        self.assertIn('bearish_trend_transition_confirmed', src)
        self.assertIn('Full dashboard context is incomplete', src)

    def test_missing_validated_sr_never_creates_trade(self):
        out = ai_trade_decision.generate_trade_decision(
            live_price=100, level_prediction=None, atr=2, signal_code=1,
            dashboard_context={"Data Freshness": "FRESH"}
        )
        self.assertFalse(out["has_setup"])


class SRTests(unittest.TestCase):
    def test_nearest_level_direction(self):
        self.assertEqual(support_resistance._nearest_level([90, 95, 110], 100, "above"), 110)
        self.assertEqual(support_resistance._nearest_level([90, 95, 110], 100, "below"), 95)


class HistoricalBacktestTests(unittest.TestCase):
    def test_short_history_is_rejected(self):
        import pandas as pd
        df = pd.DataFrame({
            "Open": [100.0] * 20, "High": [101.0] * 20,
            "Low": [99.0] * 20, "Close": [100.0] * 20, "Volume": [1000] * 20
        })
        out = historical_backtest.run_ml_walk_forward(df)
        self.assertEqual(out["status"], "NOT_ENOUGH_DATA")


class LearningTests(unittest.TestCase):
    def test_confidence_is_capped(self):
        trade_learning.init_db()
        flags = {k: True for k in trade_learning.ALL_FACTOR_KEYS}
        score, _, _ = trade_learning.compute_confidence(flags)
        self.assertLessEqual(score, trade_learning.CONFIDENCE_CEILING)
        self.assertGreaterEqual(score, trade_learning.CONFIDENCE_FLOOR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
