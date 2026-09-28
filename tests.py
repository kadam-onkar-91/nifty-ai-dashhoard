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


    def test_engine_uses_playbooks_not_hard_veto_chain(self):
        import inspect
        src = inspect.getsource(ai_trade_decision)
        for play in ("BOUNCE_BUY", "REJECTION_SELL", "BREAKDOWN_SELL", "BREAKOUT_BUY"):
            self.assertIn(play, src)
        # the old blockers that froze the engine must not come back
        self.assertNotIn("Full dashboard context is incomplete", src)
        self.assertNotIn("No valid entry location", src)

    def test_no_price_or_atr_never_creates_trade(self):
        out = ai_trade_decision.generate_trade_decision(
            live_price=None, level_prediction=None, atr=None, signal_code=1,
            dashboard_context={"Data Freshness": "Live / fresh"}
        )
        self.assertFalse(out["has_setup"])

    def test_fall_then_stall_produces_a_trade(self):
        trade_learning.init_db()
        import numpy as np, pandas as pd
        rng = np.random.default_rng(7)
        chop = [23260 + 20 * np.sin(i / 5) + rng.normal(0, 4) for i in range(45)]
        x, closes = chop[-1], list(chop)
        for _ in range(30):
            x += -3.6 + rng.normal(0, 5); closes.append(x)
        for dx in (-2, 3, -4, 2, -1, 4, 3, -2, 5, 4, 6):
            x += dx; closes.append(x)
        idx = pd.date_range("2026-09-24 09:15", periods=len(closes), freq="5min")
        c = np.array(closes); o = np.r_[c[0], c[:-1]]
        df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 3, "Low": np.minimum(o, c) + -3,
                           "Close": c, "Volume": 1000.0}, index=idx)
        import indicators
        df["EMA_20"] = df.Close.ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df.Close.ewm(span=50, adjust=False).mean()
        tr = pd.concat([df.High - df.Low, (df.High - df.Close.shift()).abs(), (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(span=14, adjust=False).mean().fillna(10)
        df["RSI"] = indicators.calculate_rsi(df, 14).fillna(50)
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = indicators.calculate_macd(df)
        df["VWAP"] = df.Close.expanding().mean()
        got = set()
        for n in range(48, len(df) + 1):
            d = df.iloc[:n]
            out = ai_trade_decision.generate_trade_decision(
                live_price=float(d.Close.iloc[-1]), level_prediction=None, atr=float(d.ATR.iloc[-1]),
                signal_code=0, is_choppy=True, df=d, sr_context={"status": "OK", "zones": []},
                dashboard_context={"Data Freshness": "Live / fresh"})
            self.assertIn("market_view", out)
            if out["has_setup"]:
                got.add(out["direction"])
                self.assertLessEqual(out["confidence_pct"], trade_learning.CONFIDENCE_CEILING)
                if out["direction"] == "SELL":
                    self.assertLess(out["target"], out["underlying_entry"] < out["stop_loss"] and out["underlying_entry"])
        self.assertTrue(got, "engine took no trade at all on a clear fall/bounce")


class SRTests(unittest.TestCase):
    def test_nearest_level_direction(self):
        self.assertEqual(support_resistance._nearest_level([90, 95, 110], 100, "above"), 110)
        self.assertEqual(support_resistance._nearest_level([90, 95, 110], 100, "below"), 95)


class RealSRTests(unittest.TestCase):
    def _frames(self):
        import numpy as np, pandas as pd
        rng = np.random.default_rng(3)
        days = pd.bdate_range(end="2026-09-25", periods=120)
        c = 23000 + np.cumsum(rng.normal(0, 60, len(days)))
        daily = pd.DataFrame({"Open": c, "High": c + 90, "Low": c - 90, "Close": c}, index=days)
        idx = []
        for d in days[-5:]:
            idx += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min"))
        p = c[-6] + np.cumsum(rng.normal(0, 4, len(idx)))
        df = pd.DataFrame({"Open": p, "High": p + 4, "Low": p - 4, "Close": p, "Volume": 1e5},
                          index=pd.DatetimeIndex(idx))
        return df, daily

    def test_zones_are_spaced_and_contain_previous_day_levels(self):
        import real_sr
        df, daily = self._frames()
        live = float(df.Close.iloc[-1])
        out = real_sr.build_real_sr_zones(df, live, 11.0, df_daily=daily)
        self.assertTrue(out["supports"] or out["resistances"])
        for side in ("supports", "resistances"):
            zs = sorted(out[side], key=lambda z: z["price"])
            for a, b in zip(zs, zs[1:]):
                self.assertGreaterEqual(b["price"] - a["price"], 20.0 - 1e-6)
        last = daily.iloc[-2]   # last COMPLETED session (the final daily row is "today", still forming)
        allp = [z["price"] for z in out["zones"]]
        self.assertTrue(any(abs(p - float(last["High"])) < 6 for p in allp))
        self.assertTrue(any(abs(p - float(last["Low"])) < 6 for p in allp))

    def test_simulated_option_chain_is_never_a_level(self):
        import real_sr, pandas as pd
        df, daily = self._frames()
        live = float(df.Close.iloc[-1])
        oc = pd.DataFrame({"Strike": [round(live / 50) * 50 + 100], "Call OI": [9e6], "Put OI": [1.0]})
        out = real_sr.build_real_sr_zones(df, live, 11.0, df_daily=daily, df_option_chain=oc, oi_trusted=False)
        self.assertFalse(any("Option Chain OI wall" in s for z in out["zones"] for s in z["sources"]))


class SegmentLearningTests(unittest.TestCase):
    def _patch_rows(self, rows):
        import json
        self._orig = trade_learning._resolved_learning_rows
        trade_learning._resolved_learning_rows = lambda: [(json.dumps({"_entry_meta": m}), st) for m, st in rows]
        trade_learning._SEG_CACHE["stats"] = None

    def tearDown(self):
        if hasattr(self, "_orig"):
            trade_learning._resolved_learning_rows = self._orig
        trade_learning._SEG_CACHE["stats"] = None

    def test_losing_segment_is_paused_after_enough_samples(self):
        meta = {"playbook": "BOUNCE_BUY", "grade": "C", "hour_bucket": "1030-1230", "side": "BUY", "probe": False}
        rows = [(meta, "LOSS")] * 7 + [(meta, "WIN")] * 1      # 8 trades, 12.5% wins
        self._patch_rows(rows)
        blocked, why = trade_learning.segment_gate(meta)
        self.assertTrue(blocked, why)

    def test_unknown_segment_is_allowed_so_engine_can_learn(self):
        self._patch_rows([])
        blocked, _ = trade_learning.segment_gate({"playbook": "BOUNCE_BUY", "grade": "A", "hour_bucket": "1030-1230",
                                                  "side": "BUY", "probe": True})
        self.assertFalse(blocked)

    def test_winning_segment_is_never_blocked(self):
        meta = {"playbook": "REJECTION_SELL", "grade": "A", "hour_bucket": "1030-1230", "side": "SELL", "probe": False}
        self._patch_rows([(meta, "WIN")] * 6 + [(meta, "LOSS")] * 4)
        self.assertFalse(trade_learning.segment_gate(meta)[0])

    def test_probes_pause_themselves_when_they_lose(self):
        meta = {"playbook": "BOUNCE_BUY", "grade": "A", "hour_bucket": "1030-1230", "side": "BUY", "probe": True}
        self._patch_rows([(meta, "LOSS")] * 5 + [(meta, "WIN")])
        blocked, why = trade_learning.segment_gate(meta)
        self.assertTrue(blocked, why)

    def test_entry_window_blocks_open_and_close(self):
        from datetime import datetime
        import ai_trade_decision as ad
        real = ad._ist_now
        try:
            for hh, mm, expect_outside in ((9, 20, True), (9, 30, False), (12, 0, False), (15, 10, True), (15, 25, True)):
                ad._ist_now = lambda hh=hh, mm=mm: datetime(2026, 9, 29, hh, mm, tzinfo=ad._IST)
                self.assertEqual(ad._outside_entry_window(), expect_outside, (hh, mm))
        finally:
            ad._ist_now = real


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


class CommodityEngineTests(unittest.TestCase):
    def test_profiles(self):
        import commodity_engine
        self.assertTrue({"GOLD", "SILVER"}.issubset(set(commodity_engine.COMMODITIES)))
        for asset in ("GOLD", "SILVER"):
            f = commodity_engine._commodity_fundamentals(asset)
            self.assertEqual(f["status"], "STRUCTURAL_AVAILABLE")
            self.assertTrue(f["drivers"])

    def test_invalid_asset(self):
        import commodity_engine
        out = commodity_engine.analyze_commodity("PLATINUM")
        self.assertEqual(out["status"], "INVALID")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Multi-stock smoke tests (static/no network dependency)
try:
    import multi_stock_engine as _mse
    assert len(_mse.NSE_STOCKS) >= 50
    assert len(set(_mse.NSE_STOCKS)) == len(_mse.NSE_STOCKS)
    assert 'RELIANCE' in _mse.NSE_STOCKS and 'YESBANK' in _mse.NSE_STOCKS
    print('Multi-stock universe smoke tests: PASS')
except Exception as _e:
    raise AssertionError(f'Multi-stock smoke tests failed: {_e}')

