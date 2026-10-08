import unittest
from datetime import datetime
import numpy as np
import pandas as pd

import entry_quality, oi_buildup, event_calendar, trade_diagnostics


def _day_df(prev_close=22600.0, start=22570.0, end=22213.0, n=60):
    """prev day's last candle + today's steady fall to `end`, ending at the day low."""
    idx = pd.date_range("2026-10-07 15:25", periods=1, freq="5min").append(pd.date_range("2026-10-08 09:15", periods=n, freq="5min"))
    closes = np.concatenate([[prev_close], np.linspace(start, end, n)])
    df = pd.DataFrame({"Open": np.r_[prev_close, closes[:-1][1:] if False else closes[:-1][1:]] if False else np.r_[prev_close, closes[:-1]][: len(closes)],
                       "Close": closes}, index=idx)
    df["Open"] = np.r_[prev_close, closes[:-1]]
    df["High"] = df[["Open", "Close"]].max(axis=1) + 2
    df["Low"] = df[["Open", "Close"]].min(axis=1) - 2
    df["RSI"] = 38.0; df["EMA_20"] = df["Close"] + 15; df["BB_Upper"] = df["Close"] + 60; df["BB_Lower"] = df["Close"] - 60; df["ATR"] = 17.0
    return df


class EntryQualityDayMove(unittest.TestCase):
    def test_sell_at_day_low_after_big_fall_is_severe(self):
        eq = entry_quality.assess_entry(_day_df(), "SELL", 17.0)
        self.assertGreaterEqual(eq["metrics"]["day_move_pct"], 1.4)
        self.assertLessEqual(eq["metrics"]["day_pos"], 0.12)
        self.assertTrue(eq["severe"])
        self.assertTrue(any("chasing the end" in f for f in eq["chase_flags"]))

    def test_buy_after_big_fall_is_not_flagged_for_day_move(self):
        eq = entry_quality.assess_entry(_day_df(), "BUY", 17.0)
        self.assertFalse(any("day's" in f for f in eq["chase_flags"]))

    def test_small_day_move_not_flagged(self):
        eq = entry_quality.assess_entry(_day_df(start=22590, end=22520), "SELL", 17.0)
        self.assertFalse(any("day's" in f for f in eq["chase_flags"]))

    def test_volatility_shock_flag(self):
        df = _day_df(start=22600, end=22580)
        df.iloc[-2, df.columns.get_loc("High")] = df["Close"].iloc[-2] + 80
        eq = entry_quality.assess_entry(df, "BUY", 17.0)
        self.assertTrue(any("volatility shock" in f for f in eq["chase_flags"]))


class OiBuildup(unittest.TestCase):
    def _chain(self, d_call, d_put):
        rows = []
        for k in range(22000, 22500, 50):
            rows.append({"Strike": k, "Call OI": 1000 + d_call, "Put OI": 1000 + d_put, "Call Prev OI": 1000, "Put Prev OI": 1000})
        return pd.DataFrame(rows)

    def test_fall_absorbed_is_against_sell(self):
        r = oi_buildup.assess(self._chain(-200, 300), 22213, "SELL", price_change_pct=-1.5)
        self.assertEqual(r["bias"], "BUY"); self.assertTrue(r["against"]); self.assertGreater(r["penalty_logit"], 0)
        self.assertIn("ABSORBED", r["label"])

    def test_short_buildup_confirms_sell(self):
        r = oi_buildup.assess(self._chain(300, -100), 22213, "SELL", price_change_pct=-1.5)
        self.assertTrue(r["confirms"]); self.assertLess(r["penalty_logit"], 0); self.assertIn("SHORT BUILDUP", r["label"])

    def test_missing_prev_oi_is_neutral(self):
        df = self._chain(0, 0).drop(columns=["Call Prev OI", "Put Prev OI"])
        r = oi_buildup.assess(df, 22213, "SELL", -1.0)
        self.assertFalse(r["available"]); self.assertEqual(r["penalty_logit"], 0.0)


class EventCalendar(unittest.TestCase):
    def test_rbi_day_blocks_at_announcement(self):
        self.assertEqual(event_calendar.event_risk(datetime(2026, 12, 4, 10, 0))["level"], "BLOCK")

    def test_rbi_day_afternoon_caution(self):
        r = event_calendar.event_risk(datetime(2026, 12, 4, 13, 0))
        self.assertEqual(r["level"], "CAUTION"); self.assertGreater(r["penalty_logit"], 0)

    def test_fed_reaction_morning_caution(self):
        self.assertEqual(event_calendar.event_risk(datetime(2026, 10, 29, 9, 30))["level"], "CAUTION")

    def test_normal_day_none_and_expiry_flags(self):
        r = event_calendar.event_risk(datetime(2026, 10, 13, 11, 0))      # a Tuesday: weekly expiry
        self.assertEqual(r["level"], "NONE"); self.assertTrue(r["expiry_day"]); self.assertFalse(r["monthly_expiry"])
        self.assertTrue(event_calendar.event_risk(datetime(2026, 10, 27, 11, 0))["monthly_expiry"])
        self.assertFalse(event_calendar.event_risk(datetime(2026, 10, 14, 11, 0))["expiry_day"])


class Diagnostics(unittest.TestCase):
    def _rows(self, n_loss, n_win=0):
        mk = lambda y: {"flags": {"entry_not_extended": False, "entry_confirmed_bounce": False}, "label": y, "weight": 1.0,
                        "feats": {"hour": 14.4, "day_move_pct": 1.7, "day_pos": 0.05}, "source": "real"}
        return [mk(0) for _ in range(n_loss)] + [mk(1) for _ in range(n_win)]

    def test_tags(self):
        t = trade_diagnostics.risk_tags({"entry_not_extended": False, "entry_confirmed_bounce": False},
                                        {"hour": 14.4, "day_move_pct": 1.7, "day_pos": 0.05, "oi_flow": -1.0})
        for x in ("extended_entry", "no_confirmation", "day_move_exhausted", "at_day_extreme", "late_session", "oi_flow_against"):
            self.assertIn(x, t)

    def test_no_penalty_with_too_few_trades(self):
        pen, _ = trade_diagnostics.tag_penalty(["late_session"], self._rows(3))
        self.assertEqual(pen, 0.0)

    def test_penalty_after_repeated_losses_and_cap(self):
        pen, note = trade_diagnostics.tag_penalty(["late_session", "extended_entry", "day_move_exhausted", "at_day_extreme"], self._rows(10, 1))
        self.assertGreater(pen, 0); self.assertLessEqual(pen, trade_diagnostics.MAX_TAG_PENALTY); self.assertIn("repeat-mistake", note)

    def test_good_tag_not_penalised(self):
        pen, _ = trade_diagnostics.tag_penalty(["late_session"], self._rows(2, 10))
        self.assertEqual(pen, 0.0)

    def test_loss_summary_counts_real_losses(self):
        s = trade_diagnostics.loss_reason_summary(self._rows(5, 1))
        self.assertTrue(s and s[0]["losses"] == 5)


class EngineIntegration(unittest.TestCase):
    def test_event_block_stops_new_entries(self):
        import ai_trade_decision as atd
        d = atd.generate_trade_decision(22213.0, {}, 17.0, signal_code=-1, now_ist=datetime(2026, 12, 4, 10, 0), dashboard_context={},
                                      strategy_result={"has_setup": True, "score": 95, "direction": "SELL", "selected_strategies": ["x"]})
        self.assertFalse(d["has_setup"])
        self.assertTrue("Event risk window" in d["reason"] or "STALE" in d["reason"], d["reason"])
        self.assertEqual(atd.classify_block("Event risk window: x"), "Event risk window")


if __name__ == "__main__":
    unittest.main()


class WinFloorValve(unittest.TestCase):
    def _floor(self, last_ts, now):
        import ai_trade_decision as atd, trade_learning as tl
        orig = tl.get_recent_setups
        tl.get_recent_setups = lambda limit=1: ([{"timestamp": last_ts}] if last_ts else [])
        try:
            return atd._win_floor(now)
        finally:
            tl.get_recent_setups = orig

    def test_normal_floor_after_recent_trade(self):
        self.assertEqual(self._floor("2026-10-08 14:20:16", datetime(2026, 10, 9, 10, 0))[0], 0.35)

    def test_floor_eases_after_two_trading_days_without_setup(self):
        f, note = self._floor("2026-10-06 10:00:00", datetime(2026, 10, 8, 11, 0))
        self.assertEqual(f, 0.31); self.assertIn("eased", note)

    def test_weekend_does_not_count_as_drought(self):
        self.assertEqual(self._floor("2026-10-09 14:00:00", datetime(2026, 10, 12, 10, 0))[0], 0.35)   # Fri -> Mon
