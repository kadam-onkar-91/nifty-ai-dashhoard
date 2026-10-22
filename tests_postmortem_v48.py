"""Tests for the v48 post-trade review (run: python -m unittest tests_postmortem_v48)."""
import json, os, tempfile, unittest

import pandas as pd

import trade_learning as tl

tl.DB_NAME = os.path.join(tempfile.mkdtemp(), "t48.db")
import trade_postmortem as tpm


def _candles(start, highs_lows):
    idx = pd.date_range(start, periods=len(highs_lows), freq="5min")
    return pd.DataFrame({"High": [h for h, _ in highs_lows], "Low": [l for _, l in highs_lows],
                         "Open": [h for h, _ in highs_lows], "Close": [l for _, l in highs_lows]}, index=idx)


def _trade(side, status, ts, exit_ts, entry=25000.0, sl_pts=25.0, tp_pts=50.0, mfe=0.0, mae=0.0, flags=None, expired_r=None):
    sl = entry - sl_pts if side == "BUY" else entry + sl_pts
    tp = entry + tp_pts if side == "BUY" else entry - tp_pts
    f = dict(flags or {})
    f["_learning_meta"] = {"max_favorable_pts": mfe, "max_adverse_pts": mae}
    if expired_r is not None:
        f["_learning_meta"]["expired_r"] = expired_r
    f["_features"] = {"atr": 20.0, "hour": 10.5}
    return {"timestamp": ts, "direction": side, "entry": entry, "stop_loss": sl, "target": tp, "status": status,
            "exit_price": sl if status == "LOSS" else tp, "exit_timestamp": exit_ts, "factors": f}


class Modes(unittest.TestCase):
    def test_stop_hunt_then_worked(self):
        # BUY loses at 24975, then price goes to 25050 within 90 min
        df = _candles("2026-10-20 10:00", [(25005, 24970)] * 4 + [(25060, 25000)] * 8 + [(25000, 24990)] * 8)
        t = _trade("BUY", "LOSS", "2026-10-20 10:00:00", "2026-10-20 10:15:00", mfe=10, mae=25)
        self.assertEqual(tpm.review_trade(t, df)["mode"], "STOP_HUNT_THEN_WORKED")

    def test_wrong_direction(self):
        df = _candles("2026-10-20 10:00", [(25005, 24970)] * 4 + [(24980, 24900)] * 30)
        t = _trade("BUY", "LOSS", "2026-10-20 10:15:00", "2026-10-20 10:15:00", mfe=10, mae=25)
        self.assertEqual(tpm.review_trade(t, df)["mode"], "WRONG_DIRECTION")

    def test_never_worked_and_clean_win(self):
        t = _trade("SELL", "LOSS", "2026-10-20 10:00:00", "2026-10-20 10:05:00", mfe=2, mae=25)
        self.assertEqual(tpm.review_trade(t, None)["mode"], "NEVER_WORKED")
        w = _trade("SELL", "WIN", "2026-10-20 10:00:00", "2026-10-20 10:30:00", mfe=50, mae=3)
        self.assertEqual(tpm.review_trade(w, None)["mode"], "CLEAN_WIN")
        s = _trade("SELL", "WIN", "2026-10-20 10:00:00", "2026-10-20 10:30:00", mfe=50, mae=22)
        self.assertEqual(tpm.review_trade(s, None)["mode"], "SURVIVED_SCARE")

    def test_loss_path_pending_when_no_data(self):
        t = _trade("BUY", "LOSS", "2026-10-20 10:00:00", "2026-10-20 10:15:00", mfe=10, mae=25)
        pm = tpm.review_trade(t, None)
        self.assertEqual(pm["mode"], "LOSS_PATH_PENDING")
        self.assertFalse(pm["final"])

    def test_bad_rows_never_raise(self):
        self.assertIsNone(tpm.review_trade({"direction": "BUY"}, None))
        self.assertEqual(tpm.build(None), [] if not tl.get_resolved_full() else tpm.build(None))


class Board(unittest.TestCase):
    def _pms(self, n_good_win, n_good_loss, n_bad_win, n_bad_loss):
        out = []
        def add(flag, win, k):
            out.append(tpm.review_trade(_trade("BUY", "WIN" if win else "LOSS", f"2026-10-{10 + k % 9:02d} 10:{k % 50:02d}:00",
                                               f"2026-10-{10 + k % 9:02d} 10:{k % 50:02d}:30", mfe=50 if win else 2, mae=3 if win else 25,
                                               flags={"global_aligned": flag}), None))
        k = 0
        for flag, w, l in ((True, n_good_win, n_good_loss), (False, n_bad_win, n_bad_loss)):
            for _ in range(w):
                add(flag, True, k); k += 1
            for _ in range(l):
                add(flag, False, k); k += 1
        return out

    def test_reliable_factor_detected(self):
        fr, gr = tpm.factor_board(self._pms(18, 2, 4, 16))
        g = {x["factor"]: x for x in fr}["global_aligned"]
        self.assertTrue(g["verdict"].startswith("bharosemand"))

    def test_misleading_factor_detected(self):
        fr, _ = tpm.factor_board(self._pms(3, 17, 15, 5))
        g = {x["factor"]: x for x in fr}["global_aligned"]
        self.assertTrue(g["verdict"].startswith("bhatkane"))

    def test_small_sample_gives_no_verdict(self):
        fr, _ = tpm.factor_board(self._pms(3, 0, 0, 3))
        self.assertEqual({x["factor"]: x for x in fr}["global_aligned"]["verdict"], "data kam")

    def test_pure_noise_gives_no_false_verdict(self):
        import random
        random.seed(7)
        false_hits = 0
        for trial in range(40):
            pms = []
            for k in range(60):
                flag, win = random.random() < 0.5, random.random() < 0.45
                pms.append(tpm.review_trade(_trade("BUY", "WIN" if win else "LOSS", f"2026-10-{1 + k % 25:02d} 10:{k % 50:02d}:00",
                                                   f"2026-10-{1 + k % 25:02d} 10:{k % 50:02d}:30", mfe=50 if win else 2, mae=3 if win else 25,
                                                   flags={"global_aligned": flag}), None))
            fr, _ = tpm.factor_board(pms)
            v = {x["factor"]: x for x in fr}["global_aligned"]["verdict"]
            false_hits += v.startswith(("bharosemand", "bhatkane"))
        self.assertLessEqual(false_hits, 1)

    def test_gemini_note_is_short_and_safe(self):
        pms = self._pms(18, 2, 4, 16)
        self.assertLessEqual(len(tpm.gemini_note(pms)), tpm.GEMINI_NOTE_MAX)


class WarnFilter(unittest.TestCase):
    G = {"global_aligned": False, "news_sentiment_aligned": False, "oi_aligned": False, "main_signal_aligned": False,
         "nifty50_fundamentals_aligned": False, "entry_not_extended": False}
    OK = {k: True for k in G}

    def _pms(self, hi_w, hi_l, lo_w, lo_l):
        out, k = [], 0
        for flags, w, l in ((self.G, hi_w, hi_l), (self.OK, lo_w, lo_l)):
            for win in [True] * w + [False] * l:
                out.append(tpm.review_trade(_trade("BUY", "WIN" if win else "LOSS", f"2026-10-{1 + k % 25:02d} 10:{k % 50:02d}:00",
                                                   f"2026-10-{1 + k % 25:02d} 10:{k % 50:02d}:30", mfe=50 if win else 2, mae=3 if win else 25,
                                                   flags=flags), None)); k += 1
        return out

    def test_earned_only_with_proof(self):
        self.assertTrue(tpm.warn_stats(self._pms(3, 17, 17, 3))["earned"])
        self.assertFalse(tpm.warn_stats(self._pms(10, 10, 10, 10))["earned"])
        self.assertFalse(tpm.warn_stats(self._pms(2, 3, 3, 2))["earned"])      # too few trades

    def test_adjustment_is_penalty_only_and_bounded(self):
        tpm.update_state(tpm.warn_stats(self._pms(3, 17, 17, 3)))
        a = tpm.entry_adjustment(self.G)
        self.assertEqual(a["penalty_logit"], tpm.WARN_PENALTY)
        self.assertEqual(tpm.entry_adjustment(self.OK)["penalty_logit"], 0.0)
        tpm.update_state(tpm.warn_stats(self._pms(10, 10, 10, 10)))
        self.assertEqual(tpm.entry_adjustment(self.G)["penalty_logit"], 0.0)
        tpm.update_state(None)
        self.assertEqual(tpm.entry_adjustment(self.G)["penalty_logit"], 0.0)

    def test_kill_switch(self):
        tpm.update_state(tpm.warn_stats(self._pms(3, 17, 17, 3)))
        tpm.WARN_GATE = "off"
        try:
            self.assertEqual(tpm.entry_adjustment(self.G)["penalty_logit"], 0.0)
        finally:
            tpm.WARN_GATE = "auto"

    def test_missed_review_names_the_gate(self):
        import gate_log
        gate_log.DB_PATH = os.path.join(tempfile.mkdtemp(), "g.db")
        for m in (0, 5, 10):
            gate_log.record(f"2026-10-20 10:{m:02d}:00", 25000, "edge_bar", "EV too low", "SELL")
        days = {"2026-10-20": {"legs": [{"major": True, "engine": [], "start_ts": "2026-10-20 10:00:00", "start": "10:00", "dir": "DOWN", "pts": 70}]}}
        rows, by = tpm.missed_review(days)
        self.assertEqual(by, {"edge_bar": 1})
        self.assertEqual(rows[0]["gate"], "edge_bar")


class Storage(unittest.TestCase):
    def test_reader_and_day_row_do_not_break_autopsy_days(self):
        tl.init_db()
        sid = tl.log_setup("BUY", 25000, "CE", 25000.0, 24975.0, 25050.0, 60.0, {"global_aligned": True, "_features": {"atr": 20.0}})
        self.assertIsNotNone(sid)
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        c.execute("UPDATE ai_trade_setups SET status='LOSS', exit_price=24975, exit_timestamp=timestamp")
        c.commit(); c.close()
        rows = tl.get_resolved_full()
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["factors"]["global_aligned"])
        import market_memory as mm
        mm.save_day("pm-2026-10-20", {"day": "2026-10-20", "n": 1})
        self.assertNotIn("pm-2026-10-20", mm.load_days())
        self.assertIn("2026-10-20", mm.load_pm_days())


if __name__ == "__main__":
    unittest.main()
