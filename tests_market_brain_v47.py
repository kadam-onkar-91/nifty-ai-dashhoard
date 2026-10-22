"""Tests for the v47 adaptive layer (run: python -m unittest tests_market_brain_v47)."""
import datetime, os, sys, tempfile, time, unittest

import numpy as np
import pandas as pd

import trade_learning as tl

tl.DB_NAME = os.path.join(tempfile.mkdtemp(), "t47.db")
import market_adaptive as mad
import market_memory as mm


def _snap(ts, strats=None, f25=None, src="live"):
    y = {"f25": f25, "f50": "NONE", "up": 0, "dn": 0, "ret": 0, "bars": 12} if f25 else None
    return {"ts": ts, "f": {"strats": strats} if strats else {}, "y": y, "src": src, "fc": None}


class StrategyScorecard(unittest.TestCase):
    def _snaps(self, wins, losses, base_rows=80):
        s = []
        t0 = pd.Timestamp("2026-10-01 09:20")
        # background: balanced market so base rate is 50%
        for i in range(base_rows):
            s.append(_snap(str(t0 + pd.Timedelta(minutes=5 * i)), None, "UP" if i % 2 else "DN", src="backfill"))
        t = pd.Timestamp("2026-10-05 09:20")
        for i in range(wins + losses):
            s.append(_snap(str(t), ["orb:BUY"], "UP" if i < wins else "DN"))
            t += pd.Timedelta(minutes=60)                      # far apart -> separate episodes
        return s

    def test_small_sample_ignored(self):
        self.assertEqual(mad._compute_strategy_stats(self._snaps(5, 2)), {})

    def test_shrinkage_keeps_effect_small(self):
        st = mad._compute_strategy_stats(self._snaps(11, 1))["orb"]
        self.assertEqual(st["samples"], 12)
        self.assertGreater(st["win_rate"], 50)
        self.assertLess((st["win_rate"] - 50) * 0.16, 3.0)     # 11/12 raw (92%) moves the strategy score by only ~2 points

    def test_consecutive_candles_are_one_episode(self):
        s = []
        t = pd.Timestamp("2026-10-05 09:20")
        for i in range(12):                                    # the same signal on 12 consecutive candles = ONE episode
            s.append(_snap(str(t), ["orb:BUY"], "UP"))
            t += pd.Timedelta(minutes=5)
        st = mad._compute_strategy_stats(s + self._snaps(0, 0))
        self.assertEqual(st, {})                               # 1 episode < minimum

    def test_losers_score_below_50(self):
        st = mad._compute_strategy_stats(self._snaps(2, 14))["orb"]
        self.assertLess(st["win_rate"], 45)


class EngineAdjustment(unittest.TestCase):
    def setUp(self):
        mad.ENGINE_ADJUST = "auto"
        mad.update_state({}, [], {}, [], {})

    def test_inert_by_default(self):
        self.assertEqual(mad.engine_adjustment("BUY")["delta_logit"], 0.0)

    def test_untrusted_forecaster_never_nudges(self):
        mad.update_state({"ready": True, "trusted": False, "p_up25": 0.9}, [], {}, [], {"n_trusted": 99, "acc_trusted": 0.9})
        self.assertEqual(mad.engine_adjustment("BUY")["delta_logit"], 0.0)

    def test_earned_forecaster_nudges_within_cap(self):
        mad.update_state({"ready": True, "trusted": True, "p_up25": 0.95}, [], {}, [], {"n_trusted": 40, "acc_trusted": 0.6})
        a = mad.engine_adjustment("BUY")
        self.assertTrue(a["active"])
        self.assertLessEqual(a["delta_logit"], mad.TOTAL_CAP)
        self.assertGreater(a["delta_logit"], 0)
        self.assertLess(mad.engine_adjustment("SELL")["delta_logit"], 0)

    def test_pattern_needs_enough_samples_and_significance(self):
        base = {"up": 0.4, "dn": 0.57}
        weak = {"key": "p", "text": "x", "n": 20, "up": 0.2, "dn": 0.8, "none": 0}
        mad.update_state({}, [weak], base, ["p"], {})
        self.assertEqual(mad.engine_adjustment("BUY")["delta_logit"], 0.0)          # n too small
        strong = {"key": "p", "text": "x", "n": 600, "up": 0.15, "dn": 0.83, "none": 0.02}
        mad.update_state({}, [strong], base, ["p"], {})
        self.assertLess(mad.engine_adjustment("BUY")["delta_logit"], 0)              # solid evidence AGAINST buying
        noise = {"key": "p", "text": "x", "n": 300, "up": 0.41, "dn": 0.56, "none": 0.03}
        mad.update_state({}, [noise], base, ["p"], {})
        self.assertEqual(mad.engine_adjustment("BUY")["delta_logit"], 0.0)

    def test_kill_switch_and_stale_state(self):
        mad.update_state({"ready": True, "trusted": True, "p_up25": 0.95}, [], {}, [], {"n_trusted": 40, "acc_trusted": 0.6})
        mad.ENGINE_ADJUST = "off"
        self.assertEqual(mad.engine_adjustment("BUY")["delta_logit"], 0.0)
        mad.ENGINE_ADJUST = "auto"
        mad._STATE["ts"] = time.time() - 5000
        self.assertEqual(mad.engine_adjustment("BUY")["delta_logit"], 0.0)


class Storage(unittest.TestCase):
    def test_daily_metrics_do_not_pollute_autopsy_days(self):
        tl.DB_NAME = os.path.join(tempfile.mkdtemp(), "s.db")
        mm._cache.clear()
        mm.save_day("2026-10-20", {"day": "2026-10-20", "open": 1})
        mm.save_day("stats-2026-10-20", {"day": "2026-10-20", "lab_n": 5})
        self.assertEqual(list(mm.load_days()), ["2026-10-20"])
        self.assertEqual(list(mm.load_stats()), ["2026-10-20"])


class GeminiPrompt(unittest.TestCase):
    def test_learned_block_reaches_prompt_and_is_capped(self):
        import ai_trade_decision as atd
        seen = {}

        class R:
            status_code = 200
            def json(self): return {"candidates": [{"content": {"parts": [{"text": '{"own_direction":"BUY","approved":true,"verdict":"APPROVE","confidence":60,"reason":"x","risk_flags":[]}'}]}}]}
            text = ""
        def fake_post(url, headers=None, json=None, timeout=None, **k):
            seen["body"] = str(json)
            return R()
        real = atd.requests.post
        atd.requests.post = fake_post
        try:
            mad.publish_gemini_text("- Engine's last 5 resolved trades: 1 wins / 4 losses." + "x" * 5000)
            self.assertLessEqual(len(atd.LEARNED_CONTEXT_TEXT), 5100)
            try:
                atd.gemini_trade_review(["k"], {}, {"direction": "BUY"}, "BUY", live_price=22600, use_search=False)
            except Exception:
                pass
        finally:
            atd.requests.post = real
            mad.publish_gemini_text("")
        self.assertIn("body", seen, "the review prompt was never sent")
        if "body" in seen:
            self.assertIn("LEARNED MEMORY", seen["body"])
            self.assertIn("NEVER overrides the live packet", seen["body"])


class StrategyEngineHook(unittest.TestCase):
    def test_analyze_uses_market_adaptive_stats(self):
        import strategy_engine as se
        rng = np.random.default_rng(3)
        idx = pd.date_range("2026-10-20 09:15", periods=120, freq="5min")
        c = 22600 + np.cumsum(rng.normal(0, 6, 120))
        df = pd.DataFrame({"Open": c - 1, "High": c + 4, "Low": c - 4, "Close": c, "Volume": 1000.0}, index=idx)
        df.attrs["market_source"] = "UPSTOX_LIVE"
        keys = list(se.STRATEGY_META)
        calls = []
        orig = mad.strategy_stats
        mad.strategy_stats = lambda snaps=None: (calls.append(1) or {keys[0]: {"samples": 40, "win_rate": 75.0}})
        try:
            r = se.analyze(df)
        finally:
            mad.strategy_stats = orig
        self.assertEqual(r["status"], "READY")
        self.assertTrue(calls, "market_adaptive.strategy_stats should be consulted when the old hook is empty")


if __name__ == "__main__":
    unittest.main()
