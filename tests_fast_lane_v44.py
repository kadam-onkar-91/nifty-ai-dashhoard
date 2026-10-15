"""v44 fast-lane tests (synthetic 5-min candles).  Run: python -m unittest tests_fast_lane_v44"""
import unittest
import numpy as np
import pandas as pd
import fast_lane


def _mk(closes, spread=4.0, start="2026-10-14 09:15"):
    idx = pd.date_range(start, periods=len(closes), freq="5min")
    rows, prev = [], closes[0]
    for c in closes:
        o = prev
        h = max(o, c) + spread * 0.6
        l = min(o, c) - spread * 0.6
        rows.append((o, h, l, c, 1000))
        prev = c
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)


def _range(n=40, base=22700, amp=6):
    rng = np.random.default_rng(1)
    return list(base + rng.uniform(-amp, amp, n))


class FastLane(unittest.TestCase):
    def test_waterfall_sell_is_caught_early(self):
        c = _range() + [22688, 22672]       # two strong down candles after a range
        r = fast_lane.analyze(_mk(c))
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "SELL")
        self.assertLess(r["stop_loss"] - r["underlying_entry"], 31)
        self.assertGreater(r["underlying_entry"] - r["target"], 20)

    def test_does_not_chase_after_move_is_over(self):
        c = _range() + [22690, 22672, 22650, 22625, 22600, 22575, 22560, 22548, 22540, 22534]
        r = fast_lane.analyze(_mk(c))
        self.assertTrue(r is None or r["playbook"] != "FL_MOMENTUM_BREAK" or r["direction"] != "SELL")

    def test_range_market_no_trade(self):
        self.assertIsNone(fast_lane.analyze(_mk(_range(60))))

    def test_bounce_after_stretch(self):
        c = _range(30, 22700) + [22690, 22670, 22650, 22628, 22612, 22600]
        df = _mk(c)
        last = df.index[-1]
        # final candle: long lower wick rejection, closes back up
        df.loc[last, ["Open", "High", "Low", "Close"]] = [22600, 22622, 22578, 22618]
        r = fast_lane.analyze(df)
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "BUY")

    def test_flag_break_catches_missed_first_candle(self):
        # impulse down (missed), tight 4-candle pause, then breaks down again
        c = _range() + [22685, 22660, 22640, 22636, 22639, 22635, 22638, 22615]
        r = fast_lane.analyze(_mk(c, spread=3.0))
        self.assertIsNotNone(r)
        self.assertEqual(r["direction"], "SELL")

    def test_level_break_pdh(self):
        c = _range(40, 22700) + [22706, 22722]
        r = fast_lane.analyze(_mk(c), prev_day={"pdh": 22712, "pdl": 22600})
        self.assertTrue(r is None or r["direction"] == "BUY")

    def test_short_history_none(self):
        self.assertIsNone(fast_lane.analyze(_mk([22700] * 10)))


class FastLaneResearch(unittest.TestCase):
    """v44c: the fast lane may not trade against the rest of the dashboard's research."""
    def _run(self, **kw):
        import datetime as dt
        import ai_trade_decision as A
        df = _mk(_range() + [22688, 22672])
        return A.generate_trade_decision(22672, None, 12.0, strategy_result={}, dashboard_context={"Data Freshness": "LIVE"},
                                         price_df=df, now_ist=dt.datetime(2026, 10, 14, 10, 40), **kw)

    def test_research_against_blocks(self):
        d = self._run(breadth_advances=40, breadth_declines=10, global_avg_change=0.8, sniper_bias="BULLISH",
                      global_research={"status": "AVAILABLE", "directional_bias": "BUY", "strength": "STRONG"})
        self.assertFalse(d.get("fast_lane"))

    def test_moderate_opposition_is_advice_not_veto(self):
        d = self._run(breadth_advances=40, breadth_declines=10, global_avg_change=0.8, sniper_bias="BULLISH")
        self.assertTrue(d.get("fast_lane"))
        self.assertTrue(any("Research against" in f for f in d.get("soft_flags", [])))

    def test_research_with_trade_takes_it(self):
        d = self._run(breadth_advances=10, breadth_declines=40, global_avg_change=-0.8, sniper_bias="BEARISH")
        self.assertTrue(d.get("has_setup") and d.get("fast_lane") and d["direction"] == "SELL")
        self.assertIn("research net", d["fast_lane_why"])


if __name__ == "__main__":
    unittest.main()


class FastLaneV44d(unittest.TestCase):
    def test_carry_keeps_candidate_alive_while_gemini_reviews(self):
        import time, datetime as dt
        import ai_trade_decision as A
        df = _mk(_range() + [22688, 22672])
        kw = dict(strategy_result={}, dashboard_context={"Data Freshness": "LIVE"}, now_ist=dt.datetime(2026, 10, 14, 10, 40))
        d1 = A.generate_trade_decision(22672, None, 12.0, price_df=df, **kw)
        self.assertTrue(d1.get("fast_lane"))
        # next refresh: move kept running -> classic fast-lane detection would call it a chase; the carry keeps it
        df2 = _mk(_range() + [22688, 22672, 22669, 22671])
        self.assertIsNone(fast_lane.analyze(df2))      # fresh detection no longer sees it ...
        d2 = A.generate_trade_decision(22669, None, 12.0, price_df=df2, fl_carry={"cand": d1["fast_lane_candidate"], "t": time.time()}, **kw)
        self.assertTrue(d2.get("fast_lane"), d2.get("reason"))
        self.assertAlmostEqual(d2["underlying_entry"], 22669, places=0)
        # expired carry is ignored
        d3 = A.generate_trade_decision(22669, None, 12.0, price_df=df2, fl_carry={"cand": d1["fast_lane_candidate"], "t": time.time() - 3600}, **kw)
        self.assertFalse(d3.get("fast_lane"))

    def test_fast_lane_local_fallback(self):
        import ai_trade_decision as A
        ok = A.local_fallback_review({"fast_lane": True, "fast_lane_score": 80, "risk_reward": 1.8, "direction": "SELL"})
        no = A.local_fallback_review({"fast_lane": True, "fast_lane_score": 60, "risk_reward": 1.8, "direction": "SELL"})
        self.assertTrue(ok["approved"]); self.assertFalse(no["approved"])

    def test_gemini_gets_fast_lane_report(self):
        import ai_trade_decision as A
        seen = {}

        class R:
            status_code = 200
            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": '{"own_direction":"SELL","approved":true,"verdict":"APPROVE","confidence":70,"reason":"ok","risk_flags":[]}'}]}}]}

        def fake_post(url, headers=None, json=None, timeout=0):
            seen["prompt"] = json["contents"][0]["parts"][0]["text"]
            return R()

        class FakePool:
            REVIEW_MODELS = ["m"]
            PoolHTTPError = A.gemini_pool.PoolHTTPError
            GeminiUnavailable = A.gemini_pool.GeminiUnavailable
            @staticmethod
            def run(keys, models, fn, max_attempts=1, deadline_s=1):
                return fn(keys[0], models[0]), 0, models[0]

        old_post, old_pool = A.requests.post, A.gemini_pool
        A.requests.post, A.gemini_pool = fake_post, FakePool
        try:
            dec = {"fast_lane": True, "fast_lane_playbook": "FL_MOMENTUM_BREAK", "fast_lane_score": 80, "fast_lane_why": "x | research net +3",
                   "direction": "SELL", "underlying_entry": 22672, "stop_loss": 22688, "target": 22643, "risk_reward": 1.8}
            out = A.gemini_trade_review(["k"], {"a": "b"}, {}, "SELL", live_price=22672, strategy_formed_at="t", decision=dec, use_search=False)
        finally:
            A.requests.post, A.gemini_pool = old_post, old_pool
        self.assertEqual(out["status"], "APPROVED")
        self.assertIn("FAST LANE REPORT", seen["prompt"])
        self.assertIn("FL_MOMENTUM_BREAK", seen["prompt"])

    def test_replay_runs_and_counts_moves(self):
        rng = np.random.default_rng(3)
        c = list(22720 + rng.uniform(-5, 5, 40)) + [22712, 22695, 22680, 22668, 22660, 22655, 22652, 22654, 22653, 22656]
        c += [22660, 22672, 22686, 22700, 22706, 22708, 22706, 22707] + list(22707 + rng.uniform(-4, 4, 10))
        c += [22715, 22730, 22748, 22762, 22775, 22786, 22790, 22788]
        r = fast_lane.replay(_mk(c, start="2026-10-14 08:30"), 50)
        self.assertGreaterEqual(r["stats"]["moves"], 2)
        self.assertEqual(r["stats"]["missed"], 0)


class FastLaneV45(unittest.TestCase):
    def test_plans_compression_gives_both_triggers(self):
        c = _range(40) + [22703, 22701, 22702, 22700, 22703, 22702, 22701, 22703]
        ps = fast_lane.plans(_mk(c))
        kinds = {p["kind"] for p in ps}
        self.assertTrue({"PLAN_BREAKOUT", "PLAN_BREAKDOWN"} <= kinds)
        for p in ps:
            self.assertTrue(p["risk_reward"] >= 1.5)
            self.assertTrue((p["stop_loss"] < p["trigger"]) if p["direction"] == "BUY" else (p["stop_loss"] > p["trigger"]))

    def test_plans_bounce_into_support(self):
        c = _range(30, 22700) + [22690, 22680, 22672, 22664, 22658, 22654, 22652, 22651]
        ps = fast_lane.plans(_mk(c), prev_day={"pdl": 22648, "pdh": 22800})
        self.assertTrue(any(p["kind"] == "PLAN_BOUNCE" and p["direction"] == "BUY" for p in ps))

    def test_plan_prompt_mentions_preapproval(self):
        import ai_trade_decision as A
        seen = {}

        class R:
            status_code = 200
            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": '{"own_direction":"BUY","approved":true,"verdict":"APPROVE","confidence":70,"reason":"ok","risk_flags":[]}'}]}}]}

        def fake_post(url, headers=None, json=None, timeout=0):
            seen["prompt"] = json["contents"][0]["parts"][0]["text"]
            return R()

        class FakePool:
            REVIEW_MODELS = ["m"]
            PoolHTTPError = A.gemini_pool.PoolHTTPError
            GeminiUnavailable = A.gemini_pool.GeminiUnavailable
            @staticmethod
            def run(keys, models, fn, max_attempts=1, deadline_s=1):
                return fn(keys[0], models[0]), 0, models[0]

        old_post, old_pool = A.requests.post, A.gemini_pool
        A.requests.post, A.gemini_pool = fake_post, FakePool
        try:
            dec = {"fast_lane": True, "fast_lane_playbook": "PLAN_BOUNCE", "fast_lane_score": 60, "fast_lane_why": "x", "direction": "BUY",
                   "underlying_entry": 22652, "stop_loss": 22636, "target": 22681, "risk_reward": 1.8, "soft_flags": ["Research against this BUY: x (advisory)"]}
            out = A.gemini_trade_review(["k"], {"a": "b"}, {}, "BUY", live_price=22652, strategy_formed_at="t", decision=dec, use_search=False)
        finally:
            A.requests.post, A.gemini_pool = old_post, old_pool
        self.assertTrue(out["approved"])
        self.assertIn("PRE-APPROVAL", seen["prompt"])
        self.assertIn("ADVICE", seen["prompt"])

    def test_plan_quota_guard_constants(self):
        import ai_trade_decision as A
        self.assertLessEqual(A.FAST_LANE_PLAN_REVIEWS_PER_HOUR, 6)
        self.assertLessEqual(A.FAST_LANE_PLAN_IMMINENT_ATR, 1.5)

    def test_advisory_flag_exists_and_research_veto_is_extreme_only(self):
        import ai_trade_decision as A
        self.assertTrue(A.ADVISORY_GATES)
        self.assertTrue(A.FAST_LANE_REQUIRES_GEMINI)
        self.assertGreaterEqual(A.FAST_LANE_RESEARCH_VETO, 5)


class FastLaneV45Speed(unittest.TestCase):
    def _review(self, playbook):
        import ai_trade_decision as A
        seen = {}

        class R:
            status_code = 200
            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": '{"own_direction":"BUY","approved":true,"verdict":"APPROVE","confidence":70,"reason":"ok","risk_flags":[]}'}]}}]}

        class FakePool:
            REVIEW_MODELS = ["strong-model"]
            PoolHTTPError = A.gemini_pool.PoolHTTPError
            GeminiUnavailable = A.gemini_pool.GeminiUnavailable
            @staticmethod
            def run(keys, models, fn, max_attempts=1, deadline_s=1):
                seen["models"], seen["deadline"], seen["attempts"] = list(models), deadline_s, max_attempts
                return fn(keys[0], models[0]), 0, models[0]

        old_post, old_pool = A.requests.post, A.gemini_pool
        A.requests.post, A.gemini_pool = (lambda url, headers=None, json=None, timeout=0: R()), FakePool
        try:
            dec = {"fast_lane": True, "fast_lane_playbook": playbook, "fast_lane_score": 70, "fast_lane_why": "x", "direction": "BUY",
                   "underlying_entry": 22652, "stop_loss": 22636, "target": 22681, "risk_reward": 1.8}
            A.gemini_trade_review(["k"], {"a": "b"}, {}, "BUY", live_price=22652, strategy_formed_at="t", decision=dec, use_search=False)
        finally:
            A.requests.post, A.gemini_pool = old_post, old_pool
        return seen

    def test_plan_review_uses_lite_models_first_and_short_deadline(self):
        import ai_trade_decision as A
        s = self._review("PLAN_BOUNCE")
        self.assertIn("lite", s["models"][0])
        self.assertEqual(s["deadline"], A.FAST_LANE_REVIEW_DEADLINE_S)

    def test_fast_lane_trade_review_keeps_strong_model_but_short_deadline(self):
        import ai_trade_decision as A
        s = self._review("FL_MOMENTUM_BREAK")
        self.assertEqual(s["models"], ["strong-model"])
        self.assertLessEqual(s["deadline"], 60)
