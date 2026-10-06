"""
tests_edge_v36.py -- tests for the v36 upgrade.
Run:  python -m unittest tests_edge_v36 -v

Covers
  * edge_model         : monotonic prior, learning, calibration, R:R scaling, no-data honesty
  * trade_learning     : intrabar SL/target scan, both-in-one-candle = LOSS, EOD / overnight expiry,
                         EXPIRED trades become learnable, factor flags are NOT wiped (old SQLite bug),
                         same-direction cooldown, loss-streak pause
  * ai_trade_decision  : session window, expectancy gate, strong global veto, target cap, entry meta
"""
import os, sys, types, json, tempfile, unittest
from datetime import datetime, timedelta, time as dtime

for _m in ("feedparser", "streamlit", "yfinance", "upstox_client", "supabase", "plotly"):
    try:
        __import__(_m)
    except Exception:
        sys.modules[_m] = types.ModuleType(_m)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import edge_model
import trade_learning as tl
import ai_trade_decision as atd

KEYS = tl.ALL_FACTOR_KEYS


def _flags(n_true, **extra):
    f = {k: (i < n_true) for i, k in enumerate(KEYS)}
    f.update(extra)
    return f


class EdgeModelTests(unittest.TestCase):
    def test_prior_is_monotonic_and_capped(self):
        ps = [edge_model.rule_prior(n) for n in range(0, 30)]
        self.assertTrue(all(b >= a for a, b in zip(ps, ps[1:])))
        self.assertLessEqual(max(ps), edge_model.PRIOR_MAX)
        self.assertGreaterEqual(min(ps), edge_model.PRIOR_MIN)

    def test_no_history_means_prior_only_and_says_so(self):
        e = edge_model.estimate_edge(_flags(10), [], KEYS, rr=1.5)
        self.assertEqual(e["sample_size"], 0)
        self.assertEqual(e["calibrated_on_trades"], 0)
        self.assertEqual(e["learned_count"], 0)
        self.assertLess(e["win_probability"], 0.60)        # never claims a high win rate it cannot know

    def test_learning_rewards_a_factor_that_really_wins(self):
        rng = np.random.default_rng(1)
        rows = []
        for _ in range(60):
            good = bool(rng.random() < 0.5)
            win = int(rng.random() < (0.75 if good else 0.25))
            rows.append(({"vwap_aligned": good, "ml_agrees": bool(rng.random() < .5)}, win))
        keys = ["vwap_aligned", "ml_agrees"]
        hi = edge_model.estimate_edge({"vwap_aligned": True, "ml_agrees": True}, rows, keys, 1.5)
        lo = edge_model.estimate_edge({"vwap_aligned": False, "ml_agrees": True}, rows, keys, 1.5)
        self.assertGreater(hi["win_probability"], lo["win_probability"] + 0.05)
        self.assertIn("vwap_aligned", hi["learned_factors"])

    def test_calibration_pulls_probability_down_when_engine_really_loses(self):
        rows = [({"vwap_aligned": i % 2 == 0}, 1 if i % 5 == 0 else 0) for i in range(60)]   # 20% real win rate
        e = edge_model.estimate_edge({"vwap_aligned": True}, rows, ["vwap_aligned"], 1.5)
        base = edge_model.estimate_edge({"vwap_aligned": True}, [], ["vwap_aligned"], 1.5)
        self.assertLess(e["win_probability"], base["win_probability"])
        self.assertLess(e["expectancy_r"], 0.0)            # engine would correctly refuse to trade

    def test_expectancy_does_not_reward_far_targets(self):
        a = edge_model.estimate_edge(_flags(10), [], KEYS, rr=1.5)["expectancy_r"]
        b = edge_model.estimate_edge(_flags(10), [], KEYS, rr=5.0)["expectancy_r"]
        self.assertAlmostEqual(a, b, places=1)             # a 5R wall is NOT a bigger "edge" than 1.5R

    def test_required_probability(self):
        self.assertAlmostEqual(edge_model.required_probability(1.5, 0.0), 0.4)
        self.assertAlmostEqual(edge_model.required_probability(2.0, 0.0), 1 / 3, places=4)


FIXED_NOW = datetime(2026, 10, 5, 11, 0, 0)      # a Monday, mid-session -- tests must not depend on the real clock


class DbCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._old = tl.DB_NAME
        self._old_now = tl._now_ist
        tl.DB_NAME = os.path.join(self._dir, "t.db")
        tl._now_ist = lambda: FIXED_NOW
        tl._supabase_url = tl._supabase_key = None
        tl._SEG_CACHE.update({"ts": 0.0, "stats": None})
        tl.init_db()

    def tearDown(self):
        tl.DB_NAME = self._old
        tl._now_ist = self._old_now


class LearningDbTests(DbCase):

    def _open(self, direction="BUY", entry=24000.0, sl=23980.0, tp=24030.0, ts=None, flags=None):
        flags = flags or _flags(8, **{"_entry_meta": {"playbook": "X"}})
        sid = tl.log_setup(direction, 24000, "CE" if direction == "BUY" else "PE", entry, sl, tp, 55.0, flags)
        if ts:
            import sqlite3
            c = sqlite3.connect(tl.DB_NAME)
            c.execute("UPDATE ai_trade_setups SET timestamp=? WHERE id=?", (ts, sid))
            c.commit(); c.close()
        return sid

    def _row(self, sid):
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        r = c.execute("SELECT status, exit_price, factors_json FROM ai_trade_setups WHERE id=?", (sid,)).fetchone()
        c.close()
        return r

    def _bars(self, rows, start):
        idx = pd.date_range(start, periods=len(rows) + 1, freq="5min")
        data = list(rows) + [(rows[-1][0], rows[-1][1])]       # last row = forming candle (ignored)
        return pd.DataFrame({"High": [h for h, _ in data], "Low": [l for _, l in data]}, index=idx)

    def test_intrabar_stop_is_not_missed(self):
        opened = FIXED_NOW - timedelta(minutes=12)
        sid = self._open(ts=opened.strftime("%Y-%m-%d %H:%M:%S"))
        # candle after entry spiked down through the SL, live price has since come back
        df = self._bars([(24005, 23975), (24008, 23995)], opened + timedelta(minutes=5))
        tl.resolve_open_setups(24002.0, df=df)
        st, px, _ = self._row(sid)
        self.assertEqual(st, "LOSS"); self.assertEqual(px, 23980.0)

    def test_candle_touching_both_levels_is_scored_as_loss(self):
        opened = FIXED_NOW - timedelta(minutes=12)
        sid = self._open(ts=opened.strftime("%Y-%m-%d %H:%M:%S"))
        df = self._bars([(24035, 23975), (24001, 23999)], opened + timedelta(minutes=5))
        tl.resolve_open_setups(24000.0, df=df)
        self.assertEqual(self._row(sid)[0], "LOSS")            # pessimistic, never flatters the tracker

    def test_target_touch_between_refreshes_is_a_win(self):
        opened = FIXED_NOW - timedelta(minutes=12)
        sid = self._open(ts=opened.strftime("%Y-%m-%d %H:%M:%S"))
        df = self._bars([(24032, 23995), (24010, 24000)], opened + timedelta(minutes=5))
        tl.resolve_open_setups(24005.0, df=df)
        self.assertEqual(self._row(sid)[0], "WIN")

    def test_bars_before_entry_are_ignored(self):
        opened = FIXED_NOW - timedelta(minutes=2)
        sid = self._open(ts=opened.strftime("%Y-%m-%d %H:%M:%S"))
        df = self._bars([(24050, 23900), (24005, 23995)], opened - timedelta(minutes=10))   # old candles
        tl.resolve_open_setups(24001.0, df=df)
        self.assertEqual(self._row(sid)[0], "OPEN")

    def test_factor_flags_survive_resolution_checks(self):
        """Old SQLite path overwrote factors_json with only the meta on every refresh."""
        sid = self._open(flags=_flags(8, **{"_entry_meta": {"playbook": "KEEP_ME"}}))
        tl.resolve_open_setups(24001.0)
        f = json.loads(self._row(sid)[2])
        self.assertEqual(f["_entry_meta"]["playbook"], "KEEP_ME")
        self.assertTrue(f[KEYS[0]])
        self.assertIn("_learning_meta", f)

    def test_overnight_carry_is_expired_at_last_seen_price_not_gap_open(self):
        yesterday = (FIXED_NOW - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        sid = self._open(ts=yesterday)
        # pretend we saw 24012 during the old session
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        f = json.loads(c.execute("SELECT factors_json FROM ai_trade_setups WHERE id=?", (sid,)).fetchone()[0])
        f["_learning_meta"] = {"last_seen_price": 24012.0}
        c.execute("UPDATE ai_trade_setups SET factors_json=? WHERE id=?", (json.dumps(f), sid)); c.commit(); c.close()
        tl.resolve_open_setups(24200.0)                        # gap-up open next day
        st, px, fj = self._row(sid)
        self.assertEqual(st, "EXPIRED"); self.assertEqual(px, 24012.0)
        self.assertEqual(json.loads(fj)["_learning_meta"]["expired_reason"], "carried_overnight")

    def test_expired_trades_become_learnable_by_their_result(self):
        def add(r):
            f = _flags(5, **{"_learning_meta": {"expired_r": r}})
            return json.dumps(f)
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        for r in (-0.9, 0.8, 0.1, None):
            fj = add(r) if r is not None else json.dumps(_flags(5))
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','BUY',1,'CE',1,1,1,1,?,'EXPIRED')", (fj,))
        c.commit(); c.close()
        labels = [st for _, st in tl._resolved_learning_rows()]
        self.assertEqual(sorted(labels), ["LOSS", "WIN"])      # -0.9R -> LOSS, +0.8R -> WIN, inconclusive/old skipped

    def test_same_direction_cooldown_after_loss(self):
        now = FIXED_NOW
        sid = self._open(direction="BUY")
        tl._sqlite_close(sid, "LOSS", 23980.0)
        blocked, why = tl.entry_cooldown("BUY", now=now + timedelta(minutes=3))
        self.assertTrue(blocked, why)
        self.assertFalse(tl.entry_cooldown("SELL", now=now + timedelta(minutes=3))[0])
        self.assertFalse(tl.entry_cooldown("BUY", now=now + timedelta(minutes=tl.COOLDOWN_AFTER_LOSS_MIN + 2))[0])

    def test_loss_streak_pause_expires(self):
        tl.LOSS_STREAK_PAUSE = 3
        self.addCleanup(setattr, tl, "LOSS_STREAK_PAUSE", None)
        now = FIXED_NOW
        for d in ("BUY", "SELL", "BUY"):
            sid = self._open(direction=d, flags=_flags(8))
            tl._sqlite_close(sid, "LOSS", 1.0)
        # streak pause blocks even the opposite direction ...
        self.assertTrue(tl.entry_cooldown("SELL", now=now + timedelta(minutes=25))[0])
        # ... but only for LOSS_STREAK_PAUSE_MIN minutes
        self.assertFalse(tl.entry_cooldown("SELL", now=now + timedelta(minutes=tl.LOSS_STREAK_PAUSE_MIN + 5))[0])


def _strategy(direction="BUY", score=70.0):
    return {"has_setup": True, "direction": direction, "score": score, "edge": 9.0,
            "selected_strategies": ["Trend Pullback"], "regime": "TREND",
            "live_state": {"score": 45 if direction == "BUY" else -45}}


def _decide(direction="BUY", now=dtime(11, 0), global_research=None, is_choppy=False, strategy=None, live=24000.0, atr=14.0, ladder=None, zone_edge=6.0, chain=None):
    bull = direction == "BUY"
    sr = {"status": "OK", "location": "AT_SUPPORT" if bull else "AT_RESISTANCE",
          "confirmation": {"support_rejection_confirmed": True, "resistance_rejection_confirmed": True},
          "entry_timing": {}, "major_support": {"price": live - 2, "low": live - zone_edge, "high": live - 1, "sources": ["PDL"]},
          "major_resistance": {"price": live + 2, "low": live + 1, "high": live + zone_edge, "sources": ["PDH"]}}
    ctx = {"Data Freshness": "Live / fresh",
           "RAW Regime Engine": "BULLISH" if bull else "BEARISH",
           "RAW Multi-Timeframe Structure": ("bullish bullish bullish" if bull else "bearish bearish bearish"),
           "RAW Order Flow": {"pressure": "BUYING PRESSURE" if bull else "SELLING PRESSURE"}}
    lp = {"status": "APPROACHING_LEVEL", "approaching": "support" if bull else "resistance",
          "level_price": live - 1 if bull else live + 1, "distance_pts": 1.0, "break_pct": 70, "bounce_pct": 72,
          "directional_bias": "Bullish" if bull else "Bearish",
          "factors": ["Heavy Put OI" if bull else "Heavy Call OI", "above a rising VWAP" if bull else "below a falling VWAP"]}
    return atd.generate_trade_decision(
        live_price=live, level_prediction=lp, atr=atr, max_pain=23900 if bull else 24100,
        signal_code=1 if bull else -1, ml_agrees=True, breadth_advances=35 if bull else 12,
        breadth_declines=15 if bull else 38, global_avg_change=0.4 if bull else -0.4, live_vix=13.0,
        india_news_sentiment="BULLISH" if bull else "BEARISH", level_ladder=ladder, sr_context=sr,
        sniper_bias="BULLISH" if bull else "BEARISH", is_choppy=is_choppy, dashboard_context=ctx,
        global_research=global_research, nifty50_news_sentiment="BULLISH" if bull else "BEARISH",
        nifty50_fundamentals_bias="NEUTRAL", strategy_result=strategy or _strategy(direction), raw_option_chain=chain,
        now_ist=datetime(2026, 10, 5, now.hour, now.minute))


class DecisionTests(DbCase):
    def test_trades_inside_session_and_returns_edge_fields(self):
        out = _decide()
        self.assertTrue(out["has_setup"], out.get("reason"))
        for k in ("expectancy_r", "win_probability", "breakeven_probability", "required_ev_r", "risk_reward"):
            self.assertIn(k, out)
        self.assertGreaterEqual(out["risk_reward"], 1.5)
        meta = out["factor_flags"]["_entry_meta"]
        self.assertEqual(meta["side"], "BUY"); self.assertEqual(meta["logic_version"], tl.LOGIC_VERSION)
        self.assertEqual(out["factors_true"], sum(1 for k, v in out["factor_flags"].items() if v and k not in
                         ("_entry_meta", "_playbook_label", "_setup_score", "_reasons", "_edge")))

    def test_session_window_blocks_open_noise_and_late_entries(self):
        self.assertFalse(_decide(now=dtime(9, 16))["has_setup"])
        late = _decide(now=dtime(15, 5))
        self.assertFalse(late["has_setup"]); self.assertIn("entry window", late["reason"])

    def test_strong_opposite_global_research_vetoes(self):
        gr = {"status": "AVAILABLE", "directional_bias": "SELL", "strength": "STRONG"}
        out = _decide("BUY", global_research=gr)
        self.assertFalse(out["has_setup"]); self.assertIn("Global research", out["reason"])
        mild = {"status": "AVAILABLE", "directional_bias": "SELL", "strength": "MODERATE"}
        self.assertTrue(_decide("BUY", global_research=mild)["has_setup"])

    def test_losing_record_stops_the_engine_by_itself(self):
        """With a real 15% win rate the calibrated expectancy turns negative and the engine refuses."""
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        for i in range(40):
            fj = json.dumps(_flags(10, **{"_entry_meta": {"logic_version": tl.LOGIC_VERSION}}))
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','BUY',1,'CE',1,1,1,1,?,?)",
                      (fj, "WIN" if i % 7 == 0 else "LOSS"))
        c.commit(); c.close()
        out = _decide()
        self.assertFalse(out["has_setup"])
        self.assertIn("No positive edge", out["reason"])

    def test_far_oi_wall_target_is_capped(self):
        ladder = {"supports": [], "resistances": [{"level_price": 24120.0, "distance_pts": 120.0,
                  "directional_bias": "Bullish", "break_pct": 70, "bounce_pct": 70, "factors": []}]}
        out = _decide(ladder=ladder)
        self.assertTrue(out["has_setup"], out.get("reason"))
        sl_dist = abs(out["underlying_entry"] - out["stop_loss"])
        self.assertLessEqual(abs(out["target"] - out["underlying_entry"]), atd.MAX_TARGET_R * sl_dist + 0.01)
        self.assertGreaterEqual(out["risk_reward"], 1.5)

    def test_no_track_record_never_claims_more_than_a_coin_flip(self):
        out = _decide()
        self.assertTrue(out["has_setup"], out.get("reason"))
        self.assertLessEqual(out["win_probability_ref"] if "win_probability_ref" in out else out["edge"]["win_probability_ref_rr"],
                             edge_model.UNCALIBRATED_CAP + 1e-9)

    def test_cooldown_blocks_reentry_after_stop_out(self):
        sid = tl.log_setup("BUY", 24000, "CE", 24000.0, 23980.0, 24030.0, 55.0, _flags(8))
        tl._sqlite_close(sid, "LOSS", 23980.0)
        out = _decide("BUY")
        self.assertFalse(out["has_setup"]); self.assertIn("Cooldown", out["reason"])


class GeminiFinalBlockTests(DbCase):
    """Gemini is the final block: anything but an approval means NO trade (no probe, no reduced size)."""
    def _cand(self):
        out = _decide()
        self.assertTrue(out["has_setup"], out.get("reason"))
        return out

    def test_approved_sets_external_ai_factor(self):
        d = atd.apply_gemini_verdict(self._cand(), {"status": "APPROVED", "approved": True})
        self.assertTrue(d["has_setup"]); self.assertTrue(d["factor_flags"]["external_ai_aligned"])

    def test_any_rejection_blocks_the_whole_setup(self):
        for r in ({"status": "REJECTED", "approved": False, "reason": "price extended, chasing"},
                  {"status": "NO_TRADE", "approved": False, "reason": "x"},
                  {"status": "LOCAL_FALLBACK_BLOCKED", "approved": False, "reason": "y"},
                  {"status": "UNAVAILABLE", "approved": False, "reason": "z"}):
            d = atd.apply_gemini_verdict(self._cand(), r)
            self.assertFalse(d["has_setup"]); self.assertIn("Final review did not approve", d["reason"])
            self.assertNotIn("probe", d)

    def test_local_fallback_approval_still_trades(self):
        d = atd.apply_gemini_verdict(self._cand(), {"status": "LOCAL_FALLBACK_APPROVED", "approved": True})
        self.assertTrue(d["has_setup"]); self.assertFalse(d["factor_flags"]["external_ai_aligned"])

    def test_gemini_gets_latest_candles_not_oldest(self):
        rows = [{"Close": 22700.0, "i": i} for i in range(300)]
        for i in range(250, 300):
            rows[i]["Close"] = 22552.9
        ctx = {"RAW Primary Price/Indicator Data (latest 500 candles)": json.dumps(rows)}
        packet = atd._gemini_compact_context(ctx)
        self.assertIn("22552.9", packet); self.assertIn("LAST row = newest", packet)
        self.assertIn('"i":299', packet.replace(" ", ""))


class StopAndDiagnosticsTests(DbCase):
    def test_structure_aware_stop_goes_beyond_the_zone_but_is_capped(self):
        near = _decide(zone_edge=6.0)                       # zone edge 6 pts away -> normal 20 pt stop
        self.assertTrue(near["has_setup"], near.get("reason"))
        self.assertAlmostEqual(abs(near["underlying_entry"] - near["stop_loss"]), 20.0, places=1)
        far = _decide(zone_edge=26.0)                       # edge 26 pts away -> 26 + 0.25*14 = 29.5
        self.assertTrue(far["has_setup"], far.get("reason"))
        self.assertAlmostEqual(abs(far["underlying_entry"] - far["stop_loss"]), atd.MAX_SL_PTS, places=1)   # 26+3.5=29.5 -> capped at 28
        self.assertGreaterEqual(far["risk_reward"], 1.5)
        huge = _decide(zone_edge=80.0)                      # absurd zone -> capped
        self.assertTrue(huge["has_setup"], huge.get("reason"))
        self.assertLessEqual(abs(huge["underlying_entry"] - huge["stop_loss"]), atd.MAX_SL_PTS + 1e-6)

    def test_classify_block_names_the_gate(self):
        c = atd.classify_block
        self.assertEqual(c("Final review did not approve this strategy+AI setup: x"), "Gemini final review")
        self.assertEqual(c("No positive edge: estimated win chance"), "No positive edge (EV)")
        self.assertEqual(c("Cooldown: last BUY trade was stopped out"), "Cooldown / pause")
        self.assertEqual(c("something unrelated"), "Other")

    def test_light_defaults(self):
        self.assertEqual(tl.COOLDOWN_AFTER_LOSS_MIN, 10)
        self.assertIsNone(tl.LOSS_STREAK_PAUSE)
        for d in ("BUY", "SELL", "BUY", "SELL"):               # a long losing run no longer freezes the engine by itself
            sid = tl.log_setup(d, 24000, "CE", 24000.0, 23980.0, 24030.0, 55.0, _flags(8)); tl._sqlite_close(sid, "LOSS", 1.0)
        self.assertFalse(tl.entry_cooldown("BUY", now=FIXED_NOW + timedelta(minutes=30))[0])


class BackgroundReviewTests(unittest.TestCase):
    """The page must never wait on Gemini."""
    def setUp(self):
        atd._REVIEW_JOBS.clear()
        self._orig = atd.gemini_trade_review

    def tearDown(self):
        atd.gemini_trade_review = self._orig
        atd._REVIEW_JOBS.clear()

    def test_returns_immediately_then_result_on_next_call(self):
        import time, threading
        gate = threading.Event()
        def slow(*a, **k):
            gate.wait(5)
            return {"status": "APPROVED", "approved": True}
        atd.gemini_trade_review = slow
        t0 = time.time()
        r1 = atd.gemini_review_async(("c", "BUY"), ["k"], {}, {}, "BUY")
        self.assertLess(time.time() - t0, 0.5)               # did not wait for the 'slow Gemini'
        self.assertEqual(r1["status"], "PENDING"); self.assertFalse(r1["approved"])
        r2 = atd.gemini_review_async(("c", "BUY"), ["k"], {}, {}, "BUY")
        self.assertEqual(r2["status"], "PENDING")             # still running; no second job started
        self.assertEqual(sum(1 for t in threading.enumerate() if t.name == "gemini-review"), 1)
        gate.set()
        for _ in range(50):
            r3 = atd.gemini_review_async(("c", "BUY"), ["k"], {}, {}, "BUY")
            if r3["status"] != "PENDING": break
            time.sleep(0.05)
        self.assertEqual(r3["status"], "APPROVED")

    def test_crashing_review_does_not_leave_job_running(self):
        import time
        def boom(*a, **k): raise RuntimeError("x")
        atd.gemini_trade_review = boom
        atd.gemini_review_async(("c2", "SELL"), ["k"], {}, {}, "SELL")
        for _ in range(50):
            r = atd.gemini_review_async(("c2", "SELL"), ["k"], {}, {}, "SELL")
            if r["status"] != "PENDING": break
            time.sleep(0.05)
        self.assertEqual(r["status"], "UNAVAILABLE")

    def test_pending_blocks_trade_and_is_classified(self):
        cand = {"has_setup": True, "factor_flags": {}}
        d = atd.apply_gemini_verdict(cand, {"status": "PENDING", "approved": False,
                                            "reason": "Gemini review background me shuru hua"})
        self.assertFalse(d["has_setup"])
        self.assertEqual(atd.classify_block(d["reason"]), "Gemini review pending (background)")

    def test_pool_deadline_stops_retry_storm(self):
        import gemini_pool, time
        calls = []
        def fail(key, model):
            calls.append(1); time.sleep(0.12)
            raise gemini_pool.PoolHTTPError(503, "overloaded")
        t0 = time.time()
        with self.assertRaises(gemini_pool.GeminiUnavailable):
            gemini_pool.run(["a", "b", "c"], ["m1", "m2", "m3", "m4"], fail, max_attempts=30, deadline_s=0.3)
        self.assertLess(time.time() - t0, 1.5)
        self.assertLess(len(calls), 8)


class FallbackTests(DbCase):
    def test_fail_open_is_on_like_before(self):
        self.assertTrue(atd.GEMINI_FAIL_OPEN)

    def test_raw_unavailable_is_never_an_approval(self):
        d = atd.apply_gemini_verdict(_decide(), {"status": "UNAVAILABLE", "approved": False, "reason": "cooling down"})
        self.assertFalse(d["has_setup"])

    def test_good_setup_passes_local_fallback_when_gemini_down(self):
        cand = _decide()
        fb = atd.local_fallback_review(cand, {"status": "UNAVAILABLE", "reason": "all keys cooling down"})
        self.assertEqual(fb["status"], "LOCAL_FALLBACK_APPROVED")

    def test_weak_setup_is_blocked_by_local_fallback(self):
        cand = dict(_decide()); cand["major_conflicts"] = 2
        self.assertEqual(atd.local_fallback_review(cand)["status"], "LOCAL_FALLBACK_BLOCKED")
        cand = dict(_decide()); cand["risk_reward"] = 1.2
        self.assertEqual(atd.local_fallback_review(cand)["status"], "LOCAL_FALLBACK_BLOCKED")
        cand = dict(_decide()); cand["expectancy_r"] = 0.0
        self.assertEqual(atd.local_fallback_review(cand)["status"], "LOCAL_FALLBACK_BLOCKED")

    def test_fallback_trade_is_recorded_as_fallback_not_gemini(self):
        cand = _decide()
        fb = atd.local_fallback_review(cand, {"status": "UNAVAILABLE", "reason": "429 quota"})
        fb["gemini_error"] = "429 quota"
        d = atd.apply_gemini_verdict(cand, fb)
        self.assertTrue(d["has_setup"])
        g = d["factor_flags"]["_gemini"]
        self.assertEqual(g["reviewer"], "LOCAL_FALLBACK"); self.assertEqual(g["gemini_error"], "429 quota")
        self.assertFalse(d["factor_flags"]["external_ai_aligned"])
        sid = tl.log_setup(d["direction"], d["strike"], d["option_type"], d["underlying_entry"], d["stop_loss"],
                           d["target"], d["confidence_pct"], d["factor_flags"])
        self.assertEqual(tl.get_recent_setups(limit=1)[0]["gemini"]["reviewer"], "LOCAL_FALLBACK")

    def test_real_gemini_approval_is_recorded_as_gemini(self):
        d = atd.apply_gemini_verdict(_decide(), {"status": "APPROVED", "approved": True, "model": "gemini-x", "reason": "ok"})
        self.assertEqual(d["factor_flags"]["_gemini"]["reviewer"], "GEMINI")


class GeminiProofStoredTests(DbCase):
    def test_approved_trade_stores_gemini_report_and_reads_back(self):
        cand = _decide()
        rev = {"status": "APPROVED", "approved": True, "model": "gemini-3.8-flash", "search_used": True,
               "confidence": 82, "reason": "Structure, VWAP and OI all support the BUY."}
        d = atd.apply_gemini_verdict(cand, rev)
        self.assertTrue(d["has_setup"])
        tl.log_setup(d["direction"], d["strike"], d["option_type"], d["underlying_entry"], d["stop_loss"],
                     d["target"], d["confidence_pct"], d["factor_flags"])
        row = tl.get_recent_setups(limit=1)[0]
        self.assertEqual(row["gemini"]["status"], "APPROVED")
        self.assertEqual(row["gemini"]["model"], "gemini-3.8-flash")
        self.assertIn("OI", row["gemini"]["reason"])

    def test_old_trade_without_record_reads_as_none(self):
        tl.log_setup("BUY", 24000, "CE", 24000.0, 23980.0, 24030.0, 55.0, _flags(8))
        self.assertIsNone(tl.get_recent_setups(limit=1)[0]["gemini"])

    def test_gemini_proof_does_not_leak_into_learning_factors(self):
        d = atd.apply_gemini_verdict(_decide(), {"status": "APPROVED", "approved": True, "model": "m", "reason": "r"})
        sid = tl.log_setup(d["direction"], d["strike"], d["option_type"], d["underlying_entry"], d["stop_loss"],
                           d["target"], d["confidence_pct"], d["factor_flags"])
        tl._sqlite_close(sid, "WIN", d["target"])
        rows = tl._labelled_flag_rows()
        self.assertEqual(len(rows), 1)
        self.assertNotIn("_gemini", rows[0][0])


class OiWallTargetTests(DbCase):
    CHAIN = pd.DataFrame({"Strike": [24010, 24030, 24050, 24060, 24100, 23990, 23950, 23940, 23900],
                          "Call OI": [100, 200, 900, 300, 1000, 10, 10, 10, 10],
                          "Put OI":  [10, 10, 10, 10, 10, 120, 300, 900, 1000]})

    def test_parser_understands_real_chain_columns(self):
        for raw in (self.CHAIN, self.CHAIN.to_json(orient="records")):
            self.assertIsNotNone(atd._oi_wall_target(raw, 24000.0, "BUY"))
            self.assertIsNotNone(atd._oi_wall_target(raw, 24000.0, "SELL"))

    def test_target_sits_in_front_of_nearest_significant_wall(self):
        t, wall = atd._oi_wall_target(self.CHAIN, 24000.0, "BUY", atr=14.0)
        self.assertEqual(wall, 24050.0)                      # 24050 (900) is >= 60% of the max 1000; 24030 (200) is not
        self.assertAlmostEqual(t, 24050 - max(atd.OI_FRONT_RUN_MIN_PTS, 0.25 * 14.0), places=1)
        t2, wall2 = atd._oi_wall_target(self.CHAIN, 24000.0, "SELL", atr=14.0)
        self.assertEqual(wall2, 23940.0)
        self.assertGreater(t2, wall2)                        # SELL target stops ABOVE the put wall

    def test_decision_uses_the_wall_as_target(self):
        out = _decide(live=24000.0, chain=self.CHAIN)
        self.assertTrue(out["has_setup"], out.get("reason"))
        # BUY: nearest significant call wall is 24050 -> target just in front of it
        self.assertAlmostEqual(out["target"], 24050 - max(atd.OI_FRONT_RUN_MIN_PTS, 0.25 * 14.0), places=1)
        self.assertIn("OI wall", out["target_note"])
        self.assertGreaterEqual(out["risk_reward"], 1.2)

    def test_wall_too_close_lowers_probability_but_does_not_block(self):
        near = pd.DataFrame({"Strike": [24015, 24100], "Call OI": [1000, 300], "Put OI": [10, 10]})
        base = _decide(live=24000.0)
        out = _decide(live=24000.0, chain=near)
        self.assertTrue(out["has_setup"], out.get("reason"))
        self.assertLess(out["win_probability"], base["win_probability"])
        self.assertIn("probability reduced", out["target_note"])

    def test_old_losing_logic_does_not_freeze_new_logic(self):
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        for i in range(40):                                   # 40 trades from an OLD logic, 10% win rate
            fj = json.dumps(_flags(10, **{"_entry_meta": {"logic_version": "v22_opportunity_first"}}))
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','BUY',1,'CE',1,1,1,1,?,?)",
                      (fj, "WIN" if i % 10 == 0 else "LOSS"))
        c.commit(); c.close()
        out = _decide()
        self.assertTrue(out["has_setup"], out.get("reason"))   # fresh logic is judged on its own record


class TimeStopTests(DbCase):
    def _open_at(self, minutes_ago):
        ts = (FIXED_NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")
        sid = tl.log_setup("BUY", 24000, "CE", 24000.0, 23980.0, 24040.0, 55.0, _flags(8))
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME); c.execute("UPDATE ai_trade_setups SET timestamp=? WHERE id=?", (ts, sid)); c.commit(); c.close()
        return sid

    def _status(self, sid):
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        r = c.execute("SELECT status, factors_json FROM ai_trade_setups WHERE id=?", (sid,)).fetchone(); c.close()
        return r

    def test_stagnant_trade_is_closed_after_time_stop(self):
        sid = self._open_at(tl.TIME_STOP_MIN + 5)
        tl.resolve_open_setups(24002.0)                       # barely moved (0.1R)
        st, fj = self._status(sid)
        self.assertEqual(st, "EXPIRED")
        self.assertEqual(json.loads(fj)["_learning_meta"]["expired_reason"], "time_stop_no_progress")

    def test_trade_that_made_progress_keeps_running(self):
        sid = self._open_at(tl.TIME_STOP_MIN + 5)
        df = pd.DataFrame({"High": [24012.0, 24006.0, 24004.0], "Low": [23999.0, 24001.0, 24002.0]},
                          index=pd.date_range(FIXED_NOW - timedelta(minutes=tl.TIME_STOP_MIN), periods=3, freq="5min"))
        tl.resolve_open_setups(24003.0, df=df)                # best excursion +12 pts = 0.6R >= 0.35R
        self.assertEqual(self._status(sid)[0], "OPEN")

    def test_young_trade_is_not_time_stopped(self):
        sid = self._open_at(20)
        tl.resolve_open_setups(24001.0)
        self.assertEqual(self._status(sid)[0], "OPEN")


class OneSetupOneTradeTests(DbCase):
    def _closed_trade(self, direction="BUY", minutes_ago=25, status="WIN"):
        ts = (FIXED_NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")
        sid = tl.log_setup(direction, 24000, "CE" if direction == "BUY" else "PE", 24000.0, 23980.0, 24030.0, 55.0, _flags(8))
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME); c.execute("UPDATE ai_trade_setups SET timestamp=? WHERE id=?", (ts, sid)); c.commit(); c.close()
        tl._sqlite_close(sid, status, 24030.0)

    def _sig(self, minutes_ago):
        s = dict(_strategy("BUY")); s["formed_at"] = (FIXED_NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S"); return s

    def test_same_signal_cannot_be_traded_twice_after_a_win(self):
        self._closed_trade(status="WIN")
        out = _decide("BUY", strategy=self._sig(30))             # formed BEFORE the trade was opened
        self.assertFalse(out["has_setup"]); self.assertIn("Same setup already traded", out["reason"])
        self.assertEqual(atd.classify_block(out["reason"]), "Same setup already traded")

    def test_same_signal_cannot_be_traded_twice_after_a_loss_either(self):
        self._closed_trade(status="LOSS", minutes_ago=40)        # old enough that the cooldown is over
        self.assertFalse(_decide("BUY", strategy=self._sig(50))["has_setup"])

    def test_a_genuinely_new_signal_is_allowed_with_fresh_research(self):
        self._closed_trade(status="WIN", minutes_ago=25)
        out = _decide("BUY", strategy=self._sig(5))              # formed 20 min AFTER the previous trade opened
        self.assertTrue(out["has_setup"], out.get("reason"))

    def test_opposite_direction_is_not_affected(self):
        self._closed_trade(direction="BUY", status="WIN")
        s = dict(_strategy("SELL")); s["formed_at"] = (FIXED_NOW - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        self.assertTrue(_decide("SELL", strategy=s)["has_setup"])

    def test_tz_aware_formed_at_is_understood(self):
        self._closed_trade(status="WIN")
        s = dict(_strategy("BUY")); s["formed_at"] = (FIXED_NOW - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S") + "+05:30"
        self.assertFalse(_decide("BUY", strategy=s)["has_setup"])


    def test_signal_on_the_candle_right_after_the_trade_is_fresh(self):
        # trade opened 11:00:00 from the candle that opened 10:55; the NEXT candle (opens 11:00) forms its own signal
        self._closed_trade(status="WIN", minutes_ago=0)
        self.assertTrue(atd.trade_learning.signal_already_traded("BUY", "2026-10-05 10:55:00")[0])
        self.assertFalse(atd.trade_learning.signal_already_traded("BUY", "2026-10-05 11:00:00")[0])


if __name__ == "__main__":
    unittest.main()
