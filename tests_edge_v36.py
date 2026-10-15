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
        try:
            import shadow_trades as _st; _st._CACHE.clear()
        except Exception:
            pass

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
        self.assertEqual(out["factors_true"], sum(1 for k, v in out["factor_flags"].items() if v and not str(k).startswith("_")))

    def test_session_window_blocks_open_noise_and_late_entries(self):
        self.assertFalse(_decide(now=dtime(9, 16))["has_setup"])
        late = _decide(now=dtime(15, 12))   # entry window now runs to 15:10 (was 14:45)
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
        # BUY: nearest significant call wall is 24050 -> target just in front of it (46) but never beyond what price can reach:
        # 20 pt stop x MAX_TARGET_R 2.2 = 44 pts
        wall_target = 24050 - max(atd.OI_FRONT_RUN_MIN_PTS, 0.25 * 14.0)
        self.assertAlmostEqual(out["target"], min(wall_target, 24000 + atd.MAX_TARGET_R * 20.0), places=1)
        self.assertLessEqual(out["target"], wall_target)
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


class SafeIoTests(unittest.TestCase):
    def setUp(self):
        import safe_io
        self.sio = safe_io
        safe_io._STATE.clear()

    def test_slow_source_is_skipped_after_timeout_and_picked_up_next_time(self):
        import time, threading
        gate = threading.Event()
        calls = []
        def slow():
            calls.append(1); gate.wait(5); return "FRESH"
        t0 = time.time()
        r1 = self.sio.guarded("k1", slow, timeout=0.3, default="DEFAULT")
        self.assertLess(time.time() - t0, 1.0); self.assertEqual(r1, "DEFAULT")       # page was NOT blocked
        r2 = self.sio.guarded("k1", slow, timeout=0.2, default="DEFAULT")
        self.assertEqual(r2, "DEFAULT"); self.assertEqual(len(calls), 1)               # no duplicate job
        gate.set(); time.sleep(0.3)
        self.assertEqual(self.sio.guarded("k1", slow, timeout=1.0, default="DEFAULT"), "FRESH")

    def test_error_keeps_last_good_value(self):
        state = {"n": 0}
        def flaky():
            state["n"] += 1
            if state["n"] == 2: raise RuntimeError("boom")
            return f"v{state['n']}"
        self.assertEqual(self.sio.guarded("k2", flaky, timeout=1, default="D"), "v1")
        self.assertEqual(self.sio.guarded("k2", flaky, timeout=1, default="D"), "v1")   # 2nd call failed -> last good
        self.assertEqual(self.sio.guarded("k2", flaky, timeout=1, default="D"), "v3")

    def test_first_call_error_returns_default(self):
        def bad(): raise ValueError("x")
        self.assertEqual(self.sio.guarded("k3", bad, timeout=1, default=("a", None)), ("a", None))

    def test_stepper_reports_slowest_steps(self):
        import time
        class PH:
            def __init__(self): self.msgs = []
            def caption(self, m): self.msgs.append(m)
        ph = PH(); sp = self.sio.Stepper(ph)
        sp.step("A"); time.sleep(0.05); sp.step("B"); time.sleep(0.12); total, rows = sp.done()
        self.assertEqual([r[0] for r in rows], ["A", "B"])
        self.assertIn("Loading", ph.msgs[0]); self.assertIn("load hua", ph.msgs[-1])

    def test_breadth_placeholder_has_the_same_shape_as_the_real_return(self):
        import streamlit as _st
        if not hasattr(_st, "cache_data"):
            self.skipTest("real streamlit not installed in this environment")
        import market_breadth
        out = market_breadth.unavailable_heavyweights()
        self.assertEqual(len(out), 6); self.assertIsInstance(out[0], pd.DataFrame)
        self.assertIsNone(out[1])


def _candles(closes, rng=6.0, rsi=55.0, wicks=None):
    """Synthetic 5-min frame with the indicator columns the app's df has."""
    import numpy as _np
    n = len(closes)
    opens = [closes[0] - 1] + list(closes[:-1])
    idx = pd.date_range("2026-10-05 09:15", periods=n, freq="5min")
    df = pd.DataFrame({"Open": opens, "Close": closes,
                       "High": [max(o, c) + rng / 2 for o, c in zip(opens, closes)],
                       "Low": [min(o, c) - rng / 2 for o, c in zip(opens, closes)]}, index=idx)
    df["ATR"] = 14.0
    df["RSI"] = rsi
    df["EMA_20"] = pd.Series(closes, index=idx).ewm(span=20, adjust=False).mean()
    df["BB_Upper"] = df["EMA_20"] + 30
    df["BB_Lower"] = df["EMA_20"] - 30
    return df


class EntryQualityTests(unittest.TestCase):
    import entry_quality as eqm

    def _flat_then(self, tail):
        base = [24000.0 + (i % 3) for i in range(30)]
        return _candles(base + tail)

    def test_calm_pullback_entry_is_clean(self):
        closes = [24000 + (i % 4) for i in range(40)]
        out = self.eqm.assess_entry(_candles(closes, rsi=52), "BUY")
        self.assertFalse(out["chase_flags"], out["summary"])
        self.assertLessEqual(out["penalty_logit"], 0.0)

    def test_buying_the_top_of_a_vertical_run_is_flagged(self):
        run = [24000 + 14 * k for k in range(1, 9)]                     # 8 straight up candles, +112 pts
        df = self._flat_then(run); df.loc[df.index[-2], "RSI"] = 79.0
        out = self.eqm.assess_entry(df, "BUY")
        self.assertTrue(out["severe"]); self.assertGreaterEqual(len(out["chase_flags"]), 3)
        self.assertGreater(out["penalty_logit"], 0.4)
        self.assertTrue(self.eqm.fallback_blocks(out)[0])

    def test_selling_the_bottom_is_flagged_symmetrically(self):
        run = [24000 - 14 * k for k in range(1, 9)]
        df = self._flat_then(run); df.loc[df.index[-2], "RSI"] = 21.0
        out = self.eqm.assess_entry(df, "SELL")
        self.assertTrue(out["severe"]); self.assertTrue(self.eqm.fallback_blocks(out)[0])
        self.assertFalse(self.eqm.assess_entry(df, "BUY")["severe"])      # same chart is not a chase for the OTHER side

    def test_liquidity_sweep_reversal_is_a_confirmation(self):
        closes = [24000.0 + (i % 5) for i in range(30)] + [23996, 23992, 23975, 24004, 24010, 24012]
        df = _candles(closes, rng=6.0, rsi=48)
        df.loc[df.index[-4], "Low"] = 23950.0                              # wick under the prior swing low
        out = self.eqm.assess_entry(df, "BUY")
        self.assertTrue(any("liquidity sweep" in c for c in out["confirm_flags"]), out["summary"])

    def test_confirmation_unblocks_a_mild_chase(self):
        eq = {"metrics": {"x": 1}, "chase_flags": ["a", "b"], "confirm_flags": ["liquidity sweep"], "severe": False}
        self.assertFalse(self.eqm.fallback_blocks(eq)[0])
        eq["confirm_flags"] = []
        self.assertTrue(self.eqm.fallback_blocks(eq)[0])

    def test_bad_input_never_raises(self):
        self.assertEqual(self.eqm.assess_entry(None, "BUY")["chase_flags"], [])
        self.assertEqual(self.eqm.assess_entry(pd.DataFrame({"Close": [1, 2]}), "BUY")["chase_flags"], [])


class FallbackHasPriceActionSenseTests(DbCase):
    def _candidate(self, eq=None):
        d = dict(_decide())
        d["entry_quality"] = eq
        return d

    def test_fallback_blocks_a_chase_that_numbers_alone_would_approve(self):
        good = self._candidate()
        self.assertEqual(atd.local_fallback_review(good)["status"], "LOCAL_FALLBACK_APPROVED")
        chase = self._candidate({"metrics": {"rsi": 78}, "chase_flags": ["RSI 78 overbought", "price 3.2 ATR from EMA20"],
                                 "confirm_flags": [], "severe": False})
        out = atd.local_fallback_review(chase)
        self.assertEqual(out["status"], "LOCAL_FALLBACK_BLOCKED"); self.assertIn("CHASE", out["reason"])

    def test_fallback_needs_5_confirmations(self):
        d = self._candidate(); d["critical_confirmations"] = 4
        self.assertEqual(atd.local_fallback_review(d)["status"], "LOCAL_FALLBACK_BLOCKED")


class DecisionUsesEntryQualityTests(DbCase):
    def test_mild_chase_lowers_probability_but_does_not_block_by_itself(self):
        base = _decide()
        mild = {"BUY": {"penalty_logit": 0.2, "chase_flags": ["RSI 72"], "confirm_flags": [], "severe": False, "summary": "x"}}
        res = _decide_eq(mild)
        self.assertTrue(res["has_setup"], res.get("reason"))
        self.assertLess(res["win_probability"], base["win_probability"])
        self.assertFalse(res["factor_flags"]["entry_not_extended"])

    def test_heavy_chase_is_declined_by_the_engine_itself(self):
        heavy = {"BUY": {"penalty_logit": 0.6, "chase_flags": ["RSI 79", "3.4 ATR from EMA20", "8 candles in a row"],
                         "confirm_flags": [], "severe": True, "summary": "x"}}
        res = _decide_eq(heavy)
        self.assertFalse(res["has_setup"]); self.assertIn("No positive edge", res["reason"])

    def test_pcr_velocity_against_the_trade_costs_a_little(self):
        base = _decide()
        res = _decide_eq(None, pcr={"bias": "SELL", "per15": -0.08})
        self.assertTrue(res["has_setup"], res.get("reason"))
        self.assertLessEqual(res["win_probability"], base["win_probability"])
        res2 = _decide_eq(None, pcr={"bias": "BUY", "per15": 0.08})
        self.assertTrue(res2["factor_flags"]["pcr_velocity_aligned"])


def _decide_eq(eq, pcr=None):
    """_decide() but passing entry_quality / pcr_velocity (kept separate so older helpers stay untouched)."""
    orig = atd.generate_trade_decision
    def wrapped(*a, **k):
        k["entry_quality"] = eq; k["pcr_velocity"] = pcr
        return orig(*a, **k)
    atd.generate_trade_decision = wrapped
    try:
        return _decide()
    finally:
        atd.generate_trade_decision = orig


class PcrVelocityTests(unittest.TestCase):
    def setUp(self):
        import pcr_velocity
        self.pv = pcr_velocity; pcr_velocity.reset()

    def test_needs_history_first(self):
        self.pv.update(1.0, now=1000)
        self.assertEqual(self.pv.bias(now=1005)["bias"], "NEUTRAL")

    def test_rising_and_falling_pcr(self):
        for k in range(12):
            self.pv.update(0.90 + 0.01 * k, now=1000 + 60 * k)             # +0.01 / min = +0.15 per 15 min
        self.assertEqual(self.pv.bias(now=1000 + 60 * 11)["bias"], "BUY")
        self.pv.reset()
        for k in range(12):
            self.pv.update(1.10 - 0.01 * k, now=1000 + 60 * k)
        self.assertEqual(self.pv.bias(now=1000 + 60 * 11)["bias"], "SELL")

    def test_flat_pcr_is_neutral_and_bad_values_ignored(self):
        for k in range(12):
            self.pv.update(1.0, now=1000 + 60 * k)
        self.pv.update("abc", now=2000); self.pv.update(-3, now=2001)
        self.assertEqual(self.pv.bias(now=1000 + 60 * 11)["bias"], "NEUTRAL")


class TargetRealismAndGeminiTimeoutTests(DbCase):
    def test_far_wall_target_is_capped_to_what_price_can_reach(self):
        far = pd.DataFrame({"Strike": [24100.0], "Call OI": [1000.0], "Put OI": [10.0]})   # wall 100 pts away
        out = _decide(live=24000.0, chain=far, zone_edge=26.0)                             # 28 pt stop
        self.assertTrue(out["has_setup"], out.get("reason"))
        dist = abs(out["target"] - out["underlying_entry"]); sl = abs(out["underlying_entry"] - out["stop_loss"])
        reach = atd.TARGET_REACH_ATR * 14.0 * (tl.MAX_HOLD_MINUTES / 5.0) ** 0.5
        self.assertLessEqual(dist, max(reach, 1.5 * sl) + 0.01)
        self.assertLessEqual(dist, atd.MAX_TARGET_R * sl + 0.01)
        self.assertGreaterEqual(out["risk_reward"], 1.5)

    def test_gemini_is_given_realistic_time_in_the_background(self):
        self.assertGreaterEqual(atd.REVIEW_DEADLINE_S, 120)
        import inspect
        src = inspect.getsource(atd.gemini_trade_review)
        self.assertIn("body_search, 40", src); self.assertIn("body_json, 25", src)


import shadow_trades
import learner_nn


class ShadowTradeTests(DbCase):
    def setUp(self):
        super().setUp()
        shadow_trades._state.update({"sb_ok": False, "sb_checked": 1e18})      # force the local backend

    def _cand(self, d="BUY", entry=24000.0, sl=23980.0, tp=24040.0):
        return {"direction": d, "underlying_entry": entry, "stop_loss": sl, "target": tp, "factor_flags": _flags(8)}

    def test_log_rejects_nonsense_and_dedupes(self):
        self.assertIsNone(shadow_trades.log({"direction": "BUY", "underlying_entry": 100, "stop_loss": 120, "target": 90}, "x", now=FIXED_NOW))
        self.assertIsNotNone(shadow_trades.log(self._cand(), "Gemini rejected", now=FIXED_NOW))
        self.assertIsNone(shadow_trades.log(self._cand(), "Gemini rejected", now=FIXED_NOW + timedelta(minutes=1)))      # one open per direction
        self.assertIsNotNone(shadow_trades.log(self._cand("SELL", 24000, 24020, 23960), "x", now=FIXED_NOW))             # other side is separate

    def test_daily_cap(self):
        for k in range(shadow_trades.SHADOW_MAX_PER_DAY + 5):
            sid = shadow_trades.log(self._cand(), "x", now=FIXED_NOW + timedelta(minutes=20 * k))
            if sid:
                shadow_trades._update(sid, {"status": "LOSS"})
        today = [r for r in shadow_trades._all_rows() if r["timestamp"][:10] == FIXED_NOW.strftime("%Y-%m-%d")]
        self.assertLessEqual(len(today), shadow_trades.SHADOW_MAX_PER_DAY)

    def test_resolves_win_loss_and_both_in_one_candle_is_loss(self):
        a = shadow_trades.log(self._cand(), "A", now=FIXED_NOW - timedelta(minutes=20))
        idx = pd.date_range(FIXED_NOW - timedelta(minutes=15), periods=3, freq="5min")
        df = pd.DataFrame({"High": [24045.0, 24001.0, 24001.0], "Low": [23999.0, 23999.0, 23999.0]}, index=idx)
        shadow_trades.resolve(24001.0, df)
        self.assertEqual(shadow_trades._all_rows()[0]["status"], "WIN")
        b = shadow_trades.log(self._cand("SELL", 24000, 24020, 23960), "B", now=FIXED_NOW - timedelta(minutes=20))
        df2 = pd.DataFrame({"High": [24025.0, 24001.0, 24001.0], "Low": [23955.0, 23999.0, 23999.0]}, index=idx)
        shadow_trades.resolve(24001.0, df2)
        st = {r["id"]: r["status"] for r in shadow_trades._all_rows()}
        self.assertEqual(st[b], "LOSS")

    def test_summary_tells_whether_the_gate_was_right(self):
        for k, status in enumerate(["LOSS", "LOSS", "LOSS", "WIN"]):
            sid = shadow_trades.log(self._cand(), "Gemini rejected", now=FIXED_NOW - timedelta(hours=3) + timedelta(minutes=20 * k))
            shadow_trades._update(sid, {"status": status})
        s = shadow_trades.summary()["by_gate"]["Gemini rejected"]
        self.assertEqual((s["n"], s["wins"], s["losses"], s["win_rate"]), (4, 1, 3, 25.0))

    def test_shadow_rows_are_weighted_and_do_not_touch_real_counts(self):
        sid = shadow_trades.log(self._cand(), "x", now=FIXED_NOW - timedelta(hours=2))
        shadow_trades._update(sid, {"status": "WIN"})
        rows = tl._labelled_rows_full()
        self.assertEqual([(r["source"], r["weight"]) for r in rows], [("shadow", shadow_trades.SHADOW_WEIGHT)])
        self.assertEqual(tl.get_overall_track_record()["sample_size"], 0)      # real record unchanged
        self.assertIsNone(tl.get_open_setup())                                 # a shadow never blocks / becomes a real trade


class EdgeModelWeightTests(unittest.TestCase):
    def test_weighted_rows_count_less(self):
        keys = ["vwap_aligned"]
        real = [({"vwap_aligned": i % 2 == 0}, 1 if i % 2 == 0 else 0, 1.0) for i in range(40)]
        shadow = [({"vwap_aligned": i % 2 == 0}, 1 if i % 2 == 0 else 0, 0.5) for i in range(40)]
        a = edge_model.factor_lifts(real, keys)["vwap_aligned"]["n_true"]
        b = edge_model.factor_lifts(shadow, keys)["vwap_aligned"]["n_true"]
        self.assertAlmostEqual(b, a / 2, places=1)
        self.assertEqual(edge_model.estimate_edge({"vwap_aligned": True}, real[:2], keys, 1.5)["sample_size"], 2.0)

    def test_old_two_tuple_rows_still_work(self):
        e = edge_model.estimate_edge({"vwap_aligned": True}, [({"vwap_aligned": True}, 1)] * 20, ["vwap_aligned"], 1.5)
        self.assertEqual(e["sample_size"], 20.0)


class LearnerNnTests(unittest.TestCase):
    KEYS = tl.ALL_FACTOR_KEYS

    def _history(self, n, rng, chase_loses=True):
        rows = []
        for _ in range(n):
            f = {k: bool(rng.random() < 0.5) for k in self.KEYS}
            rsi = float(rng.normal(55, 12))
            feats = {"rsi": rsi, "ema20_dist_atr": float(rng.normal(0.8, 1)), "rr": 1.7, "hour": float(rng.integers(9, 15)), "is_buy": 1.0}
            win = (rng.random() < 0.12) if (chase_loses and rsi > 68) else (rng.random() < 0.62)
            rows.append({"flags": f, "label": int(win), "weight": 1.0, "feats": feats})
        return rows

    def test_memory_penalises_a_repeat_of_past_mistakes_and_rewards_clean_setups(self):
        rng = np.random.default_rng(3); rows = self._history(80, rng)
        allf = {k: True for k in self.KEYS}
        chase = learner_nn.learned_adjustment(allf, {"rsi": 78.0, "ema20_dist_atr": 2.5, "rr": 1.7, "hour": 11.0, "is_buy": 1.0}, self.KEYS, rows)
        calm = learner_nn.learned_adjustment(allf, {"rsi": 50.0, "ema20_dist_atr": 0.3, "rr": 1.7, "hour": 11.0, "is_buy": 1.0}, self.KEYS, rows)
        self.assertLess(chase["delta_logit"], -0.15); self.assertGreater(calm["delta_logit"], 0.0)
        self.assertIn("most similar past setups lost", chase["note"])

    def test_no_history_means_no_opinion(self):
        out = learner_nn.learned_adjustment({k: True for k in self.KEYS}, {"rsi": 50.0}, self.KEYS, [])
        self.assertEqual(out["delta_logit"], 0.0); self.assertFalse(out["nn"]["valid"])

    def test_network_stays_off_with_too_little_data_or_pure_noise(self):
        rng = np.random.default_rng(5)
        self.assertFalse(learner_nn.train(self._history(20, rng), self.KEYS, force=True)["valid"])
        noise = [{"flags": {k: bool(rng.random() < .5) for k in self.KEYS}, "label": int(rng.random() < .5), "weight": 1.0,
                  "feats": {"rsi": float(rng.normal(55, 10)), "is_buy": 1.0}} for _ in range(120)]
        self.assertFalse(learner_nn.train(noise, self.KEYS, force=True)["valid"])     # must NOT claim skill on noise

    def test_network_turns_on_when_there_is_real_signal(self):
        rng = np.random.default_rng(11)
        rows = []
        for _ in range(500):
            rsi = float(rng.normal(55, 14))
            f = {k: bool(rng.random() < 0.5) for k in self.KEYS}
            win = (rng.random() < 0.05) if rsi > 62 else (rng.random() < 0.75)
            rows.append({"flags": f, "label": int(win), "weight": 1.0, "feats": {"rsi": rsi, "ema20_dist_atr": 0.5, "rr": 1.7, "hour": 11.0, "is_buy": 1.0}})
        b = learner_nn.train(rows, self.KEYS, force=True)
        self.assertTrue(b["valid"], b["reason"])
        hi = learner_nn.learned_adjustment({k: True for k in self.KEYS}, {"rsi": 80.0, "ema20_dist_atr": .5, "rr": 1.7, "hour": 11.0, "is_buy": 1.0}, self.KEYS, rows)
        lo = learner_nn.learned_adjustment({k: True for k in self.KEYS}, {"rsi": 40.0, "ema20_dist_atr": .5, "rr": 1.7, "hour": 11.0, "is_buy": 1.0}, self.KEYS, rows)
        self.assertLess(hi["nn"]["p"], lo["nn"]["p"] - 0.3)
        self.assertLessEqual(abs(hi["delta_logit"]), learner_nn.NN_MAX_DELTA + learner_nn.MEM_MAX_PENALTY + 1e-9)

    def test_garbage_input_never_raises(self):
        out = learner_nn.learned_adjustment({}, {"rsi": "abc", "atr": None}, self.KEYS, self._history(40, np.random.default_rng(1)))
        self.assertIn("delta_logit", out)


class LearningLoopEndToEndTests(DbCase):
    def setUp(self):
        super().setUp()
        shadow_trades._state.update({"sb_ok": False, "sb_checked": 1e18})
        learner_nn._CACHE.update({"key": None, "ts": 0.0, "bundle": None})

    def test_repeating_a_losing_pattern_makes_the_next_similar_setup_less_likely(self):
        base = _decide()
        self.assertTrue(base["has_setup"], base.get("reason"))
        import sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        for k in range(10):                                  # ten losses of exactly this kind of setup (same factors + features)
            f = dict(base["factor_flags"])
            f["_entry_meta"] = {"logic_version": tl.LOGIC_VERSION}
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (f"2026-10-0{1 + k % 3} 10:{k:02d}:00", "BUY", 1, "CE", 1, 1, 1, 1, json.dumps(f), "LOSS"))
        c.commit(); c.close()
        learner_nn._CACHE.update({"key": None, "ts": 0.0, "bundle": None})
        again = _decide()
        if again["has_setup"]:
            self.assertLess(again["win_probability"], base["win_probability"])
            self.assertIn("lost", again["learning_note"])
        else:
            self.assertIn("No positive edge", again["reason"])          # blocked by the learned edge -- also correct

    def test_blocked_candidate_is_returned_so_it_can_be_shadow_tracked(self):
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
        cand = out["candidate"]
        self.assertEqual(cand["direction"], "BUY"); self.assertLess(cand["stop_loss"], cand["underlying_entry"] ); self.assertGreater(cand["target"], cand["underlying_entry"])
        self.assertIn("_features", cand["factor_flags"])

    def test_shadow_wins_can_lift_a_frozen_engine(self):
        """real record is bad -> engine blocks.  Followed-anyway shadow trades that WON pull the estimate back up."""
        import sqlite3
        shadow_trades._state.update({"sb_ok": False, "sb_checked": 1e18})
        c = sqlite3.connect(tl.DB_NAME)
        for i in range(30):
            fj = json.dumps(_flags(10, **{"_entry_meta": {"logic_version": tl.LOGIC_VERSION}}))
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','BUY',1,'CE',1,1,1,1,?,?)",
                      (fj, "WIN" if i % 8 == 0 else "LOSS"))
        c.commit(); c.close()
        frozen = _decide()
        self.assertFalse(frozen["has_setup"])
        for k in range(40):
            sid = shadow_trades._insert({"timestamp": f"2026-10-01 10:{k:02d}:00", "direction": "BUY", "underlying_entry": 1.0, "stop_loss": 0.5,
                                         "target": 2.0, "blocked_by": "No positive edge (EV)", "factors_json": json.dumps(_flags(10))})
            shadow_trades._update(sid, {"status": "WIN"})
        learner_nn._CACHE.update({"key": None, "ts": 0.0, "bundle": None})
        after = _decide()
        self.assertGreater(after.get("confidence_pct", 0), frozen["confidence_pct"])


if __name__ == "__main__":
    unittest.main()
