"""
tests_v42_momentum_tick.py -- v42: momentum-capture lane, learned-drag caps, tick-level SL/target, OI target picker, SL ceiling, search brief.
Run:  python -m unittest tests_v42_momentum_tick -v
"""
import json, os, sqlite3, sys, unittest
from datetime import datetime, time as dtime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests_edge_v36 import DbCase, _decide, _flags, _strategy, FIXED_NOW  # noqa: E402  (also installs the import stubs)
import pandas as pd
import ai_trade_decision as atd
import trade_learning as tl


def _lose_a_lot(n=40):
    """A real 14% win rate -> the calibrated edge model refuses ordinary setups (as designed)."""
    c = sqlite3.connect(tl.DB_NAME)
    for i in range(n):
        fj = json.dumps(_flags(10, **{"_entry_meta": {"logic_version": tl.LOGIC_VERSION}, "live_dom_entry": True}))
        c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                  "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','BUY',1,'CE',1,1,1,1,?,?)",
                  (fj, "WIN" if i % 7 == 0 else "LOSS"))
    c.commit(); c.close()


def _strong(direction, live=50.0, cons=95.0, n=4):
    s = _strategy(direction, 64.0)
    sg = 1 if direction == "BUY" else -1
    s["live_state"] = {"score": sg * live}
    s["buy_consensus"] = cons if direction == "BUY" else 0.0
    s["sell_consensus"] = cons if direction == "SELL" else 0.0
    s["buy_live_count"] = n if direction == "BUY" else 0
    s["sell_live_count"] = n if direction == "SELL" else 0
    return s


class MomentumLaneTests(DbCase):
    def test_lane_needs_both_models_and_cannot_rescue_a_really_losing_record(self):
        """v50: the lane used to waive the whole learned bar.  Now it needs Model B to agree, a win chance >= LANE_MIN_WIN_PROB and
        an EV shortfall <= LANE_MAX_EV_SHORTFALL."""
        _lose_a_lot()                                                     # 1 win in 7: honest win chance ~28-30 %
        plain = _decide("SELL")
        self.assertFalse(plain["has_setup"]); self.assertIn("No positive edge", plain["reason"])
        for d in ("SELL", "BUY"):
            out = _decide(d, strategy=_strong(d, live=80), macro=True)
            self.assertFalse(out["has_setup"], d)                         # a 14 % record is not overridden any more

    def test_lane_waives_a_small_shortfall_only_when_model_b_agrees(self):
        c = sqlite3.connect(tl.DB_NAME)
        for i in range(40):                                               # 1 win in 4 -> chance ~38-41 %, EV just under the bar
            fj = json.dumps(_flags(10, **{"_entry_meta": {"logic_version": tl.LOGIC_VERSION}, "live_dom_entry": True}))
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','BUY',1,'CE',1,1,1,1,?,?)",
                      (fj, "WIN" if i % 4 == 0 else "LOSS"))
        c.commit(); c.close()
        self.assertFalse(_decide("SELL")["has_setup"])
        old = atd.LANE_MAX_EV_SHORTFALL
        try:
            self.assertFalse(_decide("SELL", strategy=_strong("SELL", live=80), macro=True)["has_setup"])   # shortfall 0.12 > 0.10
            atd.LANE_MAX_EV_SHORTFALL = 0.15
            out = _decide("SELL", strategy=_strong("SELL", live=80), macro=True)
            self.assertTrue(out["has_setup"], out.get("reason")); self.assertTrue(out["momentum_lane"])
            self.assertIn("Momentum-capture lane", out["confidence_note"])
            self.assertEqual(out["factor_flags"]["_entry_meta"]["lane"], "momentum")
            self.assertFalse(_decide("SELL", strategy=_strong("SELL", live=80), macro=False)["has_setup"])  # Model B not agreeing -> no lane
        finally:
            atd.LANE_MAX_EV_SHORTFALL = old

    def test_lane_needs_strong_live_evidence(self):
        _lose_a_lot()
        for kw in (dict(live=30), dict(cons=60), dict(n=2)):
            out = _decide("SELL", strategy=_strong("SELL", **dict(dict(live=80), **kw)), macro=True)
            self.assertFalse(out["has_setup"], kw)

    def test_lane_does_not_override_a_severe_chase(self):
        c = sqlite3.connect(tl.DB_NAME)
        for i in range(40):
            fj = json.dumps(_flags(10, **{"_entry_meta": {"logic_version": tl.LOGIC_VERSION}, "live_dom_entry": True}))
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','BUY',1,'CE',1,1,1,1,?,?)",
                      (fj, "WIN" if i % 4 == 0 else "LOSS"))
        c.commit(); c.close()
        self.addCleanup(setattr, atd, "LANE_MAX_EV_SHORTFALL", atd.LANE_MAX_EV_SHORTFALL)
        atd.LANE_MAX_EV_SHORTFALL = 0.15
        eq = {"penalty_logit": 0.6, "chase_flags": ["RSI 80", "far from EMA", "day move"], "confirm_flags": [], "severe": True, "metrics": {}}
        out = _decide("SELL", strategy=_strong("SELL", live=80), macro=True)
        self.assertTrue(out["has_setup"], out.get("reason"))
        # same setup, but entry-quality says severe chase -> lane off -> the learned bar vetoes again
        import ai_trade_decision as m
        orig = m.generate_trade_decision
        def wrapped(*a, **k):
            k["entry_quality"] = eq
            return orig(*a, **k)
        m.generate_trade_decision = wrapped
        try:
            out2 = _decide("SELL", strategy=_strong("SELL", live=80), macro=True)
        finally:
            m.generate_trade_decision = orig
        self.assertFalse(out2["has_setup"])

    def test_local_fallback_accepts_momentum_lane(self):
        d = {"direction": "SELL", "expectancy_r": -0.6, "critical_confirmations": 6, "major_conflicts": 0, "risk_reward": 1.6, "momentum_lane": True}
        self.assertTrue(atd.local_fallback_review(d)["approved"])
        d["momentum_lane"] = False
        self.assertFalse(atd.local_fallback_review(d)["approved"])

    def test_learned_drag_is_capped(self):
        self.assertLessEqual(atd.LEARNED_DRAG_CAP, 0.30); self.assertLessEqual(atd.TAG_PENALTY_CAP, 0.20)


class StopCeilingTests(DbCase):
    def test_stop_is_never_wider_than_28_points(self):
        for atr in (14.0, 40.0, 80.0):
            for d in ("BUY", "SELL"):
                out = _decide(d, atr=atr, zone_edge=60.0)
                if out["has_setup"]:
                    self.assertLessEqual(abs(out["underlying_entry"] - out["stop_loss"]), 28.0 + 1e-6, (atr, d))


class OiPickerTests(DbCase):
    def _chain(self, rows):
        return pd.DataFrame(rows, columns=["Strike", "Call OI", "Put OI"])

    def test_target_goes_in_front_of_the_first_wall_that_pays(self):
        # BUY at 24000, SL 20: wall 24015 is too close (0.75R), wall 24060 pays -> target just before 24060
        chain = self._chain([[24015, 900, 5], [24060, 1000, 5]])
        t, wall, blocking, rr = atd._oi_pick_target(chain, 24000.0, "BUY", 20.0, atr=14.0)
        self.assertEqual(wall, 24060.0); self.assertEqual(blocking, 24015.0)
        self.assertLess(t, 24060.0); self.assertGreaterEqual(rr, atd.OI_MIN_RR)

    def test_sell_uses_put_walls_below(self):
        chain = self._chain([[23940, 5, 1000]])
        t, wall, blocking, rr = atd._oi_pick_target(chain, 24000.0, "SELL", 20.0, atr=14.0)
        self.assertEqual(wall, 23940.0); self.assertGreater(t, 23940.0); self.assertIsNone(blocking)

    def test_nothing_pays_returns_blocking_wall_only(self):
        chain = self._chain([[24014, 1000, 5]])
        t, wall, blocking, rr = atd._oi_pick_target(chain, 24000.0, "BUY", 20.0, atr=14.0)
        self.assertIsNone(t); self.assertEqual(blocking, 24014.0)

    def test_decision_target_never_beyond_wall_that_pays(self):
        chain = self._chain([[24050, 1000, 5]])
        out = _decide("BUY", chain=chain)
        self.assertTrue(out["has_setup"], out.get("reason"))
        self.assertLessEqual(out["target"], 24050.0)


class TickResolveTests(DbCase):
    def _open(self, direction="BUY", entry=24000.0, sl=23980.0, tp=24030.0):
        return tl.log_setup(direction, 24000, "CE" if direction == "BUY" else "PE", entry, sl, tp, 55.0, _flags(8, **{"_entry_meta": {"playbook": "X"}}))

    def _status(self, sid):
        c = sqlite3.connect(tl.DB_NAME)
        r = c.execute("SELECT status, exit_price FROM ai_trade_setups WHERE id=?", (sid,)).fetchone(); c.close()
        return r

    def test_buy_stop_hits_on_the_tick_without_any_candle(self):
        sid = self._open("BUY")
        tl.resolve_tick(23990.0); self.assertEqual(self._status(sid)[0], "OPEN")
        tl.resolve_tick(23980.0); st, px = self._status(sid)
        self.assertEqual(st, "LOSS"); self.assertEqual(px, 23980.0)

    def test_sell_stop_and_target_on_the_tick(self):
        sid = self._open("SELL", sl=24020.0, tp=23970.0)
        tl.resolve_tick(24019.0); self.assertEqual(self._status(sid)[0], "OPEN")
        tl.resolve_tick(24020.5); self.assertEqual(self._status(sid)[0], "LOSS")
        sid2 = self._open("SELL", entry=24000.0, sl=24020.0, tp=23970.0)
        tl.resolve_tick(23969.0); self.assertEqual(self._status(sid2)[0], "WIN")

    def test_tick_path_never_applies_time_exits(self):
        sid = self._open("BUY")
        tl._now_ist = lambda: datetime(2026, 10, 5, 15, 25)       # after EOD square-off
        tl.resolve_tick(24005.0)
        self.assertEqual(self._status(sid)[0], "OPEN")             # only the heavy path squares off
        tl.resolve_open_setups(24005.0, df=None)
        self.assertEqual(self._status(sid)[0], "EXPIRED")

    def test_tick_guard_module_loads_and_is_idempotent(self):
        import tick_guard
        self.assertTrue(callable(tick_guard.ensure_started)); self.assertIn("alive", tick_guard.status())


class NewsBriefTests(unittest.TestCase):
    def test_brief_is_cached_and_failure_is_reported_not_raised(self):
        atd._NEWS.update({"ts": 0.0, "text": "", "grounded": False, "model": None, "err": "", "fail_ts": 0.0, "busy": False})
        out = atd.get_news_brief([])                 # no keys -> unavailable, must not raise
        self.assertEqual(out["text"], ""); self.assertTrue(out["err"])
        atd._NEWS.update({"ts": __import__("time").time(), "text": "brief", "grounded": True, "model": "m", "fail_ts": 0.0})
        hit = atd.get_news_brief(["k"])
        self.assertEqual(hit["text"], "brief"); self.assertTrue(hit["grounded"])


if __name__ == "__main__":
    unittest.main()
