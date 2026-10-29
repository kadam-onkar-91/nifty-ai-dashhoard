"""
tests_v54_reaction.py -- v54: turning-point REACTION entries (resistance-rejection SELL / support-bounce BUY).
Run:  python -m unittest tests_v54_reaction -v
"""
import json, os, sqlite3, sys, unittest
from datetime import time as dtime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests_edge_v36 import DbCase, _decide, _flags  # noqa: E402  (also installs the import stubs)
from tests_v42_momentum_tick import _strong  # noqa: E402
import numpy as np
import pandas as pd
import strategy_engine as se
import ai_trade_decision as atd
import trade_learning as tl

TOP = {"ok": True, "score": 70.0, "rise24_atr": 2.6, "fall24_atr": 0.4, "off_high24_atr": 0.4, "off_low24_atr": 2.6,
       "top_reject": True, "bottom_reject": False, "drop_atr": 0.4, "bounce_atr": 2.6}
BOTTOM = {"ok": True, "score": -70.0, "rise24_atr": 0.4, "fall24_atr": 2.6, "off_high24_atr": 2.6, "off_low24_atr": 0.4,
          "top_reject": False, "bottom_reject": True, "drop_atr": 2.6, "bounce_atr": 0.4}


class StrategyLayer(unittest.TestCase):
    def test_confirmed_rejection_at_the_extreme_is_a_reaction(self):
        self.assertTrue(se.reaction_confirmed(-1, TOP))
        self.assertTrue(se.reaction_confirmed(1, BOTTOM))

    def test_not_a_reaction_without_rejection_or_far_from_the_extreme_or_wrong_side(self):
        self.assertFalse(se.reaction_confirmed(-1, dict(TOP, top_reject=False)))
        self.assertFalse(se.reaction_confirmed(-1, dict(TOP, fall24_atr=1.8)))        # already well off the high: that is a continuation, not a turn
        self.assertFalse(se.reaction_confirmed(-1, dict(TOP, rise24_atr=0.8)))        # no real move to reject
        self.assertFalse(se.reaction_confirmed(1, TOP))
        self.assertFalse(se.reaction_confirmed(-1, BOTTOM))
        self.assertFalse(se.reaction_confirmed(0, TOP)); self.assertFalse(se.reaction_confirmed(-1, {}))

    def test_reversal_strategy_passes_the_stale_block_at_a_turn_others_do_not(self):
        ok, score, note = se._signal_validity(-1, TOP, 0, 45.0, name="spring_reclaim_reversal")
        self.assertTrue(ok); self.assertIn("reaction", note); self.assertEqual(score, 45.0)
        ok_old, _, why = se._signal_validity(-1, TOP, 0, 45.0)                         # v53 behaviour (no name)
        self.assertFalse(ok_old); self.assertIn("stale", why)
        self.assertFalse(se._signal_validity(-1, TOP, 0, 45.0, name="range_expansion_breakout")[0])   # a continuation strategy is still blocked
        self.assertFalse(se._signal_validity(-1, dict(TOP, top_reject=False), 0, 45.0, name="spring_reclaim_reversal")[0])
        ok_b, _, _ = se._signal_validity(1, BOTTOM, 0, 45.0, name="vwap_sr_rejection")
        self.assertTrue(ok_b)

    def test_kill_switch_restores_v53(self):
        se.REACTION_ENTRIES = False
        try:
            self.assertFalse(se.reaction_confirmed(-1, TOP))
            self.assertFalse(se._signal_validity(-1, TOP, 0, 45.0, name="spring_reclaim_reversal")[0])
        finally:
            se.REACTION_ENTRIES = True

    def test_exhaustion_guard_still_blocks_selling_the_bottom(self):
        live = dict(BOTTOM, score=-50.0)
        ok, _, note = se._signal_validity(-1, live, 0, 50.0, name="range_expansion_breakout")
        self.assertFalse(ok); self.assertIn("bottom", note)

    def test_analyze_exposes_reaction_fields(self):
        rng = np.random.default_rng(3)
        idx = pd.date_range("2026-10-22 09:15", periods=300, freq="5min")
        c = 22700 + np.cumsum(rng.normal(0, 4, 300))
        df = pd.DataFrame({"Open": np.r_[c[0], c[:-1]], "High": c + 3, "Low": c - 3, "Close": c, "Volume": 0.0}, index=idx)
        df.attrs["market_source"] = "UPSTOX_LIVE"
        r = se.analyze(df)
        self.assertIn("reaction_selected", r); self.assertIn("reaction_strategies", r)
        for a in r.get("ranked_active") or []:
            self.assertIn("reaction", a)


def _spike_frame(mirror=False, wick=True):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2026-10-23 09:15", periods=75, freq="5min")
    c = 22700 + np.cumsum(rng.normal(0, 3, 75))
    t = pd.date_range("2026-10-27 09:15", periods=40, freq="5min")
    path = np.array((list(np.linspace(c[-1], 22735, 14)) + list(np.linspace(22735, 22752, 10)) + list(np.linspace(22752, 22790, 12)) + [22812, 22818, 22800, 22788])[:40])
    O = np.r_[c[-1], path[:-1]]; C = path.copy(); H = np.maximum(O, C) + 3; L = np.minimum(O, C) - 3
    if wick:
        H[37] = 22824; H[38] = 22826; C[38] = 22806; O[38] = 22814
    else:
        C[38] = 22822; O[38] = 22818; H[38] = 22824                      # keeps running: no rejection
    fO = np.r_[np.r_[c[0], c[:-1]], O]; fC = np.r_[c, C]
    fH = np.r_[np.r_[c[0], c[:-1]] + 3, H]; fL = np.r_[np.r_[c[0], c[:-1]] - 3, L]
    d = pd.DataFrame({"Open": fO, "High": np.maximum.reduce([fH, fO, fC]), "Low": np.minimum.reduce([fL, fO, fC]), "Close": fC, "Volume": 0.0}, index=idx.append(t))
    if mirror:
        m = 2 * 22750.0
        d = pd.DataFrame({"Open": m - d["Open"], "High": m - d["Low"], "Low": m - d["High"], "Close": m - d["Close"], "Volume": 0.0}, index=d.index)
    d.attrs["market_source"] = "UPSTOX_LIVE"
    return d


class BlowOff(unittest.TestCase):
    def test_spike_top_with_rejection_is_an_upthrust_and_a_reaction(self):
        d = _spike_frame().iloc[:75 + 39]
        self.assertEqual(se._exhaustion_reversal(se._prepare(d)), -1)
        r = se.analyze(d)
        self.assertEqual(r["direction"], "SELL"); self.assertTrue(r["reaction_selected"]); self.assertIn("spring_reclaim_reversal", r["reaction_strategies"])
        buys = [a for a in r["ranked_active"] if a["direction"] == "BUY"]
        self.assertTrue(buys and all(not a["live_valid"] for a in buys))          # continuation BUYs at the spike top are 'buying the top'

    def test_capitulation_low_with_rejection_is_a_spring_buy(self):
        d = _spike_frame(mirror=True).iloc[:75 + 39]
        self.assertEqual(se._exhaustion_reversal(se._prepare(d)), 1)
        r = se.analyze(d)
        self.assertEqual(r["direction"], "BUY"); self.assertTrue(r["reaction_selected"])

    def test_a_run_that_keeps_going_is_not_a_reversal(self):
        d = _spike_frame(wick=False).iloc[:75 + 39]
        self.assertEqual(se._exhaustion_reversal(se._prepare(d)), 0)
        self.assertFalse(se.analyze(d)["reaction_selected"])

    def test_blowoff_kill_switch(self):
        se.EXH_BLOWOFF = False
        try:
            self.assertEqual(se._exhaustion_reversal(se._prepare(_spike_frame().iloc[:75 + 39])), 0)
        finally:
            se.EXH_BLOWOFF = True


def _reaction_strategy(direction="SELL", reaction=True):
    s = _strong(direction, live=-30 if False else 50)
    s["reaction_selected"] = bool(reaction)
    s["reaction_strategies"] = ["spring_reclaim_reversal"] if reaction else []
    return s


def _decide_vs_ai(direction, ai_vote, strategy, location=None):
    """Run the engine with the AI research vote forced to `ai_vote` (+1 BUY / -1 SELL) and optionally another S/R location."""
    from tests_edge_v36 import _decide_inner  # noqa: F401
    real = atd.generate_trade_decision

    def patched(*a, **k):
        k["signal_code"] = ai_vote
        if location and isinstance(k.get("sr_context"), dict):
            k["sr_context"] = dict(k["sr_context"], location=location)
        return real(*a, **k)
    atd.generate_trade_decision = patched
    try:
        return _decide(direction, strategy=strategy)
    finally:
        atd.generate_trade_decision = real


class DecisionLayer(DbCase):
    def test_ai_vote_lag_blocks_an_ordinary_entry_but_not_a_confirmed_reaction_at_the_zone(self):
        plain = _decide_vs_ai("SELL", +1, _reaction_strategy("SELL", reaction=False))
        self.assertFalse(plain["has_setup"]); self.assertIn("AI research is BUY", plain["reason"])
        out = _decide_vs_ai("SELL", +1, _reaction_strategy("SELL", reaction=True))
        self.assertNotIn("AI research is", out.get("reason", ""), out.get("reason"))
        if out["has_setup"]:
            self.assertTrue(out["reaction_entry"]); self.assertTrue(out["factor_flags"]["reaction_entry"])
            self.assertEqual(out["factor_flags"]["_entry_meta"]["lane"], "reaction")
            self.assertIn("reaction_override", out["ai_research_source"])
            self.assertLessEqual(abs(out["stop_loss"] - out["underlying_entry"]), atd.MAX_SL_PTS + 1e-6)

    def test_reaction_needs_the_matching_zone(self):
        out = _decide_vs_ai("SELL", +1, _reaction_strategy("SELL", reaction=True), location="AT_SUPPORT")   # selling INTO support is not a rejection
        self.assertFalse(out["has_setup"]); self.assertIn("AI research is BUY", out["reason"])
        buy = _decide_vs_ai("BUY", -1, _reaction_strategy("BUY", reaction=True), location="AT_RESISTANCE")
        self.assertFalse(buy["has_setup"])

    def test_kill_switch(self):
        atd.REACTION_ENTRIES = False
        try:
            out = _decide_vs_ai("SELL", +1, _reaction_strategy("SELL", reaction=True))
        finally:
            atd.REACTION_ENTRIES = True
        self.assertFalse(out["has_setup"]); self.assertIn("AI research is BUY", out["reason"])

    def _history(self, n, win_every, reaction):
        c = sqlite3.connect(tl.DB_NAME)
        for i in range(n):
            fj = json.dumps(_flags(10, **{"_entry_meta": {"logic_version": tl.LOGIC_VERSION, "side": "SELL"},
                                          "_features": {"is_buy": 0.0}, "reaction_entry": bool(reaction), "live_dom_entry": True, "entry_not_extended": False}))
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','SELL',1,'PE',1,1,1,1,?,?)",
                      (fj, "WIN" if (win_every and i % win_every == 0) else "LOSS"))
        c.commit(); c.close()

    def test_reaction_is_not_charged_for_the_losses_of_late_chases(self):
        self._history(40, 8, reaction=False)                     # 12 % record, all ordinary (non-reaction) SELLs
        plain = _decide("SELL", strategy=_strong("SELL", live=50))
        react = _decide_vs_ai("SELL", -1, _reaction_strategy("SELL", reaction=True))
        pw = plain.get("win_probability") or (plain.get("candidate") or {}).get("factor_flags", {}).get("_features", {}).get("win_prob")
        self.assertFalse(plain["has_setup"])                     # the ordinary SELL is still refused by the real record
        self.assertGreater(float(react.get("win_probability") or 0.0), 0.0, react.get("reason"))
        self.assertTrue(react["has_setup"], react.get("reason"))

    def test_reaction_with_its_own_losing_record_is_stopped(self):
        self._history(40, 8, reaction=True)                      # 40 REACTION trades, 12 % wins -> its own record is bad
        out = _decide_vs_ai("SELL", -1, _reaction_strategy("SELL", reaction=True))
        self.assertFalse(out["has_setup"], out.get("reason"))
        self.assertIn("No positive edge", out["reason"])

    def test_entry_type_helper(self):
        self.assertEqual(tl.entry_type_of({"reaction_entry": True, "entry_not_extended": False}), "reaction")
        self.assertEqual(tl.entry_type_of({"entry_not_extended": False}), "extended")
        self.assertEqual(tl.entry_type_of({"entry_not_extended": True}), "clean")
        self.assertIsNone(tl.entry_type_of({"x": 1})); self.assertIsNone(tl.entry_type_of(None))

    def test_refusal_explains_where_the_probability_went(self):
        self._history(40, 8, reaction=False)
        out = _decide("SELL", strategy=_strong("SELL", live=50))
        self.assertFalse(out["has_setup"])
        self.assertIn("Probability drags", out["reason"]); self.assertIn("penalty_breakdown", out)
        self.assertEqual(atd.classify_block(out["reason"]), "No positive edge (EV)")


def _decide_conflicting(direction, strategy):
    """Regime + MTF (two lagging reads) argue AGAINST the trade, as right after a V-reversal."""
    real = atd.generate_trade_decision

    def patched(*a, **k):
        ctx = dict(k["dashboard_context"])
        bull = direction == "BUY"
        ctx["RAW Regime Engine"] = "BEARISH" if bull else "BULLISH"
        ctx["RAW Multi-Timeframe Structure"] = "bearish bearish bearish" if bull else "bullish bullish bullish"
        k["dashboard_context"] = ctx
        return real(*a, **k)
    atd.generate_trade_decision = patched
    try:
        return _decide(direction, strategy=strategy)
    finally:
        atd.generate_trade_decision = real


class ConflictRelief(DbCase):
    def test_two_lagging_conflicts_do_not_veto_a_live_dominant_entry(self):
        out = _decide_conflicting("BUY", _strong("BUY", live=80, cons=95, n=5))
        self.assertNotIn("confluence is not strong", out.get("reason", ""), out.get("reason"))

    def test_without_live_dominance_two_conflicts_still_veto(self):
        out = _decide_conflicting("BUY", _strong("BUY", live=20, cons=40, n=1))
        self.assertFalse(out["has_setup"]); self.assertIn("confluence is not strong", out["reason"])

    def test_kill_switch(self):
        atd.CONFLICT_RELIEF = False
        try:
            out = _decide_conflicting("BUY", _strong("BUY", live=80, cons=95, n=5))
        finally:
            atd.CONFLICT_RELIEF = True
        self.assertFalse(out["has_setup"]); self.assertIn("confluence is not strong", out["reason"])


if __name__ == "__main__":
    unittest.main()
