"""
tests_v50_two_models.py -- v50: Model B (macro_confluence) + restored global veto + stricter lane + opening-range rule.
Run:  python -m unittest tests_v50_two_models -v
"""
import os, sys, unittest
from datetime import time as dtime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests_edge_v36 import DbCase, _decide  # noqa: E402  (also installs the import stubs)
import macro_confluence as mc
import ai_trade_decision as atd

GR_STRONG_BUY = {"status": "AVAILABLE", "directional_bias": "BUY", "strength": "STRONG"}
GR_STRONG_SELL = {"status": "AVAILABLE", "directional_bias": "SELL", "strength": "STRONG"}


class MacroModelTests(unittest.TestCase):
    def test_everything_with_the_trade_is_strong_aligned(self):
        r = mc.assess("BUY", GR_STRONG_BUY, 0.5, "BULLISH", "BULLISH", "BULLISH", 40, 10, "BUYING PRESSURE", "BUY", 13.0)
        self.assertEqual(r["verdict"], "STRONG_ALIGNED"); self.assertTrue(r["global_strong_aligned"]); self.assertLess(r["penalty_logit"], 0)
        self.assertEqual(r["opposed_groups"], [])

    def test_everything_against_is_strong_opposed(self):
        r = mc.assess("BUY", GR_STRONG_SELL, -0.6, "BEARISH", "BEARISH", "BEARISH", 8, 42, "SELLING PRESSURE", "SELL", 22.0)
        self.assertEqual(r["verdict"], "STRONG_OPPOSED"); self.assertTrue(r["global_strong_opposed"]); self.assertGreater(r["penalty_logit"], 0)

    def test_sell_mirrors_buy(self):
        a = mc.assess("SELL", GR_STRONG_SELL, -0.5, "BEARISH", "BEARISH", "BEARISH", 8, 42, "SELLING PRESSURE", "SELL", 21.0)
        self.assertEqual(a["verdict"], "STRONG_ALIGNED")
        b = mc.assess("SELL", GR_STRONG_BUY, 0.5, "BULLISH", "BULLISH", "BULLISH", 40, 10, "BUYING PRESSURE", "BUY", 13.0)
        self.assertEqual(b["verdict"], "STRONG_OPPOSED")

    def test_missing_data_is_neutral_not_a_crash(self):
        self.assertEqual(mc.assess("BUY")["verdict"], "NEUTRAL")
        self.assertEqual(mc.assess("BUY", global_research="junk", breadth_advances="x", vix=float("nan"))["verdict"], "NEUTRAL")
        self.assertEqual(mc.assess("HOLD")["verdict"], "NEUTRAL")

    def test_kill_switch(self):
        mc.ENABLED = False
        try:
            self.assertEqual(mc.assess("BUY", GR_STRONG_BUY, 1.0, "BULLISH", "BULLISH", "BULLISH", 40, 10)["verdict"], "NEUTRAL")
        finally:
            mc.ENABLED = True

    def test_mixed_picture_is_neutral(self):
        r = mc.assess("BUY", None, 0.05, "BULLISH", "BEARISH", "NEUTRAL", 26, 24, "", "NEUTRAL", 16.0)
        self.assertEqual(r["verdict"], "NEUTRAL")


class EngineUsesBothModels(DbCase):
    def test_opening_range_needs_strong_macro_support(self):
        out = _decide("BUY", now=dtime(9, 35), macro=True)               # fixture = everything aligned -> STRONG_ALIGNED -> allowed
        self.assertTrue(out["has_setup"], out.get("reason"))
        weak = _decide("BUY", now=dtime(9, 35), macro=False)             # Model B neutral -> opening range still forming
        self.assertFalse(weak["has_setup"]); self.assertIn("Opening range forming", weak["reason"])
        self.assertEqual(atd.classify_block(weak["reason"]), "Opening range forming")
        self.assertNotIn("Opening range forming", _decide("BUY", now=dtime(9, 50), macro=False).get("reason", ""))   # after 09:45 no hard rule

    def test_opening_hour_raises_the_bar_unless_model_b_agrees(self):
        a = _decide("BUY", now=dtime(9, 50), macro=False)
        b = _decide("BUY", now=dtime(11, 0), macro=False)
        self.assertAlmostEqual(a.get("required_ev_r", 0) - b.get("required_ev_r", 0), atd.OPENING_EV_EXTRA, places=2)
        c = _decide("BUY", now=dtime(9, 50), macro=True)
        self.assertAlmostEqual(c["required_ev_r"], b["required_ev_r"], places=2)

    def test_model_b_agreement_adds_a_little_probability_and_is_stored(self):
        on, off = _decide("BUY", macro=True), _decide("BUY", macro=False)
        self.assertGreaterEqual(on["win_probability"], off["win_probability"])
        self.assertEqual(on["macro_model"]["verdict"], "STRONG_ALIGNED")
        self.assertTrue(on["factor_flags"]["macro_model_aligned"]); self.assertFalse(off["factor_flags"]["macro_model_aligned"])
        self.assertEqual(on["factor_flags"]["_macro"]["verdict"], "STRONG_ALIGNED")

    def test_learned_penalty_is_capped_only_when_model_b_agrees(self):
        import trade_learning as tl
        orig = tl.learned_adjustment
        tl.learned_adjustment = lambda flags, feats: {"delta_logit": -1.0, "note": "test", "nn": {}, "memory": {}}
        self.addCleanup(setattr, tl, "learned_adjustment", orig)
        base = _decide("BUY", macro=True)
        self.assertTrue(base["has_setup"])
        capped = base["win_probability"]
        uncapped = _decide("BUY", macro=False)                      # Model B off -> the full -1.0 learned drag applies
        self.assertTrue(not uncapped["has_setup"] or uncapped["win_probability"] < capped - 0.02)


if __name__ == "__main__":
    unittest.main()
