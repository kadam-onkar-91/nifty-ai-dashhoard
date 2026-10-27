import unittest
import pandas as pd
import data_confluence as dc

SEC_UP = pd.DataFrame({"Sector": list("ABCDE"), "Avg Change (%)": [0.8, 0.5, 0.4, 0.3, -0.1]})
SEC_DN = pd.DataFrame({"Sector": list("ABCDE"), "Avg Change (%)": [-0.8, -0.5, -0.4, 0.0, 0.1]})

class DC(unittest.TestCase):
    def test_all_missing_is_inert(self):
        o = dc.assess("BUY")
        self.assertEqual(o["penalty_logit"], 0.0); self.assertEqual(o["available"], 0)
    def test_support_gives_small_bonus(self):
        o = dc.assess("BUY", live_price=22770, atr=30, sector_breadth=SEC_UP, live_poc=22700, fii_footprint="BULLISH (FIIs Writing Puts)")
        self.assertLess(o["penalty_logit"], 0); self.assertGreaterEqual(o["penalty_logit"], -dc.MAX_BONUS)
        self.assertTrue(o["flags"]["sector_breadth_aligned"])
    def test_opposition_penalises_and_is_capped(self):
        o = dc.assess("BUY", live_price=22600, atr=30, sector_breadth=SEC_DN, live_poc=22700, fii_footprint="BEARISH (FIIs Writing Calls)",
                      iv_skew={"skew": 5.0}, drift_report={"status": "MODEL PERFORMANCE DRIFT DETECTED"})
        self.assertGreater(o["penalty_logit"], 0); self.assertLessEqual(o["penalty_logit"], dc.MAX_PENALTY)
        self.assertFalse(o["flags"]["model_not_drifting"])
    def test_sell_mirrors_buy(self):
        o = dc.assess("SELL", live_price=22600, atr=30, sector_breadth=SEC_DN, live_poc=22700, fii_footprint="BEARISH (x)")
        self.assertLess(o["penalty_logit"], 0)
    def test_mixed_votes_do_nothing(self):
        o = dc.assess("BUY", live_price=22770, atr=30, sector_breadth=SEC_DN, live_poc=22700)
        self.assertEqual(o["penalty_logit"], 0.0)
    def test_kill_switch(self):
        dc.ENABLED = False
        try:
            self.assertEqual(dc.assess("BUY", sector_breadth=SEC_DN)["penalty_logit"], 0.0)
        finally:
            dc.ENABLED = True
    def test_garbage_does_not_raise(self):
        o = dc.assess("BUY", sector_breadth="x", live_poc="a", fii_footprint=5, iv_skew=[1], drift_report=3)
        self.assertEqual(o["penalty_logit"], 0.0)
if __name__ == "__main__": unittest.main()
