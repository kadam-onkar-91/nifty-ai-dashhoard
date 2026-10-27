import unittest, strategy_engine as se

LIVE = {"ok": True, "score": 34, "drop_atr": 1.37, "bounce_atr": 3.1, "reclaim_up": True, "vwap_up": True,
        "reclaim_dn": False, "vwap_dn": False}

class Reclaim(unittest.TestCase):
    def test_buy_allowed_after_v_recovery(self):
        ok, sc, note = se._signal_validity(1, LIVE, 0, 44.8)
        self.assertTrue(ok); self.assertIn("reclaim", note)
    def test_buy_still_blocked_without_reclaim(self):
        ok, _, _ = se._signal_validity(1, dict(LIVE, reclaim_up=False), 0, 44.8)
        self.assertFalse(ok)
    def test_sell_against_live_up_still_blocked(self):
        ok, _, _ = se._signal_validity(-1, LIVE, 0, 33.4)
        self.assertFalse(ok)
    def test_kill_switch(self):
        se.RECLAIM_EXCEPTION = False
        try:
            ok, _, _ = se._signal_validity(1, LIVE, 0, 44.8)
            self.assertFalse(ok)
        finally:
            se.RECLAIM_EXCEPTION = True
    def test_sell_rollover_allowed(self):
        live = {"ok": True, "score": -30, "drop_atr": 3.0, "bounce_atr": 1.4, "reclaim_up": False, "vwap_up": False,
                "reclaim_dn": True, "vwap_dn": True}
        ok, _, note = se._signal_validity(-1, live, 0, 40)
        self.assertTrue(ok)
if __name__ == "__main__": unittest.main()
