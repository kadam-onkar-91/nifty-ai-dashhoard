import unittest
import numpy as np, pandas as pd
import trade_postmortem as tpm

def _df():
    idx = pd.date_range("2026-10-27 09:15", periods=40, freq="5min")
    c = np.concatenate([np.full(15, 22800.0), np.linspace(22800, 22650, 25)])
    return pd.DataFrame({"Open": c, "High": c + 3, "Low": c - 3, "Close": c}, index=idx), idx

class Loc(unittest.TestCase):
    def test_sold_bottom(self):
        df, idx = _df()
        self.assertEqual(tpm.location_mistake(df, idx[-1] + pd.Timedelta(minutes=1), "SELL", 22650, 20), "SOLD_THE_BOTTOM")
    def test_buy_at_bottom_is_fine(self):
        df, idx = _df()
        self.assertIsNone(tpm.location_mistake(df, idx[-1] + pd.Timedelta(minutes=1), "BUY", 22650, 20))
    def test_amplifier_tightens_and_is_bounded(self):
        mk = lambda n: [{"key": str(i), "status": "LOSS", "loc_mistake": "SOLD_THE_BOTTOM", "decided": True} for i in range(n)]
        old = tpm.decided; tpm.decided = lambda x: x
        try:
            self.assertEqual(tpm.exhaustion_threshold(mk(0)), 2.5)
            self.assertEqual(tpm.exhaustion_threshold(mk(1)), 2.5)
            self.assertLess(tpm.exhaustion_threshold(mk(2)), 2.5)
            self.assertEqual(tpm.exhaustion_threshold(mk(6)), 1.8)
        finally:
            tpm.decided = old
if __name__ == "__main__": unittest.main()
