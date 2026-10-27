import unittest
import numpy as np, pandas as pd
import strategy_engine as se


def _df(closes, last=None):
    rows = []
    prev = closes[0]
    for c in closes:
        o = prev; h = max(o, c) + 3; l = min(o, c) - 3
        rows.append([o, h, l, c]); prev = c
    if last:
        rows[-1] = list(last)
    d = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])
    d["ATR"] = 20.0
    d["EMA9"] = d["Close"].ewm(span=9).mean()
    d["EMA20"] = d["Close"].ewm(span=20).mean()
    d["VWAP"] = d["Close"].expanding().mean()
    d["SUPERTREND_DIR"] = -1
    return d

FALL = list(np.linspace(22800, 22800, 20)) + list(np.linspace(22800, 22650, 25))


class Spring(unittest.TestCase):
    def test_spring_buy_after_fall(self):
        d = _df(FALL + [22642, 22665])
        # sweep candle: low below swing low, long lower wick, closes up
        d.iloc[-1] = [22648, 22672, 22628, 22668, 20, d["EMA9"].iloc[-1], 0, 0, 0] if False else d.iloc[-1]
        d.loc[d.index[-1], ["Open", "High", "Low", "Close"]] = [22648, 22672, 22626, 22668]
        self.assertEqual(se._exhaustion_reversal(d), 1)

    def test_no_signal_in_flat_market(self):
        d = _df([22700 + (i % 3) for i in range(60)])
        self.assertEqual(se._exhaustion_reversal(d), 0)

    def test_clean_breakdown_not_a_spring(self):
        d = _df(FALL + [22640, 22615])      # keeps falling, red candle, no rejection
        self.assertEqual(se._exhaustion_reversal(d), 0)

    def test_sell_bottom_guard(self):
        live = {"ok": True, "score": -10, "fall24_atr": 3.1, "rise24_atr": 0.3, "off_high24_atr": 3.1, "bottom_reject": True,
                "top_reject": False, "drop_atr": 3.1, "bounce_atr": 0.3}
        ok, _, note = se._signal_validity(-1, live, 0, 50)
        self.assertFalse(ok); self.assertIn("bottom", note)

    def test_clean_breakdown_sell_passes_guard(self):
        live = {"ok": True, "score": -50, "fall24_atr": 3.1, "rise24_atr": 0.3, "off_high24_atr": 3.1, "bottom_reject": False,
                "top_reject": False, "drop_atr": 3.1, "bounce_atr": 0.3}
        ok, _, _ = se._signal_validity(-1, live, 0, 50)
        self.assertTrue(ok)

    def test_buy_top_guard_and_kill_switch(self):
        live = {"ok": True, "score": 20, "fall24_atr": 0.3, "rise24_atr": 3.2, "off_high24_atr": 0.3, "off_low24_atr": 3.2,
                "top_reject": True, "bottom_reject": False, "drop_atr": 0.3, "bounce_atr": 3.2}
        self.assertFalse(se._signal_validity(1, live, 0, 50)[0])
        se.EXHAUSTION_GUARD = False
        try:
            self.assertTrue(se._signal_validity(1, live, 0, 50)[0])
        finally:
            se.EXHAUSTION_GUARD = True

    def test_registered(self):
        self.assertIn("spring_reclaim_reversal", se.STRATEGY_META)

if __name__ == "__main__":
    unittest.main()
