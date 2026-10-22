"""v49: same-zone re-entry guard + opening-whipsaw tag (run: python -m unittest tests_v49_zone)."""
import os, sqlite3, tempfile, unittest
from datetime import datetime, timedelta

import trade_learning as tl

tl.DB_NAME = os.path.join(tempfile.mkdtemp(), "t49.db")
import trade_diagnostics as td


def _loss(direction, entry, sl, minutes_ago, now):
    tl.init_db()
    sid = tl.log_setup(direction, 22650, "PE" if direction == "SELL" else "CE", entry, sl, entry - 60 if direction == "SELL" else entry + 60, 50.0, {})
    ts = (now - timedelta(minutes=minutes_ago + 5)).strftime("%Y-%m-%d %H:%M:%S")
    xs = (now - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")
    c = sqlite3.connect(tl.DB_NAME)
    c.execute("UPDATE ai_trade_setups SET status='LOSS', exit_price=?, timestamp=?, exit_timestamp=? WHERE id=(SELECT MAX(id) FROM ai_trade_setups)", (sl, ts, xs))
    c.commit(); c.close()


class ZoneGuard(unittest.TestCase):
    def setUp(self):
        c = sqlite3.connect(tl.DB_NAME) if os.path.exists(tl.DB_NAME) else None
        if c:
            c.execute("DELETE FROM ai_trade_setups"); c.commit(); c.close()
        tl.ZONE_GUARD = True
        self.now = datetime(2026, 10, 22, 10, 0, 0)

    def test_blocks_same_zone_sell_after_stopout(self):          # the real 22-Oct case: SELL 22636.1 stopped, next SELL 22629.1
        _loss("SELL", 22636.10, 22664.10, 15, self.now)
        b, why = tl.entry_cooldown("SELL", now=self.now, price=22629.10)
        self.assertTrue(b); self.assertIn("same-zone", why)

    def test_allows_after_real_progress(self):
        _loss("SELL", 22636.10, 22664.10, 15, self.now)
        b, _ = tl.entry_cooldown("SELL", now=self.now, price=22600.0)       # 36 pts below the failed entry
        self.assertFalse(b)

    def test_buy_side_and_other_side_untouched(self):
        _loss("BUY", 22750.0, 22722.0, 15, self.now)
        self.assertTrue(tl.entry_cooldown("BUY", now=self.now, price=22752.0)[0])
        self.assertFalse(tl.entry_cooldown("SELL", now=self.now, price=22752.0)[0])

    def test_expires_and_kill_switch_and_no_price(self):
        _loss("SELL", 22636.10, 22664.10, 90, self.now)
        self.assertFalse(tl.entry_cooldown("SELL", now=self.now, price=22630.0)[0])      # older than 60 min
        _loss("SELL", 22636.10, 22664.10, 15, self.now)
        self.assertFalse(tl.entry_cooldown("SELL", now=self.now)[0])   # no price -> guard not applied
        tl.ZONE_GUARD = False
        self.assertFalse(tl.entry_cooldown("SELL", now=self.now, price=22630.0)[0])


class Tag(unittest.TestCase):
    def test_whipsaw_zone_tag(self):
        self.assertIn("opening_whipsaw", td.risk_tags({}, {"hour": 9.68}))
        self.assertIn("opening_15min", td.risk_tags({}, {"hour": 9.3}))
        self.assertNotIn("opening_whipsaw", td.risk_tags({}, {"hour": 10.5}))


if __name__ == "__main__":
    unittest.main()
