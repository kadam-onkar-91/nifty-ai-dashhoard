import os, sys, tempfile, unittest
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_log


class MissedMoves(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); gate_log.DB_PATH = os.path.join(self.d, "g.db")

    def test_rally_without_trade_names_the_blocking_gate(self):
        t0 = datetime(2026, 10, 28, 10, 25)
        for i in range(10):
            gate_log.record((t0 + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S"), 22520 + 25 * i,
                            "No positive edge (EV)", direction="BUY")
        m = gate_log.missed_moves("2026-10-28")
        self.assertEqual(len(m), 1); self.assertEqual(m[0]["dir"], "BUY"); self.assertFalse(m[0]["taken"])
        self.assertEqual(m[0]["gates"][0][0], "No positive edge (EV)")

    def test_traded_move_and_small_move(self):
        t0 = datetime(2026, 10, 28, 10, 25)
        for i in range(10):
            gate_log.record((t0 + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S"), 22520 + 25 * i,
                            "TRADE TAKEN" if i == 1 else "x", direction="BUY")
        self.assertTrue(gate_log.missed_moves("2026-10-28")[0]["taken"])
        self.assertEqual(gate_log.missed_moves("2026-10-28", min_pts=500), [])


if __name__ == "__main__":
    unittest.main()
