"""
tests_v55_live_dom.py -- v55: aligned live-dominant trend entries are judged by their OWN record; edge-blocked setups are shadow-tracked per gate.
Run:  python -m unittest tests_v55_live_dom -v
"""
import json, os, sqlite3, sys, unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests_edge_v36 import DbCase, _decide, _flags  # noqa: E402
from tests_v42_momentum_tick import _strong  # noqa: E402
import ai_trade_decision as atd
import trade_learning as tl
import shadow_trades as sh


def _history(n, win_every, live_dom, side="BUY"):
    c = sqlite3.connect(tl.DB_NAME)
    for i in range(n):
        fj = json.dumps(_flags(10, **{"_entry_meta": {"logic_version": tl.LOGIC_VERSION, "side": side},
                                      "_features": {"is_buy": 1.0 if side == "BUY" else 0.0},
                                      "live_dom_entry": bool(live_dom), "entry_not_extended": True}))
        c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                  "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00',?,1,'CE',1,1,1,1,?,?)",
                  (side, fj, "WIN" if (win_every and i % win_every == 0) else "LOSS"))
    c.commit(); c.close()


class LiveDomEntry(DbCase):
    def test_flag_is_set_and_old_losing_record_does_not_block(self):
        _history(40, 8, live_dom=False)                  # 12 % record of ordinary BUYs
        out = _decide("BUY", strategy=_strong("BUY", live=80, cons=95, n=5))
        self.assertTrue(out.get("live_dom_entry"), out.get("reason"))
        self.assertTrue(out["has_setup"], out.get("reason"))

    def test_kill_switch_restores_v54(self):
        _history(40, 8, live_dom=False)
        atd.LIVE_DOM_ENTRIES = False
        try:
            out = _decide("BUY", strategy=_strong("BUY", live=80, cons=95, n=5))
        finally:
            atd.LIVE_DOM_ENTRIES = True
        self.assertFalse(out["has_setup"]); self.assertIn("No positive edge", out["reason"])

    def test_own_losing_record_stops_it(self):
        _history(40, 8, live_dom=True)                   # 40 live-dominant trades, 12 % wins
        out = _decide("BUY", strategy=_strong("BUY", live=80, cons=95, n=5))
        self.assertFalse(out["has_setup"], out.get("reason"))
        self.assertIn("No positive edge", out["reason"])

    def test_not_live_dominant_is_unchanged(self):
        _history(40, 8, live_dom=False)
        out = _decide("BUY", strategy=_strong("BUY", live=20, cons=40, n=1))
        self.assertFalse(out.get("live_dom_entry"))
        self.assertFalse(out["has_setup"])


class ShadowQuota(DbCase):
    def _cand(self, d="BUY"):
        return {"direction": d, "underlying_entry": 100.0, "stop_loss": 90.0, "target": 120.0, "factor_flags": {}}

    def test_each_gate_gets_its_own_budget(self):
        now = datetime(2026, 10, 29, 10, 30)
        self.assertIsNotNone(sh.log(self._cand(), "S/R location / confirmation", now=now))
        # same direction, same minute, different gate -> still logged (was starved before v55)
        self.assertIsNotNone(sh.log(self._cand(), "No positive edge (EV)", now=now + timedelta(minutes=1)))
        # same gate again too soon -> deduped
        self.assertIsNone(sh.log(self._cand(), "No positive edge (EV)", now=now + timedelta(minutes=2)))


if __name__ == "__main__":
    unittest.main()
