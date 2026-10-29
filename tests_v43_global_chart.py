"""tests_v43_global_chart.py -- v43: global research can no longer hard-block, STRONG needs equity backing, coarse segments are soft.
Run:  python -m unittest tests_v43_global_chart -v
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests_edge_v36 import DbCase, _decide  # noqa: E402  (also installs the import stubs)
import pandas as pd
import global_research as gr
import trade_learning as tl


def _table(changes, **macro):
    rows = []
    names = ["S&P 500 (US)", "Nasdaq Composite (US)", "Dow Jones (US)", "Nikkei 225 (Japan)", "Shanghai Composite (China)",
             "Hang Seng (Hong Kong)", "KOSPI (South Korea)", "FTSE 100 (UK)", "DAX (Germany)", "CAC 40 (France)"]
    for n, c in zip(names, changes):
        rows.append({"Global Market / Asset": n, "Change (%)": c, "Latest Price": 100.0})
    for n, (px, ch) in macro.items():
        rows.append({"Global Market / Asset": n.replace("_", " "), "Change (%)": ch, "Latest Price": px})
    return pd.DataFrame(rows)


class GlobalStrongNeedsEquities(unittest.TestCase):
    def test_macro_only_buy_is_never_strong(self):
        # the dashboard case: 4 up / 6 down equities, but calm VIX + soft dollar + falling yields + oil down
        ch = [0.28, -0.02, 1.02, 0.95, -0.5, -0.5, 2.78, -0.29, -0.06, -0.38]
        df = _table(ch, **{"India_VIX": (12.0, 0.0), "US_Dollar_Index_(DXY)": (0, -0.6), "US_10-Year_Treasury_Yield": (0, -0.6),
                           "Crude_Oil_(WTI)": (0, -2.0)})
        out = gr.build_global_research(df)
        self.assertGreaterEqual(out["bull_points"] - out["bear_points"], 3)
        self.assertNotEqual(out["strength"], "STRONG")

    def test_real_risk_on_is_still_strong(self):
        df = _table([0.8] * 8 + [0.5, 0.4], **{"India_VIX": (12.0, 0.0), "US_Dollar_Index_(DXY)": (0, -0.6)})
        out = gr.build_global_research(df)
        self.assertEqual((out["directional_bias"], out["strength"]), ("BUY", "STRONG"))


class SoftSegments(DbCase):
    def test_coarse_segment_does_not_block_but_costs_probability(self):
        import json, sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        for i in range(12):
            fj = json.dumps({"_entry_meta": {"grade": "B", "side": "SELL", "hour_bucket": "11h", "playbook": f"P{i}", "logic_version": tl.LOGIC_VERSION}})
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','SELL',1,'PE',1,1,1,1,?,?)", (fj, "WIN" if i == 0 else "LOSS"))
        c.commit(); c.close()
        tl._SEG_CACHE["stats"] = None
        meta = {"grade": "B", "side": "SELL", "hour_bucket": "11h", "playbook": "NEWONE", "probe": False}
        self.assertEqual(tl.segment_gate(meta), (False, ""))
        pen, note = tl.segment_soft_penalty(meta)
        self.assertGreater(pen, 0.0); self.assertIn("grade=B", note)

    def test_old_logic_trades_do_not_count(self):
        import json, sqlite3
        c = sqlite3.connect(tl.DB_NAME)
        for i in range(12):
            fj = json.dumps({"_entry_meta": {"playbook": "OLD", "logic_version": "v20_old"}})
            c.execute("INSERT INTO ai_trade_setups (timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
                      "confidence_pct,factors_json,status) VALUES ('2026-01-01 10:00:00','SELL',1,'PE',1,1,1,1,?,'LOSS')", (fj,))
        c.commit(); c.close()
        tl._SEG_CACHE["stats"] = None
        self.assertEqual(tl.segment_gate({"playbook": "OLD", "probe": False}), (False, ""))


class TodaysCase(DbCase):
    def test_21_oct_macro_only_points_can_no_longer_make_a_false_strong_global_read(self):
        """The 21-Oct 'STRONG BUY' came from VIX/DXY/yields/crude while most world indices were red (global_research.py fix, kept in v50).
        Because the v29 veto is back (v50), that fix is what keeps a falling market tradable."""
        import global_research as gr
        import inspect
        src = inspect.getsource(gr)
        self.assertIn("equity_backs_buy", src); self.assertIn("equity_backs_sell", src)

    def test_a_losing_record_is_not_bypassed_even_when_the_veto_steps_aside(self):
        from tests_v42_momentum_tick import _lose_a_lot, _strong
        _lose_a_lot(8)
        g = {"status": "AVAILABLE", "directional_bias": "BUY", "strength": "STRONG"}
        out = _decide("SELL", strategy=_strong("SELL", live=55), global_research=g, macro=True)
        self.assertFalse(out["has_setup"])

    def test_veto_steps_aside_when_live_market_confirms_the_move(self):
        """v50b: the 21-Oct case.  Every strategy SELL, live price action -55, global says STRONG BUY: no longer blocked all day (penalty instead)."""
        from tests_v42_momentum_tick import _strong
        g = {"status": "AVAILABLE", "directional_bias": "BUY", "strength": "STRONG"}
        out = _decide("SELL", strategy=_strong("SELL", live=55), global_research=g, macro=True)
        self.assertTrue(out["has_setup"], out.get("reason"))
        self.assertIn("live price action", out["confidence_note"])
        weak = _decide("SELL", strategy=_strong("SELL", live=20), global_research=g, macro=True)
        self.assertFalse(weak["has_setup"]); self.assertIn("Global research is STRONGLY BUY", weak["reason"])

    def test_weak_move_with_strong_global_buy_is_still_not_forced(self):
        from tests_v42_momentum_tick import _lose_a_lot, _strong
        _lose_a_lot(40)
        g = {"status": "AVAILABLE", "directional_bias": "BUY", "strength": "STRONG"}
        out = _decide("SELL", strategy=_strong("SELL", live=20), global_research=g)
        self.assertFalse(out["has_setup"])


if __name__ == "__main__":
    unittest.main()
