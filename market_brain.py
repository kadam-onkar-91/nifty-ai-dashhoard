"""
market_brain.py -- one call from app.py:  market_brain.run(st, df, namespace)   (v46, purely additive, never raises)

Every refresh it (cheaply, once per new 5-min candle for the heavy parts):
  1. records the market (market_memory)            -> data for learning, back-filled on day one
  2. re-trains / re-validates the forecaster        -> market_learner
  3. runs the autopsy for every day in the candles  -> market_autopsy (and stores finished days permanently)
  4. draws the "Market Brain" panel.
It is ADVISORY: it never opens, blocks or edits a trade.
"""
from __future__ import annotations

import bisect
import logging

import pandas as pd

import market_adaptive as mad
import market_autopsy as ma
import market_learner as ml
import market_memory as mm
import trade_learning as tl
import trade_postmortem as tpm

logger = logging.getLogger(__name__)


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def _fc_compact(fc, ts):
    return {"ts": ts, "p_up25": fc.get("p_up25"), "p_up50": fc.get("p_up50"), "p_move50": fc.get("p_move50"),
            "trusted": bool(fc.get("trusted")), "lean": fc.get("lean"), "model": fc.get("model_used")}


def _trade_forecasts(snaps, trades):
    """Forecast that was stored at (just before) each engine trade's entry -> rows for the table."""
    withfc = [s for s in snaps if s.get("fc") and s["fc"].get("p_up25") is not None]
    keys = [s["ts"] for s in withfc]
    rows = []
    for t in trades:
        ts = str(t.get("timestamp") or "")[:19]
        i = bisect.bisect_right(keys, ts) - 1
        if i < 0:
            continue
        s = withfc[i]
        if (pd.Timestamp(ts) - pd.Timestamp(s["ts"])).total_seconds() > 600:
            continue
        v = ml.trade_view(dict(s["fc"], ready=True), t.get("direction"))
        rows.append({"Kab": ts[5:16], "Trade": f"{t.get('direction')} @ {float(t.get('underlying_entry') or 0):,.0f}", "Natija": t.get("status"),
                     "Forecaster ki raay": v["verdict"], "Trade ki disha ka chance": _pct(v["p_dir"]), "Validated?": "haan" if s["fc"].get("trusted") else "nahi"})
    return rows


def compute(df, ns):
    """Heavy work (cached by the caller).  -> dict for the panel."""
    out = {"ok": False}
    try:
        src = str((getattr(df, "attrs", {}) or {}).get("market_source") or ns.get("market_source") or "")
    except Exception:
        src = ""
    tradable = ("NON_TRADABLE" not in src) and ("UNAVAILABLE" not in src)
    d = mm.prep(df)
    if d is None:
        out["msg"] = "Candle data abhi kam/unavailable hai."
        return out
    extras = mm.build_extras(ns) if tradable else {}
    rec = mm.record(df, extras) if tradable else {"inserted": 0, "labelled": 0, "live_ts": None}
    F = mm.compute_features(d)
    f_now, ts_now = mm.latest_features(d, F, extras)
    snaps = mm.load_snapshots()
    fc = ml.forecast(snaps, f_now or {}) if f_now else {"ready": False, "msg": "features nahi bane"}
    if rec.get("live_ts") and fc.get("ready"):
        mm.save_forecast(rec["live_ts"], _fc_compact(fc, rec["live_ts"]))
    try:
        trades = tl.get_recent_setups(limit=40) or []
    except Exception:
        trades = []
    autopsies = ma.analyze_all(df, snaps, trades)
    stored = mm.load_days()
    for day, s in autopsies.items():                 # finished days are kept permanently (history longer than the candle window)
        if s.get("complete") and day not in stored:
            mm.save_day(day, s)
            stored[day] = s
    merged = dict(stored)
    merged.update(autopsies)
    pstats, pbase = ml.pattern_stats(snaps)
    out.update({"ok": True, "rec": rec, "fc": fc, "f_now": f_now or {}, "ts_now": ts_now, "snaps_n": len(snaps),
                "lab_n": sum(1 for s in snaps if s.get("y")), "live_n": sum(1 for s in snaps if s.get("src") == "live"),
                "days": merged, "pstats": pstats, "pbase": pbase, "score": ml.scorecard(snaps),
                "tf": _trade_forecasts(snaps, trades), "lessons": ma.lessons(list(merged.values()))})
    # ---- v47 adaptive layer: strategy scorecard, Gemini memory, earned engine nudge, daily upgrade log (all guarded, advisory)
    try:
        strat = mad.strategy_stats(snaps)
        try:
            import shadow_trades
            shadow = (shadow_trades.summary() or {}).get("by_gate", {})
        except Exception:
            shadow = {}
        gem_sc = mad.gemini_scorecard(trades, shadow)
        mad.update_state(fc, pstats, pbase, ml.active_patterns(f_now or {}), out["score"])
        out.update({"strat": strat, "shadow": shadow, "gem_sc": gem_sc, "adj": mad.adjustment_status()})
        out["gem_text"] = mad.gemini_context(out, trades, gem_sc, strat)
        mad.publish_gemini_text(out["gem_text"])
        day = (ts_now or "")[:10]
        if day:
            metrics = mad.daily_metrics(out, trades, day)
            mad.save_daily_metrics(metrics)
            stats = mm.load_stats()
            stats[day] = metrics
            out["stats"] = dict(sorted(stats.items()))
        out["suggest"] = mad.suggestions(out, shadow, strat)
    except Exception:
        logger.exception("adaptive layer failed (ignored)")
    # ---- v48 post-trade review: why each closed trade won / lost, which factors were right or wrong (advisory only)
    try:
        if tpm.ENABLED:
            pms = tpm.build(df)
            fr, gr = tpm.factor_board(pms)
            out["pm"] = {"trades": pms, "factors": fr, "groups": gr, "modes": tpm.mode_table(pms), "suggest": tpm.suggestions(pms),
                         "days": mm.load_pm_days()}
            day_ = (ts_now or "")[:10]
            if day_:
                tpm.save_day_reviews(pms, day_)
            ws = tpm.warn_stats(pms)
            tpm.update_state(ws)
            mr, mg = tpm.missed_review(merged)
            out["pm"].update({"warn": ws, "missed": mr, "missed_gates": mg})
            note = tpm.gemini_note(pms)
            if note:                                   # keep the whole block inside the 1600-char limit ai_trade_decision reads
                base = out.get("gem_text") or ""
                cut = base[:1140]
                if len(base) > 1140 and "\n" in cut:
                    cut = cut[:cut.rfind("\n")]
                out["gem_text"] = (cut + "\n" + note).strip()
                mad.publish_gemini_text(out["gem_text"])
    except Exception:
        logger.exception("post-trade review failed (ignored)")
    return out


def run(st, df, ns):
    """Compute (cached per candle) and draw.  Safe: any error is logged and swallowed."""
    try:
        key = (str(df.index[-1]) if df is not None and len(df) else "", len(df) if df is not None else 0)
        ck = st.session_state.get("_mb_cache")
        if ck and ck.get("key") == key and (ck.get("t", 0) + 90) > __import__("time").time():
            res = ck["res"]
        else:
            res = compute(df, ns)
            st.session_state["_mb_cache"] = {"key": key, "res": res, "t": __import__("time").time()}
        _draw(st, res, ns)
    except Exception:
        logger.exception("market brain failed (ignored)")


# ----------------------------------------------------------------------------- UI
def _draw(st, res, ns):
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧬 Market Brain (roz market se seekhne wala dimaag)")
    if not res.get("ok"):
        st.info(res.get("msg") or "Market Brain abhi data ka intezaar kar raha hai.")
        return
    st.caption(f"Har 5-min candle record hoti hai aur 60 min baad uska natija label hota hai. Seekha hua: {res['lab_n']} candles "
               f"({res['live_n']} live, baaki pichle din ka back-fill). Storage: {mm.storage_note()}. "
               "Ye sirf SALAAH/JAANKARI hai -- koi trade nahi kholta ya rokta.")
    fc = res["fc"]

    with st.expander("🔮 Agla 60-min move -- forecaster (data se, guarantee nahi)", expanded=True):
        if not fc.get("ready"):
            st.info(fc.get("msg") or "Data kam hai.")
        else:
            c1, c2, c3 = st.columns(3)
            lean = fc.get("lean")
            c1.metric("Jhukav", {"UP": "⬆️ UP", "DOWN": "⬇️ DOWN"}.get(lean, "↔️ NEUTRAL"))
            c2.metric("+25 pehle chhune ka chance", _pct(fc.get("p_up25")))
            c3.metric("50 pt move aane ka chance", _pct(fc.get("p_move50")))
            st.caption(("✅ " if fc.get("trusted") else "⚠️ ") + str(fc.get("msg")) +
                       (f" | agar 50 pt move aaya to UP hone ka chance: {_pct(fc.get('p_up50'))}" if fc.get("p_up50") is not None else ""))
            rep = (fc.get("reports") or {}).get("dir25") or {}
            if rep.get("val_n"):
                st.caption(f"Test (naye data par): {rep['val_n']} candles me sahi {_pct(rep.get('val_acc'))} vs 'bas majority guess' {_pct(rep.get('val_base_acc'))}, "
                           f"AUC {rep.get('val_auc')}. Trusted tabhi banta hai jab baseline se behtar nikle.")
            dec = ns.get("_decision")
            if isinstance(dec, dict) and dec.get("has_setup"):
                v = ml.trade_view(fc, dec.get("direction"))
                st.markdown(f"**Engine ke abhi ke {dec.get('direction')} trade par forecaster:** {v['verdict']} (is disha ka chance {_pct(v['p_dir'])})")
        act = ml.active_patterns(res["f_now"]) if res.get("f_now") else []
        if act:
            st.markdown("**Abhi active patterns:** " + " | ".join(ml.pattern_label(k) for k in act))
            ps = {p["key"]: p for p in res["pstats"]}
            lines = [f"• {p['text']}: 60 min me +25 pehle {_pct(p['up'])}, -25 pehle {_pct(p['dn'])} (n={p['n']})" for k in act if (p := ps.get(k))]
            if lines:
                st.caption("\n".join(lines))

    days = res["days"]
    with st.expander("🔬 Market Autopsy -- market kahan se palta aur kyon", expanded=False):
        if not days:
            st.info("Abhi autopsy ke liye poora din nahi mila.")
        else:
            options = sorted(days.keys(), reverse=True)
            pick = st.selectbox("Din chuno", options, index=0, key="mb_day")
            s = days[pick]
            st.markdown(f"**{pick}** | Open {s['open']:,.0f} | High {s['high']:,.0f} | Low {s['low']:,.0f} | Close {s['close']:,.0f} | "
                        f"Range {s['range']:.0f} pts" + ("" if s.get("complete") else " | (din abhi chal raha hai)"))
            st.caption(f"50+ pt ki {s['n_major']} moves me se engine ne {s['n_major_caught']} me sahi disha ka trade liya.")
            rows = []
            for lg in s["legs"]:
                rows.append({"Shuru": lg["start"], "Khatam": lg["end"], "Disha": "⬆️" if lg["dir"] == "UP" else "⬇️",
                             "Se → Tak": f"{lg['from']:,.0f} → {lg['to']:,.0f}", "Points": lg["pts"], "50+": "✅" if lg["major"] else "",
                             "Kyon palta (shuru par)": "; ".join(c["text"] for c in lg["causes"]) or "Koi clear technical trigger nahi mila (shayad news / bade order -- data me nahi)",
                             "Engine": ", ".join(lg.get("engine") or []) or ("miss" if lg["major"] else "")})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                ctx = [lg["context"] for lg in s["legs"] if lg.get("context")]
                if ctx:
                    st.caption("Live context (sabse bade turn par): " + ctx[0])
            st.caption("Ye kaaran data me dikhne wale sanket hain -- pakka saboot nahi. News ya bade players ke order data me nahi hote.")

    with st.expander("📚 Seekhe hue patterns (data se: kaun si situation ke baad kya hota hai)", expanded=False):
        ps = res["pstats"]
        if not ps:
            st.info("Abhi kisi pattern ke liye kaafi samples nahi (kam se kam 12 chahiye).")
        else:
            b = res["pbase"]
            st.caption(f"Base: {b['n']} candles me 60 min me +25 pehle {_pct(b['up'])}, -25 pehle {_pct(b['dn'])}.")
            st.dataframe(pd.DataFrame([{"Pattern": p["text"], "Samples": p["n"], "+25 pehle": _pct(p["up"]), "-25 pehle": _pct(p["dn"]),
                                        "Koi nahi": _pct(p["none"]), "UP edge": f"{100 * p['lift_up']:+.0f}", "DOWN edge": f"{100 * p['lift_dn']:+.0f}"} for p in ps]),
                         use_container_width=True, hide_index=True)
        L = res["lessons"]
        if L["majors"]:
            rk = ", ".join(f"{ma.CAUSE_NAMES.get(k, k)} ({v}/{L['majors']})" for k, v in L["ranking"][:5])
            st.caption(f"Ab tak {L['majors']} badi (50+ pt) moves dekhi, engine ne {L['caught']} pakdi. Un moves ke turn par sabse aam kaaran: {rk}.")

    with st.expander("🎯 Scorecard -- forecaster live kitna sahi nikla", expanded=False):
        sc = res["score"]
        if not sc.get("n"):
            st.info("Live forecasts ke natije abhi nahi aaye (har forecast ka natija 60 min baad pata chalta hai).")
        else:
            st.markdown(f"Live forecasts jinka natija aa chuka: **{sc['n']}** | sahi: **{_pct(sc['acc'])}**"
                        + (f" | confident wale ({sc['n_conf']}): {_pct(sc['acc_conf'])}" if sc.get("acc_conf") is not None else "")
                        + (f" | validated model ke ({sc['n_trusted']}): {_pct(sc['acc_trusted'])}" if sc.get("acc_trusted") is not None else ""))
        if res["tf"]:
            st.caption("Engine ke trades par forecaster ki raay (entry se pehle wali). Isse pata chalega ki wo losing trades pehle pakad leta ya nahi:")
            st.dataframe(pd.DataFrame(res["tf"]), use_container_width=True, hide_index=True)
        st.caption("Jab ye scorecard kaafi trades par sabit kare, tab forecaster ko trade ke gate me soft penalty ke taur par jodna theek hoga (abhi OFF).")

    with st.expander("📈 Roz ka Upgrade Report -- kya system sach me behtar ho raha hai?", expanded=False):
        adj = res.get("adj") or {}
        if adj:
            st.markdown("**Engine ko nudge:** " + ("🟢 forecaster ne kamaya hai (chalu)" if adj.get("forecaster_earned") else
                        f"⚪ abhi band (forecaster ko live me {adj.get('fc_live_n', 0)}/{mad.FC_MIN_LIVE_N} validated natije chahiye, aur {int(100 * mad.FC_MIN_ACC)}%+ sahi)")
                        + f" | solid patterns (50+ samples): {adj.get('patterns_with_enough_samples', 0)}")
            st.caption("Nudge hamesha chhota (max +/-3.5 percentage point) aur apne aap tabhi chalu hota hai jab saboot mile; saboot gayab to band.")
        stats = res.get("stats") or {}
        if stats:
            rows = [{"Din": d, "Seekhi candles": m.get("lab_n"), "Forecaster AUC (test)": m.get("fc_auc"), "Test sahi": _pct(m.get("fc_acc")),
                     "Baseline": _pct(m.get("fc_base")), "Trusted": "✅" if m.get("fc_trusted") else "", "Live sahi": _pct(m.get("live_acc")),
                     "Engine W-L": f"{m.get('engine_w', 0)}-{m.get('engine_l', 0)}"} for d, m in list(stats.items())[-14:]]
            st.markdown("**Din-ba-din trend** (asli saboot: kya AUC/accuracy badh rahi hai?)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        strat = res.get("strat") or {}
        if strat:
            st.markdown("**Strategy scorecard (live)**")
            st.dataframe(pd.DataFrame([{"Strategy": k, "Signals": v["samples"], "Sahi disha": f"{v['wins']}/{v['samples']}", "Asli %": v["raw_win_rate"],
                                        "Random candle %": v["expected_win_rate"], "Score par asar (pts)": round(max(-8.0, min(8.0, (v["win_rate"] - 50.0) * 0.16)), 1)}
                                       for k, v in sorted(strat.items(), key=lambda z: -z[1]["win_rate"])]), use_container_width=True, hide_index=True)
        else:
            st.caption(f"Strategy scorecard abhi khali: har strategy ke kam se kam {mad.STRAT_MIN_EPISODES} live signals ka natija chahiye (dashboard khula rahe to roz judta hai).")
        g = res.get("gem_sc") or {}
        if g.get("approved_n") or g.get("rejected_n"):
            st.markdown(f"**Gemini scorecard:** approve kiye hue trades me {g.get('approved_wins', 0)}/{g.get('approved_n', 0)} jeete ({_pct(g.get('approved_wr'))})"
                        + (f" | jo setups Gemini ne roke unme {g['rejected_would_win']}/{g['rejected_n']} jeet jaate ({_pct(g.get('rejected_wr'))})" if g.get("rejected_n") else ""))
        if res.get("gem_text"):
            st.caption("Gemini ko review se pehle ye yaaddasht milti hai (salaah, live data ko override nahi kar sakti):")
            st.code(res["gem_text"], language=None)
        for s_ in (res.get("suggest") or []):
            st.markdown("• " + s_)
        st.caption("Suggestions sirf jaankari hain -- koi hard gate, SL ya target apne aap nahi badalta.")

    _draw_postmortem(st, res.get("pm"))


def _draw_postmortem(st, pm):
    with st.expander("🩺 Trade Post-Mortem -- har trade kyon jeeta / kyon haara (roz seekhta hai)", expanded=False):
        if not pm or not pm.get("trades"):
            st.info("Abhi koi band trade nahi mila. Trade band hote hi uska post-mortem yahan banega.")
            return
        T = pm["trades"]
        done = [t for t in T if t["status"] in ("WIN", "LOSS")]
        w = sum(t["status"] == "WIN" for t in done)
        st.caption(f"{len(T)} band trades ka review ({w} jeet / {len(done) - w} haar). Ye SIRF jaankari/salaah hai -- koi trade, SL ya target apne aap nahi badalta.")
        if pm.get("modes"):
            st.markdown("**Haar ke kaaran (loss kaise hue)**")
            st.dataframe(pd.DataFrame([{"Kaaran": m["text"], "Trades": m["n"], "Hissa": _pct(m["share"])} for m in pm["modes"]]),
                         use_container_width=True, hide_index=True)
        if pm.get("groups"):
            st.markdown("**Signal groups: aligned hone par kya sach me jeete? (global / news / fundamentals / strategy ...)**")
            st.dataframe(pd.DataFrame([{"Group": g["factor"], "Aligned trades": g["aligned_n"], "Aligned jeet": _pct(g["aligned_wr"]),
                                        "Na-aligned trades": g["not_n"], "Na-aligned jeet": _pct(g["not_wr"]), "Natija": g["verdict"]} for g in pm["groups"]]),
                         use_container_width=True, hide_index=True)
            st.caption(f"Verdict tabhi aata hai jab dono taraf kam se kam {tpm.MIN_GROUP_N} trades hon aur farq statistically saaf ho; warna 'data kam'.")
        if pm.get("factors"):
            with st.expander("Har factor alag se", expanded=False):
                st.dataframe(pd.DataFrame([{"Factor": f["factor"], "Group": f["group"], "Aligned": f["aligned_n"], "Aligned jeet": _pct(f["aligned_wr"]),
                                            "Na-aligned": f["not_n"], "Na-aligned jeet": _pct(f["not_wr"]), "Natija": f["verdict"]} for f in pm["factors"]]),
                             use_container_width=True, hide_index=True)
        ws = pm.get("warn") or {}
        if ws.get("hi_n") or ws.get("lo_n"):
            st.markdown(f"**Chetawani-filter:** jab aadhe+ signal groups trade ke khilaf the, trades {_pct(ws.get('hi_wr'))} jeete ({ws.get('hi_n', 0)} trades) "
                        f"vs baaki {_pct(ws.get('lo_wr'))} ({ws.get('lo_n', 0)}). " +
                        ("🟢 Saabit ho chuka -- isliye aise naye trade par chhota penalty (max ~2.5 pt) lag raha hai." if ws.get("earned") else
                         f"⚪ Abhi saabit nahi (dono taraf {tpm.WARN_MIN_N}+ trades aur saaf farq chahiye) -- penalty band."))
        if pm.get("missed"):
            st.markdown("**Jo badi moves chhoot gayi -- us waqt kaun sa gate bola?**")
            if pm.get("missed_gates"):
                st.caption("Gate-wise chhoote 50+ pt moves: " + ", ".join(f"{g}: {n}" for g, n in sorted(pm["missed_gates"].items(), key=lambda z: -z[1])))
            st.dataframe(pd.DataFrame([{"Din": r["day"], "Shuru": r["start"], "Disha": "⬆️" if r["dir"] == "UP" else "⬇️", "Points": r["pts"],
                                        "Gate": r["gate"], "Wajah": r["reason"]} for r in pm["missed"]]), use_container_width=True, hide_index=True)
            st.caption("Agar ek gate bar-bar badi moves rokta dikhe aur shadow trades me wo jeet-te hon, tab use dheela karna soch sakte hain (apne aap nahi hota).")
        for s_ in pm.get("suggest") or []:
            st.markdown("• " + s_)
        st.markdown("**Har trade ki kahani (naya pehle)**")
        for t in T[:12]:
            icon = {"WIN": "✅", "LOSS": "❌"}.get(t["status"], "⏱️")
            st.markdown(f"{icon} **{t['day']} {t['time']}** {t['status']} -- {t['story']}")
        st.caption("Kaaran data me dikhne wale sanket hain. News ya bade players ka asli asar data me nahi hota -- usse pakka nahi kaha ja sakta.")

