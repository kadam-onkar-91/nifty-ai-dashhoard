"""
market_autopsy.py -- "market kahan se palta, kyon palta" (v46, purely additive).

For one trading day: finds every swing (zig-zag on the 5-min highs/lows), and for each REVERSAL point lists the data-visible
reasons around it: stop-hunt wick through the day's/previous day's extreme, volume spike, RSI / EMA stretch, strong reaction candle,
OI wall (live data), VWAP, round number, opening range ...  It also checks which big moves the trade engine caught or missed.

HONESTY  The reasons are what the DATA shows around the turn -- strong hints, not proof.  News / big-player orders that are not in
         the data cannot be seen; then the autopsy says so instead of inventing a story.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

import market_memory as mm

logger = logging.getLogger(__name__)

MIN_LEG_PTS = 35.0        # smallest swing the zig-zag reports
MAJOR_LEG_PTS = 50.0      # a "major" move (the size the user cares about)
TOP_CAUSES = 4


def _v(F, col, i):
    try:
        x = float(F[col].iloc[i])
        return None if math.isnan(x) else x
    except Exception:
        return None


def zigzag(h, l, thr):
    """[(bar_index, price, 'H'|'L')] -- confirmed pivots (a move of >= thr against the last extreme).  The still-running last
    extreme is appended as a provisional pivot."""
    n = len(h)
    piv, d = [], 0
    hi, hi_i, lo, lo_i = h[0], 0, l[0], 0
    ext = ext_i = None
    for i in range(n):
        if d == 0:
            if h[i] > hi:
                hi, hi_i = h[i], i
            if l[i] < lo:
                lo, lo_i = l[i], i
            if hi - lo >= thr:                      # whichever extreme came FIRST is the first pivot
                if hi_i < lo_i:
                    piv.append((hi_i, hi, "H"))
                    d, ext, ext_i = -1, lo, lo_i
                elif lo_i < hi_i:
                    piv.append((lo_i, lo, "L"))
                    d, ext, ext_i = 1, hi, hi_i
        elif d == 1:
            if h[i] > ext:
                ext, ext_i = h[i], i
            elif ext - l[i] >= thr:
                piv.append((ext_i, ext, "H"))
                d, ext, ext_i = -1, l[i], i
        else:
            if l[i] < ext:
                ext, ext_i = l[i], i
            elif h[i] - ext >= thr:
                piv.append((ext_i, ext, "L"))
                d, ext, ext_i = 1, h[i], i
    if d != 0 and ext_i is not None and (not piv or piv[-1][0] != ext_i):
        piv.append((ext_i, ext, "H" if d == 1 else "L"))
    return piv


def explain_pivot(d, F, p, kind, snap=None):
    """Data-visible reasons for the turn at bar p.  kind 'H' = a top (price then fell), 'L' = a bottom (price then rose).
    -> [(weight, key, Hinglish text)] strongest first."""
    out = []
    top = kind == "H"
    n = len(d)
    h, l, c = d["High"].values, d["Low"].values, d["Close"].values
    nxt = min(p + 1, n - 1)
    atr = _v(F, "atr", p) or 12.0
    side_wick = _v(F, "upper_wick" if top else "lower_wick", p)
    rng_atr = _v(F, "range_atr", p)
    prior = slice(0, p)
    if top:
        new_ext = p > 0 and h[p] >= (h[prior].max() if p > 0 else h[p])
        pd_lvl = _v(F, "dist_pdh_atr", p)
    else:
        new_ext = p > 0 and l[p] <= (l[prior].min() if p > 0 else l[p])
        pd_lvl = _v(F, "dist_pdl_atr", p)

    if new_ext and side_wick is not None and side_wick >= 0.4:
        out.append((3, "sweep", "Din ka naya %s bana kar wapas aaya (stop-hunt / liquidity sweep)" % ("high" if top else "low")))
    elif side_wick is not None and side_wick >= 0.5 and (rng_atr or 0) >= 0.8:
        out.append((2, "wick", "Lambi %s wick: %s ne price ko wapas dhakela" % ("upar ki" if top else "neeche ki", "sellers" if top else "buyers")))
    if pd_lvl is not None and abs(pd_lvl) <= 0.7:
        out.append((2, "prev_day", "Pichle din ke %s ke paas ruka" % ("high" if top else "low")))
    rsi = _v(F, "rsi", p)
    if rsi is not None and ((top and rsi >= 68) or (not top and rsi <= 32)):
        out.append((2, "rsi", "RSI %s (%.0f) -- move thak chuka tha" % ("overbought" if top else "oversold", rsi)))
    st = _v(F, "ema20_dist_atr", p)
    if st is not None and ((top and st >= 1.8) or (not top and st <= -1.8)):
        out.append((2, "stretch", "Price EMA20 se %.1f ATR %s khicha hua tha" % (abs(st), "upar" if top else "neeche")))
    vr = max([x for x in (_v(F, "vol_ratio", p), _v(F, "vol_ratio", nxt)) if x is not None] or [1.0])
    if vr >= 1.8:
        out.append((2, "volume", "Volume spike (%.1fx normal) turn ke aas-paas" % vr))
    ca, ra = _v(F, "candle_atr", nxt), _v(F, "range_atr", nxt)
    if ca is not None and ((top and ca <= -0.8) or (not top and ca >= 0.8) or (ra and ra >= 1.5 and ((top and ca < 0) or (not top and ca > 0)))):
        out.append((3, "reaction", "Agli candle me tez ulta move (turn ki confirmation)"))
    vw = _v(F, "vwap_dist_atr", p)
    if vw is not None and abs(vw) <= 0.35:
        out.append((2, "vwap", "VWAP ke paas rejection"))
    r50 = _v(F, "round50_dist", p)
    if r50 is not None and r50 <= 6:
        out.append((1, "round", "50 ke round number ke paas"))
    mins = _v(F, "mins_since_open", p)
    if mins is not None and mins <= 30:
        out.append((1, "opening", "Opening ke pehle 30 min ki volatility"))
    # LIVE-only context (option chain / PCR) -- only when that candle was recorded live
    if snap:
        wall = snap.get("call_wall" if top else "put_wall")
        if wall is not None:
            px = h[p] if top else l[p]
            if abs(float(wall) - px) <= 15:
                out.append((3, "oi_wall", "%s OI wall %.0f ke paas -- writers ne roka" % ("Call" if top else "Put", float(wall))))
    out.sort(key=lambda z: -z[0])
    return out


def _ctx_line(snap):
    if not snap:
        return "OI/PCR/global ka data us waqt record nahi tha (sirf price-based kaaran dikha sakte hain)"
    bits = []
    for k, lab, fmt in (("pcr", "PCR", "%.2f"), ("vix", "VIX", "%.1f"), ("global_avg", "global avg", "%+.2f%%"), ("breadth_net", "breadth", "%+.2f")):
        if snap.get(k) is not None:
            bits.append(lab + " " + fmt % snap[k])
    return ", ".join(bits) if bits else ""


def analyze_day(d, F, day, snaps_by_ts=None, trades=None):
    """d, F = prepared candles / features for ONE day (bars in order).  -> summary dict."""
    snaps_by_ts = snaps_by_ts or {}
    n = len(d)
    if n < 12:
        return None
    h, l, c, o = d["High"].values, d["Low"].values, d["Close"].values, d["Open"].values
    atr_med = float(np.nanmedian(d["ATR"].values))
    thr = max(MIN_LEG_PTS, 2.5 * atr_med)
    piv = zigzag(h, l, thr)
    legs = []
    for a, b in zip(piv[:-1], piv[1:]):
        pts = abs(b[1] - a[1])
        if pts < thr:
            continue
        kind_start = a[2]
        snap = snaps_by_ts.get(mm.ts_str(d.index[a[0]]))
        causes = explain_pivot(d, F, a[0], kind_start, snap)
        legs.append({
            "dir": "UP" if kind_start == "L" else "DOWN",
            "start": mm.ts_str(d.index[a[0]])[11:16], "end": mm.ts_str(d.index[b[0]])[11:16],
            "from": round(float(a[1]), 1), "to": round(float(b[1]), 1), "pts": round(float(pts), 1), "bars": int(b[0] - a[0]),
            "major": bool(pts >= MAJOR_LEG_PTS),
            "causes": [{"key": k, "text": t, "w": w} for w, k, t in causes[:TOP_CAUSES]],
            "context": _ctx_line(snap),
            "start_ts": mm.ts_str(d.index[a[0]]), "end_ts": mm.ts_str(d.index[b[0]]),
        })
    # which of those moves did the engine catch?
    for lg in legs:
        got = []
        for t in (trades or []):
            ts = str(t.get("timestamp") or "")[:19]
            want = "BUY" if lg["dir"] == "UP" else "SELL"
            if t.get("direction") == want and lg["start_ts"] <= ts <= lg["end_ts"]:
                got.append(f"{ts[11:16]} {t.get('direction')} ({t.get('status')})")
        lg["engine"] = got
    major = [x for x in legs if x["major"]]
    last_close = float(c[-1])
    summary = {
        "day": day, "open": round(float(o[0]), 1), "high": round(float(h.max()), 1), "low": round(float(l.min()), 1), "close": round(last_close, 1),
        "range": round(float(h.max() - l.min()), 1), "change": round(last_close - float(o[0]), 1), "bars": int(n),
        "thr": round(thr, 1), "legs": legs,
        "n_major": len(major), "n_major_caught": sum(1 for x in major if x["engine"]),
        "complete": bool(d.index[-1].time() >= mm.dtime(15, 25)),
    }
    return summary


def analyze_all(df, snaps=None, trades=None, now_day=None):
    """Autopsy of every day present in the candles.  -> {day: summary}.  Today's is marked complete=False until the close."""
    out = {}
    try:
        d = mm.prep(df)
        if d is None:
            return out
        F = mm.compute_features(d)
        by_ts = {s["ts"]: s["f"] for s in (snaps or [])}
        for day, idx in d.groupby(d.index.normalize()).indices.items():
            sub, fsub = d.iloc[idx], F.iloc[idx]
            s = analyze_day(sub, fsub, str(pd.Timestamp(day).date()), by_ts, trades)
            if s:
                out[s["day"]] = s
    except Exception:
        logger.exception("autopsy failed (ignored)")
    return out


def lessons(summaries):
    """Across days: which reasons show up most often behind major (>=50 pt) reversals, and how many majors the engine caught."""
    tot, caught = 0, 0
    cnt: dict = {}
    for s in summaries:
        for lg in s.get("legs", []):
            if not lg.get("major"):
                continue
            tot += 1
            caught += 1 if lg.get("engine") else 0
            for cz in lg.get("causes", []):
                cnt[cz["key"]] = cnt.get(cz["key"], 0) + 1
    rank = sorted(cnt.items(), key=lambda z: -z[1])
    return {"majors": tot, "caught": caught, "ranking": rank}


CAUSE_NAMES = {"sweep": "Stop-hunt / sweep", "wick": "Lambi rejection wick", "prev_day": "Pichle din ka high/low", "rsi": "RSI extreme",
               "stretch": "EMA se khicha hua", "volume": "Volume spike", "reaction": "Tez ulta candle", "vwap": "VWAP rejection",
               "round": "Round number", "opening": "Opening volatility", "oi_wall": "OI wall"}
