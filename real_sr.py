"""
real_sr.py -- REAL support / resistance builder.

Why this exists
---------------
The old builder had two structural problems:

1. The dashboard only holds ~5 days of 5-minute candles, but the code also
   computed "previous WEEK" and "previous MONTH" levels from that window --
   which is meaningless (a 5-day window has no complete previous month).
   Real weekly/monthly/daily-swing levels need DAILY candles.
2. Micro 5-minute swings, VWAP, EMA and every 50-pt round number were all
   thrown into one pile and clustered with a 3-pt tolerance.  That produced
   tiny 3-5 pt "zones" sitting 20-30 pts apart, so price was almost always
   labelled MIDRANGE / BETWEEN_LEVELS, and the engine bought "support" that
   was just a random wick.

What a REAL level is here
-------------------------
A level becomes actionable only when it has independent evidence:
  * higher-timeframe anchors: previous-day H/L/C, previous-week H/L,
    previous-month H/L, daily swing highs/lows, hourly swing highs/lows
  * intraday anchors: today's high/low, opening range, CPR/pivots
  * option-chain OI walls (max Call OI above spot = resistance, max Put OI
    below spot = support) -- only when the chain is LIVE, never simulated
  * PROVEN REACTIONS: price actually touched the zone and then moved away
    by >= 1.2 ATR (counted from the candles, not assumed)
Round numbers / VWAP / volume-profile / OB / FVG are only confluence.
Zones on the same side are forced to be at least MIN_SEP apart so the map
is not a cluster of 3-5 pt boxes.

Nothing here invents data: if daily candles or a live option chain are
unavailable, that evidence family is simply absent and the result says so.
"""
from app_logging import get_logger
logger = get_logger(__name__)

import numpy as np
import pandas as pd

MIN_SEP_ATR = 2.0          # same-side zones must be at least this many ATRs apart ...
MIN_SEP_FLOOR = 20.0       # ... and never closer than this many points
CLUSTER_TOL_ATR = 0.45     # members within this many ATRs merge into one zone
CLUSTER_TOL_MIN = 4.0
CLUSTER_TOL_MAX = 9.0
REACTION_MOVE_ATR = 1.2    # a "reaction" = price left the zone by this much within REACTION_BARS
REACTION_BARS = 12
GRADE_A = 6.5
GRADE_B = 4.2
GRADE_C = 2.6
CROSS_SEP_ATR = 1.5        # nearest support and nearest resistance must not be a tiny box around price
CROSS_SEP_FLOOR = 16.0
MAX_SIDE_ZONES = 5

# evidence families that are real anchors (can make a zone actionable)
ANCHOR_FAMILIES = {"daily", "weekly", "monthly", "daily_swing", "hourly_swing",
                   "opening_range", "session", "oi_wall", "pivot", "reaction"}


def _f(v):
    try:
        if v is None:
            return None
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _completed_daily(df, df_daily):
    """Completed daily candles (today's forming candle removed)."""
    today = df.index[-1].date() if isinstance(df.index, pd.DatetimeIndex) and len(df) else None
    d = None
    if df_daily is not None and not getattr(df_daily, "empty", True):
        try:
            d = df_daily[["High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
            d.index = pd.to_datetime(d.index)
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
        except Exception:
            logger.exception("daily frame unusable; deriving from intraday")
            d = None
    if d is None or len(d) < 2:
        try:
            d = df.groupby(df.index.date).agg({"High": "max", "Low": "min", "Close": "last"})
            d.index = pd.to_datetime(d.index)
        except Exception:
            return None
    if today is not None:
        d = d[d.index.date < today]
    return d if len(d) else None


def _fractal(series_hi, series_lo, lb):
    hi, lo = [], []
    h = np.asarray(series_hi, dtype=float)
    l = np.asarray(series_lo, dtype=float)
    for i in range(lb, len(h) - lb):
        if np.isfinite(h[i]) and h[i] >= np.nanmax(h[i - lb:i + lb + 1]):
            hi.append(float(h[i]))
        if np.isfinite(l[i]) and l[i] <= np.nanmin(l[i - lb:i + lb + 1]):
            lo.append(float(l[i]))
    return hi, lo


def _reaction_count(df, lo, hi, atr):
    """Count separate touch episodes of [lo, hi] followed by a real move away.

    Returns (touch_episodes, reactions_up, reactions_down).  Up-reaction = the
    zone acted as SUPPORT (price bounced up); down-reaction = acted as RESISTANCE.
    """
    try:
        h = pd.to_numeric(df["High"], errors="coerce").to_numpy()
        l = pd.to_numeric(df["Low"], errors="coerce").to_numpy()
        c = pd.to_numeric(df["Close"], errors="coerce").to_numpy()
    except Exception:
        return 0, 0, 0
    n = len(h)
    touching = (h >= lo) & (l <= hi)
    episodes = up = down = 0
    i = 0
    move = REACTION_MOVE_ATR * atr
    while i < n:
        if not touching[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and touching[j + 1]:
            j += 1
        episodes += 1
        # judge only episodes that have follow-through candles
        end = min(n, j + 1 + REACTION_BARS)
        if j + 2 <= n - 1:
            fut_hi = np.nanmax(h[j + 1:end]) if end > j + 1 else np.nan
            fut_lo = np.nanmin(l[j + 1:end]) if end > j + 1 else np.nan
            came_from_above = (c[i - 1] > hi) if i > 0 and np.isfinite(c[i - 1]) else False
            came_from_below = (c[i - 1] < lo) if i > 0 and np.isfinite(c[i - 1]) else False
            if np.isfinite(fut_hi) and came_from_above and fut_hi >= hi + move and c[j] >= lo:
                up += 1
            elif np.isfinite(fut_lo) and came_from_below and fut_lo <= lo - move and c[j] <= hi:
                down += 1
        i = j + 1
    return episodes, up, down


def _oi_walls(df_option_chain, price, atr):
    """Real OI walls: strongest Call OI above spot, strongest Put OI below spot."""
    out = []
    info = {"status": "UNAVAILABLE"}
    try:
        if df_option_chain is None or getattr(df_option_chain, "empty", True):
            return out, info
        c = df_option_chain
        if "Strike" not in c.columns or "Call OI" not in c.columns or "Put OI" not in c.columns:
            return out, info
        c = c.copy()
        c["_k"] = pd.to_numeric(c["Strike"], errors="coerce")
        c["_c"] = pd.to_numeric(c["Call OI"], errors="coerce").fillna(0.0)
        c["_p"] = pd.to_numeric(c["Put OI"], errors="coerce").fillna(0.0)
        reach = max(450.0, 8 * atr)
        c = c[(c["_k"] >= price - reach) & (c["_k"] <= price + reach)]
        if c.empty:
            return out, info
        above = c[c["_k"] > price]
        below = c[c["_k"] < price]
        info = {"status": "LIVE_OI", "rows": int(len(c))}
        if not above.empty and above["_c"].max() > 0:
            mx = float(above["_c"].max())
            for _, r in above.nlargest(3, "_c").iterrows():
                if r["_c"] >= 0.55 * mx:
                    put_here = float(r["_p"])
                    dom = float(r["_c"]) / max(put_here, 1.0)
                    out.append({"price": float(r["_k"]), "kind": "resistance",
                                "weight": 1.6 + 1.6 * float(r["_c"]) / mx + (0.4 if dom >= 1.5 else 0.0),
                                "call_oi": float(r["_c"]), "put_oi": put_here})
        if not below.empty and below["_p"].max() > 0:
            mx = float(below["_p"].max())
            for _, r in below.nlargest(3, "_p").iterrows():
                if r["_p"] >= 0.55 * mx:
                    call_here = float(r["_c"])
                    dom = float(r["_p"]) / max(call_here, 1.0)
                    out.append({"price": float(r["_k"]), "kind": "support",
                                "weight": 1.6 + 1.6 * float(r["_p"]) / mx + (0.4 if dom >= 1.5 else 0.0),
                                "put_oi": float(r["_p"]), "call_oi": call_here})
        return out, info
    except Exception:
        logger.exception("OI wall extraction failed")
        return [], {"status": "UNAVAILABLE"}


def build_real_sr_zones(df, live_price, atr, df_daily=None, df_option_chain=None,
                        oi_trusted=False, pivots=None, ob_list=None, fvg_list=None,
                        liquidity_map=None):
    """Return {'zones': [...], 'supports': [...], 'resistances': [...], 'meta': {...}}.

    Zone dicts use the same keys the trade engine already reads
    (price/low/high/side/distance_pts/touches/strength/score/sources/
    source_families/anchor_families/width_pts/quality_ok/micro_zone/actionable)
    plus grade / reactions / oi.
    """
    res = {"zones": [], "supports": [], "resistances": [], "meta": {}}
    price = _f(live_price)
    atr_v = _f(atr) or 12.0
    if df is None or getattr(df, "empty", True) or price is None or len(df) < 30:
        return res
    if not isinstance(df.index, pd.DatetimeIndex):
        return res

    tol = float(np.clip(CLUSTER_TOL_ATR * atr_v, CLUSTER_TOL_MIN, CLUSTER_TOL_MAX))
    min_sep = max(MIN_SEP_FLOOR, MIN_SEP_ATR * atr_v)
    cands = []

    def add(p, name, family, w, kind="neutral", extra=None):
        p = _f(p)
        if p is None or p <= 0 or abs(p - price) > 25 * atr_v:
            return
        d = {"price": p, "name": name, "family": family, "w": float(w), "kind": kind}
        if extra:
            d.update(extra)
        cands.append(d)

    daily = _completed_daily(df, df_daily)
    meta = {"daily_candles": int(len(daily)) if daily is not None else 0,
            "daily_source": "DAILY_FEED" if (df_daily is not None and not getattr(df_daily, "empty", True)) else "DERIVED_FROM_5MIN"}

    # ---- daily anchors -------------------------------------------------
    if daily is not None and len(daily):
        last = daily.iloc[-1]
        add(last["High"], "PDH", "daily", 3.0, "resistance")
        add(last["Low"], "PDL", "daily", 3.0, "support")
        add(last["Close"], "PDC", "daily", 2.2)
        for k, wt in zip(range(2, 6), (1.6, 1.4, 1.2, 1.0)):
            if len(daily) >= k:
                r = daily.iloc[-k]
                add(r["High"], f"Day-{k} High", "daily", wt, "resistance")
                add(r["Low"], f"Day-{k} Low", "daily", wt, "support")
        # weekly / monthly only when there is real history for them
        try:
            if len(daily) >= 8 and meta["daily_source"] == "DAILY_FEED":
                wk = daily.resample("W").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
                cur_week_start = df.index[-1].to_period("W").start_time
                wk = wk[wk.index.to_period("W").start_time < cur_week_start]
                if len(wk):
                    add(wk.iloc[-1]["High"], "Prev Week High", "weekly", 2.7, "resistance")
                    add(wk.iloc[-1]["Low"], "Prev Week Low", "weekly", 2.7, "support")
            if len(daily) >= 30 and meta["daily_source"] == "DAILY_FEED":
                try:
                    mo = daily.resample("ME").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
                except Exception:
                    mo = daily.resample("M").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
                cur_m = df.index[-1].to_period("M")
                mo = mo[mo.index.to_period("M") < cur_m]
                if len(mo):
                    add(mo.iloc[-1]["High"], "Prev Month High", "monthly", 2.5, "resistance")
                    add(mo.iloc[-1]["Low"], "Prev Month Low", "monthly", 2.5, "support")
        except Exception:
            logger.exception("weekly/monthly levels failed")
        # daily swing highs/lows (fractal, 3 bars each side) over the last ~90 sessions
        try:
            if len(daily) >= 9:
                dh, dl = _fractal(daily["High"].tail(90).to_numpy(), daily["Low"].tail(90).to_numpy(), 3)
                for v in dh:
                    add(v, "Daily swing high", "daily_swing", 1.9, "resistance")
                for v in dl:
                    add(v, "Daily swing low", "daily_swing", 1.9, "support")
        except Exception:
            logger.exception("daily swings failed")

    # ---- hourly swings (resampled from the 5-min base) -------------------
    try:
        agg = {"High": "max", "Low": "min", "Close": "last"}
        h1 = df[["High", "Low", "Close"]].resample("1h").agg(agg).dropna()
        if len(h1) >= 7:
            hh, hl = _fractal(h1["High"].to_numpy(), h1["Low"].to_numpy(), 2)
            for v in hh:
                add(v, "Hourly swing high", "hourly_swing", 1.5, "resistance")
            for v in hl:
                add(v, "Hourly swing low", "hourly_swing", 1.5, "support")
    except Exception:
        logger.exception("hourly swings failed")

    # ---- today's session + opening range ---------------------------------
    try:
        today = df.index[-1].date()
        td = df[df.index.date == today]
        if len(td) >= 12:
            add(td["High"].max(), "Session High", "session", 1.5, "resistance")
            add(td["Low"].min(), "Session Low", "session", 1.5, "support")
        if len(td) >= 4:
            orw = td[td.index <= td.index[0] + pd.Timedelta(minutes=15)]
            add(orw["High"].max(), "Opening Range High", "opening_range", 1.8, "resistance")
            add(orw["Low"].min(), "Opening Range Low", "opening_range", 1.8, "support")
    except Exception:
        logger.exception("session levels failed")

    # ---- CPR / pivots -----------------------------------------------------
    if pivots and isinstance(pivots, dict) and pivots.get("status") == "OK":
        for k, w in (("r1", 1.1), ("s1", 1.1), ("r2", 0.9), ("s2", 0.9), ("tc", 0.9), ("bc", 0.9), ("pivot", 0.8)):
            add(pivots.get(k), f"Pivot {k.upper()}", "pivot", w)

    # ---- confirmed 5-min swings (weak alone; gain strength from reactions) -
    try:
        w5 = df.tail(500)
        sh, sl = _fractal(w5["High"].to_numpy(), w5["Low"].to_numpy(), 6)
        for v in sh[-40:]:
            add(v, "5m swing high", "ltf_swing", 0.7, "resistance")
        for v in sl[-40:]:
            add(v, "5m swing low", "ltf_swing", 0.7, "support")
    except Exception:
        logger.exception("5m swings failed")

    # ---- OI walls (LIVE chain only) --------------------------------------
    oi_info = {"status": "UNAVAILABLE"}
    if oi_trusted:
        walls, oi_info = _oi_walls(df_option_chain, price, atr_v)
        for w in walls:
            add(w["price"], "Option Chain OI wall", "oi_wall", w["weight"], w["kind"],
                {"call_oi": w["call_oi"], "put_oi": w["put_oi"]})
    else:
        oi_info = {"status": "NOT_USED", "reason": "option chain not LIVE (simulated OI is never used as a level)"}
    meta["option_chain"] = oi_info

    # ---- confluence-only sources ------------------------------------------
    if "VWAP" in df.columns:
        add(df["VWAP"].iloc[-1], "Session VWAP", "vwap", 0.5)
    for z in (ob_list or []):
        try:
            p = float(str(z.get("Price")).replace(",", ""))
            add(p, "SMC Order Block", "smc", 0.8)
        except Exception:
            pass
    for z in (fvg_list or []):
        try:
            p = float(str(z.get("Price")).replace(",", ""))
            add(p, "SMC FVG", "smc", 0.4)
        except Exception:
            pass
    if liquidity_map:
        try:
            for x in liquidity_map.get("equal_highs", []):
                add(x.get("level"), "Equal High Liquidity", "liquidity", 1.0, "resistance")
            for x in liquidity_map.get("equal_lows", []):
                add(x.get("level"), "Equal Low Liquidity", "liquidity", 1.0, "support")
        except Exception:
            pass
    base100 = int(round(price / 100.0) * 100)
    for k in range(-4, 5):
        add(base100 + 100 * k, f"Round {base100 + 100 * k}", "round", 0.6)

    if not cands:
        return res

    # ---- cluster ----------------------------------------------------------
    cands.sort(key=lambda x: x["price"])
    clusters = [[cands[0]]]
    for cd in cands[1:]:
        cw = sum(m["w"] for m in clusters[-1])
        cc = sum(m["price"] * m["w"] for m in clusters[-1]) / cw
        if cd["price"] - cc <= tol:
            clusters[-1].append(cd)
        else:
            clusters.append([cd])

    zones = []
    for mem in clusters:
        tw = sum(m["w"] for m in mem)
        centre = sum(m["price"] * m["w"] for m in mem) / tw
        # Anchor to the strongest member so a PDH/PDL stays exactly PDH/PDL
        top = max(mem, key=lambda m: m["w"])
        if top["w"] >= 2.4:
            centre = top["price"]
        lo = min(m["price"] for m in mem) - 1.0
        hi = max(m["price"] for m in mem) + 1.0
        if hi - lo < 3.0:
            lo, hi = centre - 1.5, centre + 1.5

        fam_best = {}
        for m in mem:
            fam_best.setdefault(m["family"], []).append(m["w"])
        score = 0.0
        for fam, ws in fam_best.items():
            ws = sorted(ws, reverse=True)
            score += ws[0] + 0.25 * sum(ws[1:3])
        fams = sorted(fam_best.keys())
        real_fams = [f for f in fams if f not in ("round", "vwap", "smc")]
        score += 0.35 * min(max(len(real_fams) - 1, 0), 4)
        if any(m["family"] == "round" for m in mem) and real_fams:
            score += 0.4

        episodes, up, down = _reaction_count(df, lo, hi, atr_v)
        reactions = up + down
        score += min(reactions, 3) * 0.5
        if episodes >= 5 and reactions <= 1:
            score -= 0.8           # touched a lot, never bounced: consumed / not respected
        anchors = sorted(set(fams) & ANCHOR_FAMILIES)
        if reactions >= 2 and "reaction" not in anchors:
            anchors.append("reaction")

        oi_mem = [m for m in mem if m["family"] == "oi_wall"]
        oi = None
        if oi_mem:
            oi = {"call_oi": max(m.get("call_oi", 0) for m in oi_mem),
                  "put_oi": max(m.get("put_oi", 0) for m in oi_mem)}

        grade = "A" if score >= GRADE_A else ("B" if score >= GRADE_B else ("C" if score >= GRADE_C else "D"))
        multi = len(real_fams) >= 2
        actionable = bool(grade in ("A", "B") and anchors) or bool(grade == "C" and anchors and (multi or reactions >= 2))
        # a dominant LIVE option-chain OI wall is a real level on its own (writers defend it)
        if oi_mem and max(m["w"] for m in oi_mem) >= 3.0 and grade in ("A", "B", "C"):
            actionable = True

        inside = lo <= price <= hi
        if inside:
            side = "support" if price >= centre else "resistance"
        else:
            side = "support" if centre < price else "resistance"
        dist = 0.0 if inside else abs(price - centre)
        names = []
        for m in sorted(mem, key=lambda x: -x["w"]):
            if m["name"] not in names and m["family"] not in ("round", "vwap"):
                names.append(m["name"])
        if not names:
            names = [mem[0]["name"]]
        if any(m["family"] == "round" for m in mem) and real_fams:
            names.append("round-number confluence")

        zones.append({
            "price": round(centre, 2), "low": round(lo, 2), "high": round(hi, 2),
            "side": side, "distance_pts": round(dist, 2), "touches": int(episodes),
            "reactions": int(reactions), "reactions_up": int(up), "reactions_down": int(down),
            "strength": "STRONG" if grade == "A" else ("MODERATE" if grade == "B" else "WEAK"),
            "grade": grade, "score": round(score, 2), "sources": names[:6],
            "source_families": fams, "anchor_families": anchors,
            "width_pts": round(hi - lo, 2), "quality_ok": actionable,
            "micro_zone": False, "actionable": actionable, "oi": oi, "inside": inside,
        })

    # ---- enforce spacing (strongest wins) -----------------------------------
    # same-side zones >= min_sep apart, and opposite-side zones >= cross_sep apart,
    # so the map is never a tiny support/resistance box around price.  A zone that
    # price is currently INSIDE is always kept first (it is where the trade is).
    cross_sep = max(CROSS_SEP_FLOOR, CROSS_SEP_ATR * atr_v)
    pool = sorted([z for z in zones if z["actionable"]],
                  key=lambda z: (not z["inside"], -z["score"], z["distance_pts"]))
    chosen = []
    for z in pool:
        ok = True
        for k in chosen:
            gap = abs(z["price"] - k["price"])
            need = min_sep if z["side"] == k["side"] else cross_sep
            if gap < need:
                ok = False
                break
        if ok:
            chosen.append(z)
    keep_ids = {id(z) for z in chosen}
    for z in zones:
        if z["actionable"] and id(z) not in keep_ids:
            z["actionable"] = False
            z["quality_ok"] = False
            z["suppressed_by_separation"] = True
    meta_cross = round(cross_sep, 2)

    zones.sort(key=lambda z: z["distance_pts"])
    sup = [z for z in zones if z["actionable"] and z["side"] == "support"][:MAX_SIDE_ZONES]
    rsi = [z for z in zones if z["actionable"] and z["side"] == "resistance"][:MAX_SIDE_ZONES]
    meta.update({"tolerance_pts": round(tol, 2), "min_separation_pts": round(min_sep, 2), "cross_separation_pts": meta_cross,
                 "candidates": len(cands), "zones_total": len(zones)})
    res.update({"zones": zones, "supports": sup, "resistances": rsi, "meta": meta})
    return res
