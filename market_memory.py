"""
market_memory.py -- the MARKET RECORDER (v46, purely additive).

WHY
  The trade engine only learns from its own few trades.  The market gives ~75 five-minute candles every day, and every one
  of them is a free lesson: "the market looked like THIS, and the next 60 minutes did THAT".  This module records exactly that.

WHAT IT DOES
  1. For every COMPLETED 5-min candle it saves a feature snapshot (price action, volume, RSI/EMA/VWAP stretch, day position,
     opening range ... and, when the app is live, option-chain OI walls, PCR, VIX, breadth, global mood, strategy state).
  2. Once 12 more candles exist it adds the OUTCOME label: how far price went up / down in the next 60 min and which of the
     +/-25 pt and +/-50 pt levels was touched FIRST.
  3. On the first run it BACK-FILLS from the candles the app already has (about a week) so learning starts on day one
     (price-only features; the live-only extras start accumulating from today).

STORAGE  Supabase tables `market_snapshots` / `market_autopsy` if they exist (run SUPABASE_MARKET_BRAIN_MIGRATION.sql once),
         otherwise the local SQLite file (works, but is lost when the app restarts -- the back-fill re-creates the price part).

SAFETY   Nothing here can open, block or change a trade.  Every public function swallows its own errors.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from datetime import time as dtime
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

import trade_learning as tl

logger = logging.getLogger(__name__)

SNAP_TABLE = "market_snapshots"
DAY_TABLE = "market_autopsy"
HORIZON_BARS = 12            # 12 x 5 min = 60 min look-ahead
MIN_PARTIAL_BARS = 4         # end-of-day candles: label with what is left of the session once the day is over
MOVE_LEVELS = (25.0, 50.0)
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# features every row has (computed from the candles alone -> also available for the back-fill)
BASE_FEATS = [
    "ema20_dist_atr", "ema50_dist_atr", "rsi", "macd_h_atr", "bb_pos", "vwap_dist_atr", "vol_ratio", "candle_atr",
    "range_atr", "upper_wick", "lower_wick", "run6_atr", "streak", "day_move_pct", "day_pos", "dist_dayhigh_atr",
    "dist_daylow_atr", "dist_pdh_atr", "dist_pdl_atr", "or_pos", "round50_dist", "mins_since_open", "atr",
]
# live-only extras (None for back-filled rows)
EXT_FEATS = ["pcr", "vix", "breadth_net", "global_avg", "call_wall_dist", "put_wall_dist",
             "call_wall_share", "put_wall_share", "strat_dir", "strat_score"]
EXTRA_KEYS = EXT_FEATS + ["call_wall", "put_wall", "news_sent", "sr_loc", "strats"]

_SB_RETRY_S = 600
_sb_state: dict = {}
_cache: dict = {}
_state: dict = {"key": None}


# ----------------------------------------------------------------------------- small helpers
def _f(x, default=None):
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def ts_str(ts) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Kolkata").tz_localize(None)
    return t.strftime("%Y-%m-%d %H:%M:%S")


def prep(df):
    """Candle DataFrame -> clean copy with a naive-IST DatetimeIndex and every column the features need, or None."""
    try:
        if df is None or len(df) < 30:
            return None
        d = df.copy()
        idx = d.index if isinstance(d.index, pd.DatetimeIndex) else pd.DatetimeIndex(pd.to_datetime(d.index))
        if idx.tz is not None:
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        d.index = idx
        d = d[~d.index.duplicated(keep="last")].sort_index()
        for col in ("Open", "High", "Low", "Close"):
            if col not in d.columns:
                return None
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["Open", "High", "Low", "Close"])
        if len(d) < 30:
            return None
        d["Volume"] = pd.to_numeric(d["Volume"], errors="coerce").fillna(0.0) if "Volume" in d.columns else 0.0
        c = d["Close"]
        if "ATR" not in d.columns:
            tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
            d["ATR"] = tr.ewm(span=14, adjust=False).mean()
        d["ATR"] = pd.to_numeric(d["ATR"], errors="coerce").replace(0, np.nan).ffill().bfill().fillna(c * 0.002)
        if "EMA_20" not in d.columns:
            d["EMA_20"] = c.ewm(span=20, adjust=False).mean()
        if "EMA_50" not in d.columns:
            d["EMA_50"] = c.ewm(span=50, adjust=False).mean()
        if "RSI" not in d.columns:
            dl = c.diff()
            up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            d["RSI"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)
        if "MACD_Hist" not in d.columns:
            m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
            d["MACD_Hist"] = m - m.ewm(span=9, adjust=False).mean()
        if "BB_Upper" not in d.columns or "BB_Lower" not in d.columns:
            mid = c.rolling(20, min_periods=5).mean()
            sd = c.rolling(20, min_periods=5).std()
            d["BB_Upper"], d["BB_Lower"] = mid + 2 * sd, mid - 2 * sd
        return d
    except Exception:
        logger.exception("market_memory.prep failed (ignored)")
        return None


# ----------------------------------------------------------------------------- features
def compute_features(d: pd.DataFrame) -> pd.DataFrame:
    """One feature row per candle (BASE_FEATS + price/atr).  Pure function of the candles -> same code for back-fill and live."""
    o, h, l, c, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"]
    atr = d["ATR"]
    day = pd.Series(d.index.normalize(), index=d.index)
    rng = (h - l).replace(0, np.nan)

    dhigh = h.groupby(day).cummax()
    dlow = l.groupby(day).cummin()
    d_last_close = c.groupby(day).last()
    prev_close = day.map(d_last_close.shift(1))
    day_open = o.groupby(day).transform("first")
    ref = prev_close.fillna(day_open)
    day_high_full = h.groupby(day).max()
    day_low_full = l.groupby(day).min()
    pdh = day.map(day_high_full.shift(1))
    pdl = day.map(day_low_full.shift(1))

    tp = (h + l + c) / 3
    vol_ok = v.groupby(day).transform("sum") > 0
    w = v.where(vol_ok, 1.0)          # days with real volume: volume-weighted; days without (index spot): equal weights
    vwap = (tp * w).groupby(day).cumsum() / w.groupby(day).cumsum().replace(0, np.nan)
    vwap = vwap.fillna(tp)

    vr = v / v.rolling(20, min_periods=5).mean().replace(0, np.nan)
    vr = vr.fillna(1.0) if bool(vol_ok.any()) else pd.Series(1.0, index=d.index)

    bar_no = c.groupby(day).cumcount()
    or_h = h.where(bar_no < 6).groupby(day).transform("max")
    or_l = l.where(bar_no < 6).groupby(day).transform("min")
    or_pos = pd.Series(0.0, index=d.index)
    or_pos = or_pos.mask((bar_no >= 6) & (c > or_h), 1.0).mask((bar_no >= 6) & (c < or_l), -1.0)

    body = c - o
    streak = []
    s = 0
    for b in body.values:
        col = 1 if b > 0 else -1 if b < 0 else 0
        s = (s + col if (s > 0 and col > 0) or (s < 0 and col < 0) else col)
        streak.append(s)

    mins = (d.index.hour * 60 + d.index.minute) - (9 * 60 + 15)
    bbu, bbl = d["BB_Upper"], d["BB_Lower"]
    f = pd.DataFrame(index=d.index)
    f["ema20_dist_atr"] = (c - d["EMA_20"]) / atr
    f["ema50_dist_atr"] = (c - d["EMA_50"]) / atr
    f["rsi"] = d["RSI"]
    f["macd_h_atr"] = d["MACD_Hist"] / atr
    f["bb_pos"] = ((c - bbl) / (bbu - bbl).replace(0, np.nan)).clip(-0.5, 1.5)
    f["vwap_dist_atr"] = (c - vwap) / atr
    f["vol_ratio"] = vr.clip(0, 6)
    f["candle_atr"] = body / atr
    f["range_atr"] = (h - l) / atr
    f["upper_wick"] = ((h - np.maximum(o, c)) / rng).fillna(0)
    f["lower_wick"] = ((np.minimum(o, c) - l) / rng).fillna(0)
    f["run6_atr"] = (c - c.shift(6)) / atr
    f["streak"] = streak
    f["day_move_pct"] = (c - ref) / ref * 100
    f["day_pos"] = ((c - dlow) / (dhigh - dlow).replace(0, np.nan)).fillna(0.5)
    f["dist_dayhigh_atr"] = (dhigh - c) / atr
    f["dist_daylow_atr"] = (c - dlow) / atr
    f["dist_pdh_atr"] = (c - pdh) / atr
    f["dist_pdl_atr"] = (c - pdl) / atr
    f["or_pos"] = or_pos
    f["round50_dist"] = np.minimum(c % 50, 50 - (c % 50))
    f["mins_since_open"] = mins
    f["atr"] = atr
    f["price"] = c
    f["vwap"] = vwap
    return f.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------- outcome labels
def closed_days(d: pd.DataFrame, now=None) -> set:
    """Dates whose session is over (every date before the last one; the last one too once it printed a >=15:25 candle)."""
    dates = sorted(set(d.index.normalize()))
    if not dates:
        return set()
    out = set(dates[:-1])
    last = d.index[-1]
    if last.time() >= dtime(15, 25):
        out.add(dates[-1])
    return out


def compute_labels(d: pd.DataFrame, closed: set | None = None) -> dict:
    """ts_str -> {'up','dn','ret','f25','f50','bars'}  (None where the look-ahead is not finished yet).
    f25/f50 = which of +/-25 / +/-50 pts price touched FIRST inside the next 12 candles of the SAME day: UP, DN, NONE or AMB
    (both inside one candle -- order unknowable, ignored by the learners)."""
    closed = closed if closed is not None else closed_days(d)
    h, l, c = d["High"].values, d["Low"].values, d["Close"].values
    days = d.index.normalize()
    day_vals = days.values
    last_idx: dict = {}
    for i, dd in enumerate(day_vals):
        last_idx[dd] = i
    out = {}
    for i in range(len(d)):
        end = min(i + HORIZON_BARS, last_idx[day_vals[i]])
        nb = end - i
        full = nb == HORIZON_BARS
        if nb < 1 or not (full or (days[i] in closed and nb >= MIN_PARTIAL_BARS)):
            out[ts_str(d.index[i])] = None
            continue
        res = {25.0: "NONE", 50.0: "NONE"}
        done = {25.0: False, 50.0: False}
        mu = md = 0.0
        for j in range(i + 1, end + 1):
            up, dn = h[j] - c[i], c[i] - l[j]
            mu, md = max(mu, up), max(md, dn)
            for T in MOVE_LEVELS:
                if done[T]:
                    continue
                u, dw = up >= T, dn >= T
                if u and dw:
                    res[T], done[T] = "AMB", True
                elif u:
                    res[T], done[T] = "UP", True
                elif dw:
                    res[T], done[T] = "DN", True
        out[ts_str(d.index[i])] = {"up": round(float(mu), 1), "dn": round(float(md), 1), "ret": round(float(c[end] - c[i]), 1),
                                   "f25": res[25.0], "f50": res[50.0], "bars": int(nb)}
    return out


# ----------------------------------------------------------------------------- live-only extras
def chain_walls(raw_chain, price):
    """Nearest-big OI walls from the option chain DataFrame (columns Strike / Call OI / Put OI).  -> dict (may be empty)."""
    try:
        if not isinstance(raw_chain, pd.DataFrame) or raw_chain.empty or price is None:
            return {}
        cols = {str(k).lower(): k for k in raw_chain.columns}
        ks, kc, kp = cols.get("strike"), cols.get("call oi"), cols.get("put oi")
        if not (ks and kc and kp):
            return {}
        t = raw_chain[[ks, kc, kp]].apply(pd.to_numeric, errors="coerce").dropna()
        t = t[(t[ks] > price - 400) & (t[ks] < price + 400)]
        if t.empty:
            return {}
        above = t[t[ks] >= price - 5]
        below = t[t[ks] <= price + 5]
        out = {}
        mc, mp = float(t[kc].max()), float(t[kp].max())
        if not above.empty and mc > 0:
            r = above.loc[above[kc].idxmax()]
            out.update(call_wall=float(r[ks]), call_wall_dist=float(r[ks] - price), call_wall_share=float(r[kc]) / mc)
        if not below.empty and mp > 0:
            r = below.loc[below[kp].idxmax()]
            out.update(put_wall=float(r[ks]), put_wall_dist=float(r[ks] - price), put_wall_share=float(r[kp]) / mp)
        return out
    except Exception:
        return {}


def build_extras(ns: dict) -> dict:
    """Pick the live extras out of the app's variables (all optional)."""
    ex = {}
    try:
        price = _f(ns.get("live_price"))
        ex.update(chain_walls(ns.get("df_option_chain"), price))
        ex["pcr"] = _f(ns.get("current_pcr"))
        ex["vix"] = _f(ns.get("live_vix"))
        adv, dec = _f(ns.get("advances")), _f(ns.get("declines"))
        if adv is not None and dec is not None and (adv + dec) > 0:
            ex["breadth_net"] = (adv - dec) / (adv + dec)
        ex["global_avg"] = _f(ns.get("avg_market_change"))
        sr = ns.get("strategy_result")
        if isinstance(sr, dict):
            ex["strat_dir"] = 1.0 if sr.get("direction") == "BUY" else -1.0 if sr.get("direction") == "SELL" else 0.0
            ex["strat_score"] = _f(sr.get("score"))
        ra = sr.get("ranked_active") if isinstance(sr, dict) else None
        if isinstance(ra, list):             # which named strategies are live (and valid) right now -> strategy scorecard
            ex["strats"] = [f"{r.get('key')}:{r.get('direction')}" for r in ra
                            if isinstance(r, dict) and r.get("live_valid") and r.get("key") and r.get("direction")][:12]
        src = ns.get("sr_context")
        if isinstance(src, dict):
            ex["sr_loc"] = str(src.get("location", ""))[:20]
    except Exception:
        logger.exception("build_extras failed (ignored)")
    return {k: v for k, v in ex.items() if v is not None}


# ----------------------------------------------------------------------------- storage
def _sb_ok(table):
    if not tl._is_supabase_configured():
        return False
    now = time.time()
    st = _sb_state.get(table)
    if st and now - st[1] < _SB_RETRY_S:
        return st[0]
    try:
        r = requests.get(f"{tl._supabase_url}/rest/v1/{table}?select=*&limit=1", headers=tl._supabase_headers(), timeout=8)
        ok = r.status_code == 200
    except Exception:
        ok = False
    _sb_state[table] = (ok, now)
    return ok


def _local():
    conn = sqlite3.connect(tl.DB_NAME)
    conn.execute("CREATE TABLE IF NOT EXISTS market_snapshots (ts TEXT PRIMARY KEY, features TEXT, labels TEXT, src TEXT, forecast TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS market_autopsy (day TEXT PRIMARY KEY, summary TEXT)")
    conn.commit()
    return conn


def storage_note():
    if tl._is_supabase_configured() and _sb_ok(SNAP_TABLE):
        return "Supabase (permanent)"
    if tl._is_supabase_configured():
        return "local file ONLY -- SUPABASE_MARKET_BRAIN_MIGRATION.sql ek baar Supabase me run karo tabhi data permanent hoga"
    return "local file (restart par mit sakta hai)"


def _j(x):
    try:
        return json.loads(x) if isinstance(x, str) and x else (x if isinstance(x, dict) else None)
    except Exception:
        return None


def load_snapshots(max_rows=6000, force=False):
    """All snapshots, oldest first: [{'ts','f','y','src','fc'}]."""
    hit = _cache.get("snaps")
    if hit and not force and time.time() - hit[0] < 45:
        return hit[1]
    raw = None
    if _sb_ok(SNAP_TABLE):
        try:
            raw, off = [], 0
            while off < max_rows:
                r = requests.get(f"{tl._supabase_url}/rest/v1/{SNAP_TABLE}?select=ts,features,labels,src,forecast&order=ts.desc&limit=1000&offset={off}",
                                 headers=tl._supabase_headers(), timeout=15)
                r.raise_for_status()
                part = r.json()
                raw += part
                if len(part) < 1000:
                    break
                off += 1000
        except Exception:
            logger.exception("market snapshots read (supabase) failed; using local")
            raw = None
    if raw is None:
        conn = _local()
        conn.row_factory = sqlite3.Row
        raw = [dict(x) for x in conn.execute("SELECT ts,features,labels,src,forecast FROM market_snapshots ORDER BY ts DESC LIMIT ?", (max_rows,)).fetchall()]
        conn.close()
    rows = [{"ts": r["ts"], "f": _j(r.get("features")) or {}, "y": _j(r.get("labels")), "src": r.get("src") or "backfill", "fc": _j(r.get("forecast"))} for r in raw]
    rows.sort(key=lambda z: z["ts"])
    _cache["snaps"] = (time.time(), rows)
    return rows


def _insert_many(rows):
    if not rows:
        return
    _cache.pop("snaps", None)
    if _sb_ok(SNAP_TABLE):
        try:
            for i in range(0, len(rows), 200):
                r = requests.post(f"{tl._supabase_url}/rest/v1/{SNAP_TABLE}?on_conflict=ts",
                                  headers={**tl._supabase_headers(), "Prefer": "resolution=ignore-duplicates,return=minimal"},
                                  json=rows[i:i + 200], timeout=20)
                r.raise_for_status()
            return
        except Exception:
            logger.exception("market snapshots insert (supabase) failed; using local")
    conn = _local()
    conn.executemany("INSERT OR IGNORE INTO market_snapshots (ts,features,labels,src,forecast) VALUES (?,?,?,?,?)",
                     [(r["ts"], r["features"], r["labels"], r["src"], r.get("forecast")) for r in rows])
    conn.commit()
    conn.close()


def _update(ts, **cols):
    _cache.pop("snaps", None)
    if _sb_ok(SNAP_TABLE):
        try:
            r = requests.patch(f"{tl._supabase_url}/rest/v1/{SNAP_TABLE}?ts=eq.{quote(str(ts), safe='')}", headers=tl._supabase_headers(), json=cols, timeout=10)
            r.raise_for_status()
            return
        except Exception:
            logger.exception("market snapshot update (supabase) failed; using local")
    conn = _local()
    sets = ",".join(f"{k}=?" for k in cols)
    conn.execute(f"UPDATE market_snapshots SET {sets} WHERE ts=?", (*cols.values(), ts))
    conn.commit()
    conn.close()


def save_forecast(ts, forecast: dict):
    try:
        _update(ts, forecast=json.dumps(forecast, default=str))
    except Exception:
        logger.exception("save_forecast failed (ignored)")


def _load_day_rows():
    """Every row of the autopsy table: {key: dict}.  Keys 'YYYY-MM-DD' = day autopsies, 'stats-YYYY-MM-DD' = daily upgrade metrics."""
    hit = _cache.get("days")
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    raw = None
    if _sb_ok(DAY_TABLE):
        try:
            r = requests.get(f"{tl._supabase_url}/rest/v1/{DAY_TABLE}?select=day,summary&order=day.desc&limit=400", headers=tl._supabase_headers(), timeout=10)
            r.raise_for_status()
            raw = r.json()
        except Exception:
            logger.exception("autopsy read (supabase) failed; using local")
            raw = None
    if raw is None:
        conn = _local()
        conn.row_factory = sqlite3.Row
        raw = [dict(x) for x in conn.execute("SELECT day,summary FROM market_autopsy ORDER BY day DESC LIMIT 400").fetchall()]
        conn.close()
    out = {}
    for r in raw:
        s = _j(r.get("summary"))
        if s:
            out[r["day"]] = s
    _cache["days"] = (time.time(), out)
    return out


def load_days():
    """Stored daily autopsies: {day: summary_dict}."""
    return {k: v for k, v in _load_day_rows().items() if not str(k).startswith(("stats-", "pm-"))}


def load_pm_days():
    """v48: stored daily post-trade reviews {day: summary_dict}, oldest first (rows keyed 'pm-YYYY-MM-DD')."""
    rows = {k[3:]: v for k, v in _load_day_rows().items() if str(k).startswith("pm-")}
    return dict(sorted(rows.items()))


def load_stats():
    """Stored daily upgrade metrics: {day: metrics_dict}, oldest first."""
    rows = {k[6:]: v for k, v in _load_day_rows().items() if str(k).startswith("stats-")}
    return dict(sorted(rows.items()))


def save_day(day, summary: dict):
    _cache.pop("days", None)
    payload = json.dumps(summary, default=str)
    if _sb_ok(DAY_TABLE):
        try:
            r = requests.post(f"{tl._supabase_url}/rest/v1/{DAY_TABLE}?on_conflict=day",
                              headers={**tl._supabase_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
                              json=[{"day": day, "summary": payload}], timeout=10)
            r.raise_for_status()
            return
        except Exception:
            logger.exception("autopsy save (supabase) failed; using local")
    conn = _local()
    conn.execute("INSERT OR REPLACE INTO market_autopsy (day,summary) VALUES (?,?)", (day, payload))
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------- the recorder
def _clean(rowdict):
    return {k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else (round(v, 4) if isinstance(v, float) else v))
            for k, v in rowdict.items()}


def latest_features(d, F, extras=None):
    """Feature dict of the latest COMPLETED candle (+ live extras) and its timestamp string, or (None, None)."""
    try:
        now = tl._now_ist()
        forming = now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE and d.index[-1].date() == now.date()
        row = F.iloc[-2] if forming and len(F) > 1 else F.iloc[-1]
        ts = ts_str(F.index[-2] if forming and len(F) > 1 else F.index[-1])
        f = _clean(row.to_dict())
        f.update(extras or {})
        return f, ts
    except Exception:
        return None, None


def record(df, extras=None):
    """Save new completed candles (+ back-fill), then add outcome labels where the 60-min look-ahead is finished.
    Runs the heavy part once per new candle.  Returns {'inserted','labelled','live_ts'} -- never raises."""
    out = {"inserted": 0, "labelled": 0, "live_ts": None}
    try:
        d = prep(df)
        if d is None:
            return out
        now = tl._now_ist()
        forming = now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE and d.index[-1].date() == now.date()
        done = d.iloc[:-1] if forming else d
        if len(done) < 30:
            return out
        key = (ts_str(done.index[-1]), len(done))
        if _state.get("key") == key:
            return out
        F = compute_features(done)
        labs = compute_labels(done)
        snaps = load_snapshots(force=True)
        known = {s["ts"]: s for s in snaps}
        last_ts = ts_str(done.index[-1])

        new_rows = []
        for ts in F.index:
            t = ts_str(ts)
            if t in known:
                continue
            f = _clean(F.loc[ts].to_dict())
            live = (t == last_ts and bool(extras))
            if live:
                f.update(extras)
            y = labs.get(t)
            new_rows.append({"ts": t, "features": json.dumps(f), "labels": json.dumps(y) if y else None,
                             "src": "live" if live else "backfill", "forecast": None})
        _insert_many(new_rows)
        out["inserted"] = len(new_rows)
        out["live_ts"] = last_ts if any(r["ts"] == last_ts and r["src"] == "live" for r in new_rows) else None

        pend = 0
        for s in snaps:
            if s["y"] is None and labs.get(s["ts"]):
                _update(s["ts"], labels=json.dumps(labs[s["ts"]]))
                pend += 1
                if pend >= 400:
                    break
        out["labelled"] = pend
        _state["key"] = key
    except Exception:
        logger.exception("market_memory.record failed (ignored)")
    return out
