"""
gate_log.py -- a PERSISTENT, per-candle record of what the trade engine decided (and WHY it did not trade).

Why: the old "Kaun sa gate kitni baar roka" table lives in st.session_state, so it starts from zero every time the page is
opened/reloaded (the keeper robot reloads the page every 30 min).  When you open the app at 12:24 and ask "why no trade in
the morning rally?", the table could only describe the last few minutes.  This log is written to disk on every candle, so
the morning is still there at noon, and it also shows the gaps when the engine was NOT running at all (tab closed, app
asleep, Upstox not logged in) -- which is the one reason no gate can explain.

Pure bookkeeping: it never changes a trade decision, never raises into the dashboard.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timedelta

try:
    from app_logging import get_logger
    logger = get_logger(__name__)
except Exception:                       # tests / stand-alone use
    import logging
    logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("GATE_LOG_DB", "gate_log.db")
_LOCK = threading.Lock()
MARKET_START = (9, 15)
CANDLE_MIN = 5
GAP_REPORT_MIN = 15          # a hole in the log longer than this is reported as "engine was not running"


def _conn():
    c = sqlite3.connect(DB_PATH, timeout=5)
    c.execute("""CREATE TABLE IF NOT EXISTS gate_log (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   day TEXT, candle TEXT, seen_at TEXT, price REAL, gate TEXT, direction TEXT,
                   reason TEXT, source TEXT, detail TEXT)""")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_gate_candle ON gate_log(candle, gate)")
    return c


def record(candle, price, gate, reason="", direction="", source="", detail="", now=None):
    """One row per (candle, gate).  Safe to call on every 30 s refresh."""
    try:
        now = now or datetime.now()
        candle = str(candle)[:19]
        with _LOCK:
            c = _conn()
            try:
                c.execute("INSERT OR IGNORE INTO gate_log(day,candle,seen_at,price,gate,direction,reason,source,detail) "
                          "VALUES (?,?,?,?,?,?,?,?,?)",
                          (candle[:10], candle, now.strftime("%Y-%m-%d %H:%M:%S"), float(price) if price is not None else None,
                           str(gate), str(direction or ""), str(reason or "")[:400], str(source or ""), str(detail or "")[:300]))
                c.commit()
                c.execute("DELETE FROM gate_log WHERE day < date(?, '-14 day')", (candle[:10],))   # keep two weeks
                c.commit()
            finally:
                c.close()
    except Exception:
        logger.exception("gate_log.record failed (ignored)")


def rows_for_day(day):
    try:
        with _LOCK:
            c = _conn()
            try:
                cur = c.execute("SELECT candle, seen_at, price, gate, direction, reason, source, detail FROM gate_log "
                                "WHERE day=? ORDER BY candle, id", (str(day),))
                return [dict(zip(("candle", "seen_at", "price", "gate", "direction", "reason", "source", "detail"), r))
                        for r in cur.fetchall()]
            finally:
                c.close()
    except Exception:
        logger.exception("gate_log.rows_for_day failed (ignored)")
        return []


def _parse(ts):
    try:
        return datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def coverage(day, now=None, market_end=(15, 30)):
    """When was the engine actually running on `day`?  -> {first, last, candles, gaps:[(from,to,minutes)], missed_start_min}"""
    rows = rows_for_day(day)
    candles = sorted({_parse(r["candle"]) for r in rows if _parse(r["candle"])})
    out = {"candles": len(candles), "first": None, "last": None, "gaps": [], "missed_start_min": 0}
    if not candles:
        return out
    out["first"], out["last"] = candles[0], candles[-1]
    d0 = candles[0].replace(hour=MARKET_START[0], minute=MARKET_START[1], second=0)
    out["missed_start_min"] = max(0, int((candles[0] - d0).total_seconds() // 60))
    for a, b in zip(candles, candles[1:]):
        mins = int((b - a).total_seconds() // 60)
        if mins > GAP_REPORT_MIN:
            out["gaps"].append((a, b, mins))
    return out


def coverage_note(day, now=None):
    """Plain-Hinglish sentence about engine uptime today, or '' when there is nothing worth saying."""
    cov = coverage(day, now)
    if not cov["candles"]:
        return ""
    parts = []
    if cov["missed_start_min"] >= GAP_REPORT_MIN:
        parts.append(f"engine ne aaj {cov['first'].strftime('%H:%M')} se dekhna shuru kiya -- 09:15 se {cov['missed_start_min']} min ka "
                     f"hissa record me hi nahi (us waqt app/tab band tha ya Upstox login nahi tha, to koi trade ban hi nahi sakta tha)")
    for a, b, mins in cov["gaps"][:3]:
        parts.append(f"{a.strftime('%H:%M')}-{b.strftime('%H:%M')} ({mins} min) engine chal hi nahi raha tha")
    return " | ".join(parts)


def summary(day):
    rows = rows_for_day(day)
    by_gate = {}
    for r in rows:
        by_gate[r["gate"]] = by_gate.get(r["gate"], 0) + 1
    return {"rows": rows, "by_gate": by_gate, "taken": by_gate.get("TRADE TAKEN", 0)}


def missed_moves(day, min_pts=70.0, window_min=75, max_report=4):
    """v55: the moves the engine did NOT trade, and which gate held the matching-direction trade back during each.
    A move = price travels >= min_pts within window_min minutes (up or down) using the per-candle prices in the log; it counts as missed when no
    'TRADE TAKEN' row sits inside [start - 10 min, end].  -> [{dir, start, end, pts, gates:[(gate, candles)], taken}]  (largest first).
    Pure reporting; never raises."""
    try:
        rows = rows_for_day(day)
        px = {}
        for r in rows:
            t, p = _parse(r["candle"]), r.get("price")
            if t is not None and p:
                px[t] = float(p)
        ts = sorted(px)
        found = []
        for i, a in enumerate(ts):
            seg = [t for t in ts[i:] if (t - a).total_seconds() <= window_min * 60]
            if len(seg) < 2:
                continue
            lo = min(seg, key=lambda t: px[t]); hi = max(seg, key=lambda t: px[t])
            if lo == hi:
                continue
            pts = px[hi] - px[lo]
            if pts < min_pts:
                continue
            up = lo < hi                                   # low came first -> rally
            s, e = (lo, hi) if up else (hi, lo)
            found.append((pts, "BUY" if up else "SELL", s, e))
        found.sort(key=lambda x: -x[0])
        out, used = [], []
        for pts, d, s, e in found:
            if any(not (e < us or s > ue) for us, ue in used):
                continue
            used.append((s, e))
            lo_t, hi_t = s - timedelta(minutes=10), e
            taken = any(r["gate"] == "TRADE TAKEN" and lo_t <= (_parse(r["candle"]) or lo_t - timedelta(days=1)) <= hi_t for r in rows)
            gates = {}
            for r in rows:
                t = _parse(r["candle"])
                if t is None or not (s <= t <= e) or r["gate"] == "TRADE TAKEN":
                    continue
                if r.get("direction") in (d, "", None):
                    gates[r["gate"]] = gates.get(r["gate"], 0) + 1
            out.append({"dir": d, "start": s, "end": e, "pts": round(pts, 1), "taken": taken,
                        "gates": sorted(gates.items(), key=lambda kv: -kv[1])[:4]})
            if len(out) >= max_report:
                break
        return out
    except Exception:
        logger.exception("gate_log.missed_moves failed (ignored)")
        return []
