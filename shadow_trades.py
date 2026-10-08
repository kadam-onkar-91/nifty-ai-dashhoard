"""
shadow_trades.py -- learn from setups the engine did NOT take, without risking money.

THE PROBLEM THIS SOLVES
  Gates (edge bar, entry quality, Gemini ...) protect money, but each blocked setup teaches nothing: no trade -> no result ->
  no learning -> the engine never finds out whether the block was right, and a bad stretch can freeze it ("no trades -> no data").
THE FIX
  Every blocked candidate that has a full entry / SL / target is logged as a SHADOW trade and followed with the same
  SL / target / time-stop rules as a real one.  Real trades are unchanged (Gemini still the final block -- shadow trades never
  place or suggest anything).  Resolved shadows are fed to the learners with a smaller weight (SHADOW_WEIGHT) and are summarised
  per gate, so you can SEE e.g. "Gemini blocked 9 setups, 7 would have lost" (Gemini is right) or "the entry-quality gate blocked
  8, 6 would have won" (that gate is too strict).

STORAGE  Supabase table `ai_shadow_setups` if it exists (run SUPABASE_SHADOW_MIGRATION.sql once), otherwise the local SQLite file
         (works, but is lost when the app restarts).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta

import requests

import trade_learning as tl

logger = logging.getLogger(__name__)

SHADOW_WEIGHT = 0.5                # a paper trade counts half as much as a real one (no slippage/fill risk in it)
SHADOW_MAX_PER_DAY = 15
SHADOW_MIN_GAP_MIN = 15            # per direction: a new shadow only this long after the previous one (avoid 10 copies of one idea)
SHADOW_MAX_OPEN_PER_DIRECTION = 1
SB_TABLE = "ai_shadow_setups"
_SB_RETRY_S = 600
_state = {"sb_ok": None, "sb_checked": 0.0}


def _local_init():
    conn = sqlite3.connect(tl.DB_NAME)
    conn.execute("""CREATE TABLE IF NOT EXISTS ai_shadow_setups
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, direction TEXT, underlying_entry REAL,
                     stop_loss REAL, target REAL, blocked_by TEXT, factors_json TEXT, status TEXT DEFAULT 'OPEN',
                     exit_price REAL, exit_timestamp TEXT)""")
    conn.commit(); conn.close()


def _use_supabase():
    """Supabase only if configured AND the shadow table exists (checked every 10 min, never raises)."""
    if not tl._is_supabase_configured():
        return False
    now = time.time()
    if _state["sb_ok"] is not None and now - _state["sb_checked"] < _SB_RETRY_S:
        return _state["sb_ok"]
    try:
        r = requests.get(f"{tl._supabase_url}/rest/v1/{SB_TABLE}?select=id&limit=1", headers=tl._supabase_headers(), timeout=8)
        _state["sb_ok"] = r.status_code == 200
    except Exception:
        _state["sb_ok"] = False
    _state["sb_checked"] = now
    return _state["sb_ok"]


def storage_note():
    if tl._is_supabase_configured() and _use_supabase():
        return "Supabase (permanent)"
    if tl._is_supabase_configured():
        return "local file ONLY -- run SUPABASE_SHADOW_MIGRATION.sql in Supabase once to make shadow data permanent"
    return "local file (not permanent)"


# ----------------------------------------------------------------- storage primitives
_CACHE = {}          # where -> (ts, rows); 8 s, cleared on every write (a refresh asks for the rows 3-4 times)


def _all_rows(where="", params=()):
    hit = _CACHE.get(where)
    if hit and time.time() - hit[0] < 8:
        return [dict(r) for r in hit[1]]
    rows = _all_rows_uncached(where)
    _CACHE[where] = (time.time(), rows)
    return [dict(r) for r in rows]


def _all_rows_uncached(where=""):
    """List of dict rows (newest first) from whichever backend is active."""
    if _use_supabase():
        try:
            q = f"{tl._supabase_url}/rest/v1/{SB_TABLE}?order=id.desc&limit=400"
            if where == "OPEN":
                q += "&status=eq.OPEN"
            r = requests.get(q, headers=tl._supabase_headers(), timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            logger.exception("shadow read (supabase) failed; using local")
    _local_init()
    conn = sqlite3.connect(tl.DB_NAME); conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM ai_shadow_setups" + (" WHERE status='OPEN'" if where == "OPEN" else "") + " ORDER BY id DESC LIMIT 400"
    rows = [dict(x) for x in conn.execute(sql).fetchall()]
    conn.close()
    return rows


def _insert(row):
    _CACHE.clear()
    if _use_supabase():
        try:
            r = requests.post(f"{tl._supabase_url}/rest/v1/{SB_TABLE}",
                              headers={**tl._supabase_headers(), "Prefer": "return=representation"}, json=row, timeout=10)
            r.raise_for_status()
            d = r.json()
            return d[0]["id"] if d else None
        except Exception:
            logger.exception("shadow insert (supabase) failed; using local")
    _local_init()
    conn = sqlite3.connect(tl.DB_NAME)
    cur = conn.execute("""INSERT INTO ai_shadow_setups (timestamp,direction,underlying_entry,stop_loss,target,blocked_by,factors_json,status)
                          VALUES (?,?,?,?,?,?,?,'OPEN')""",
                       (row["timestamp"], row["direction"], row["underlying_entry"], row["stop_loss"], row["target"],
                        row["blocked_by"], row["factors_json"]))
    sid = cur.lastrowid
    conn.commit(); conn.close()
    return sid


def _update(sid, fields):
    _CACHE.clear()
    if _use_supabase():
        try:
            r = requests.patch(f"{tl._supabase_url}/rest/v1/{SB_TABLE}?id=eq.{sid}", headers=tl._supabase_headers(), json=fields, timeout=10)
            r.raise_for_status()
            return
        except Exception:
            logger.exception("shadow update (supabase) failed; trying local")
    _local_init()
    sets = ",".join(f"{k}=?" for k in fields)
    conn = sqlite3.connect(tl.DB_NAME)
    conn.execute(f"UPDATE ai_shadow_setups SET {sets} WHERE id=?", (*fields.values(), sid))
    conn.commit(); conn.close()


# ----------------------------------------------------------------- public API
def log(candidate, blocked_by, now=None):
    """Record a blocked candidate as a shadow trade.  candidate needs direction, underlying_entry, stop_loss, target (+ optional
    factor_flags).  Returns the id, or None when skipped (cap / gap / already tracking one in that direction / bad data)."""
    try:
        d = candidate.get("direction")
        entry, sl, tp = float(candidate["underlying_entry"]), float(candidate["stop_loss"]), float(candidate["target"])
        if d not in ("BUY", "SELL") or abs(entry - sl) <= 0 or abs(tp - entry) <= 0:
            return None
        if (d == "BUY" and not (sl < entry < tp)) or (d == "SELL" and not (tp < entry < sl)):
            return None
    except Exception:
        return None
    now = now or tl._now_ist()
    rows = _all_rows()
    today = now.strftime("%Y-%m-%d")
    if sum(1 for r in rows if str(r.get("timestamp", ""))[:10] == today) >= SHADOW_MAX_PER_DAY:
        return None
    same_dir = [r for r in rows if r.get("direction") == d]
    if sum(1 for r in same_dir if r.get("status") == "OPEN") >= SHADOW_MAX_OPEN_PER_DIRECTION:
        return None
    if same_dir:
        try:
            last = datetime.strptime(str(same_dir[0]["timestamp"])[:19], "%Y-%m-%d %H:%M:%S")
            if (now - last) < timedelta(minutes=SHADOW_MIN_GAP_MIN):
                return None
        except Exception:
            pass
    flags = dict(candidate.get("factor_flags") or {})
    flags["_shadow"] = {"blocked_by": str(blocked_by)}
    return _insert({"timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "direction": d, "underlying_entry": entry,
                    "stop_loss": sl, "target": tp, "blocked_by": str(blocked_by)[:60], "factors_json": json.dumps(flags, default=str)})


def resolve(live_price, df=None):
    """Follow open shadows with the SAME rules as real trades: candle-scanned SL/target (both-in-one-candle = LOSS), time stop,
    max hold, end-of-day square-off, overnight carry -> EXPIRED."""
    try:
        opens = _all_rows("OPEN")
    except Exception:
        return
    now = tl._now_ist()
    for row in opens:
        try:
            sid, d = row["id"], row["direction"]
            entry, sl, tp = float(row["underlying_entry"]), float(row["stop_loss"]), float(row["target"])
            is_buy = d == "BUY"
            risk = max(abs(entry - sl), 1e-9)
            opened = datetime.strptime(str(row["timestamp"])[:19], "%Y-%m-%d %H:%M:%S")
            try:
                flags = json.loads(row.get("factors_json") or "{}")
            except Exception:
                flags = {}
            meta = flags.get("_shadow_meta") if isinstance(flags.get("_shadow_meta"), dict) else {}
            bars = tl._bars_since(df, opened)
            fav = max([((float(h) - entry) if is_buy else (entry - float(l))) for h, l in bars] + [((live_price - entry) if is_buy else (entry - live_price)), float(meta.get("fav", 0.0))])
            meta["fav"] = round(fav, 2)
            outcome, px = tl._scan_bars(is_buy, sl, tp, bars)
            if outcome is None:
                if opened.date() != now.date():
                    outcome, px = "EXPIRED", float(meta.get("last", entry))
                elif (is_buy and live_price >= tp) or ((not is_buy) and live_price <= tp):
                    outcome, px = "WIN", live_price
                elif (is_buy and live_price <= sl) or ((not is_buy) and live_price >= sl):
                    outcome, px = "LOSS", live_price
                else:
                    age = now - opened
                    stagnant = age >= timedelta(minutes=tl.TIME_STOP_MIN) and fav < tl.TIME_STOP_MIN_FAV_R * risk
                    if now.time() >= tl.EOD_SQUAREOFF or age > timedelta(minutes=tl.MAX_HOLD_MINUTES) or stagnant:
                        outcome, px = "EXPIRED", live_price
            if outcome is None:
                meta["last"] = round(float(live_price), 2)
            if outcome == "EXPIRED":
                pnl = (px - entry) if is_buy else (entry - px)
                meta["expired_r"] = round(pnl / risk, 3)
            flags["_shadow_meta"] = meta
            fields = {"factors_json": json.dumps(flags, default=str)}
            if outcome:
                fields.update({"status": outcome, "exit_price": px, "exit_timestamp": tl._now_str()})
            _update(sid, fields)
        except Exception:
            logger.exception("shadow resolve failed for one row")


def _label(row):
    """1 win / 0 loss / None (inconclusive) -- expired shadows are labelled by marked-to-market R like real ones."""
    st = row.get("status")
    if st == "WIN":
        return 1
    if st == "LOSS":
        return 0
    if st == "EXPIRED":
        try:
            r = (json.loads(row.get("factors_json") or "{}").get("_shadow_meta") or {}).get("expired_r")
        except Exception:
            r = None
        if r is None:
            return None
        if r >= tl.EXPIRED_LEARN_MIN_R:
            return 1
        if r <= -tl.EXPIRED_LEARN_MIN_R:
            return 0
    return None


def labelled_rows():
    """[(flags, label, features|None, blocked_by)] oldest first, for the learners."""
    out = []
    for r in reversed(_all_rows()):
        y = _label(r)
        if y is None:
            continue
        try:
            f = json.loads(r.get("factors_json") or "{}")
        except Exception:
            continue
        flags = {k: v for k, v in f.items() if not str(k).startswith("_")}
        out.append((flags, y, f.get("_features") if isinstance(f.get("_features"), dict) else None, r.get("blocked_by")))
    return out


def summary():
    """{blocked_by: {"n","wins","losses","win_rate"}} + open count -- 'was the block right?'"""
    rows = _all_rows()
    by = {}
    for r in rows:
        y = _label(r)
        b = r.get("blocked_by") or "Other"
        e = by.setdefault(b, {"n": 0, "wins": 0, "losses": 0, "win_rate": None})
        if y is None:
            continue
        e["n"] += 1
        e["wins" if y == 1 else "losses"] += 1
    for e in by.values():
        e["win_rate"] = round(100.0 * e["wins"] / e["n"], 1) if e["n"] else None
    return {"by_gate": by, "open": sum(1 for r in rows if r.get("status") == "OPEN"), "total": len(rows)}
