"""
trade_learning.py — self-learning trade DECISION engine, separate from
the existing database.py trade log (left completely untouched).

STORAGE: uses Supabase (a free, permanent Postgres database) as the
primary backend, so trade history SURVIVES app restarts, redeploys, and
Streamlit Cloud's own sleep/wake cycles -- fixing the earlier problem
where local SQLite lived only in the app's temporary container and was
wiped every time that container was recreated.

If Supabase isn't configured (no secrets set), this module falls back
to local SQLite so the app doesn't crash -- but that fallback is NOT
persistent, and get_storage_status() reports that plainly so the UI can
warn about it instead of silently losing data again.

Everything else about this module is unchanged from before:
  1. Only speaks up when a real minimum bar of confluence is met.
  2. Reads every factor this tool computes.
  3. Logs every recommendation with its exact factor snapshot, then
     auto-resolves it (WIN/LOSS/EXPIRED) against live price.
  4. Learns, per factor, whether it has actually been predictive in
     THIS user's own resolved history -- with too little data, it says
     so instead of pretending to be confident.
  5. Confidence is capped well below 90% -- no tool can promise more.
"""

import sqlite3
import json
import requests
from datetime import datetime, timedelta

DB_NAME = "trade_learning.db"
TABLE = "ai_trade_setups"

CONFIDENCE_FLOOR = 32.0
CONFIDENCE_CEILING = 78.0
MIN_SAMPLES_FOR_LEARNING = 8
MIN_FACTORS_TRUE_TO_QUALIFY = 5
MAX_HOLD_MINUTES = 120

ALL_FACTOR_KEYS = [
    "main_signal_aligned", "ml_agrees", "level_pct_ge_65",
    "banknifty_no_divergence", "breadth_aligned", "global_aligned",
    "oi_aligned", "fvg_ob_confluence", "vwap_aligned",
    "htf_1h_aligned", "htf_15min_aligned", "round_number_level",
    "liquidity_sweep", "low_vix", "news_sentiment_aligned", "ladder_confluence_aligned",
    "sniper_setup_aligned", "away_from_max_pain",
]

FACTOR_LABELS = {
    "main_signal_aligned": "Main Institutional Confluence Signal",
    "ml_agrees": "ML Ensemble Model agrees",
    "level_pct_ge_65": "Nearest level Break/Bounce% >= 65",
    "banknifty_no_divergence": "Bank Nifty -- no divergence warning",
    "breadth_aligned": "Market breadth aligned",
    "global_aligned": "Global markets aligned",
    "oi_aligned": "Option-chain OI/PCR aligned",
    "fvg_ob_confluence": "FVG / Order Block at this level",
    "vwap_aligned": "VWAP bias aligned",
    "htf_1h_aligned": "1-Hour structure aligned",
    "htf_15min_aligned": "15-Minute structure aligned",
    "round_number_level": "Level is a round psychological number",
    "liquidity_sweep": "ICT liquidity sweep already seen at level",
    "low_vix": "India VIX calm (not an elevated-risk day)",
    "news_sentiment_aligned": "Live India news sentiment aligned (real RSS feed)",
    "ladder_confluence_aligned": "2+ of next 3 round-number ladder levels also agree",
    "sniper_setup_aligned": "Sniper Setup (PDH/PDL/CPR + SMC + OI) aligned",
    "away_from_max_pain": "Price moving away from Options Max Pain (less resistance)",
}

# ---------------------------------------------------------------------
# STORAGE CONFIGURATION
# ---------------------------------------------------------------------
_supabase_url = None
_supabase_key = None


def configure_supabase(url, key):
    """Call ONCE at app startup (from app.py, right after reading
    st.secrets) to switch this module onto permanent Supabase storage.
    If never called (or called with empty values), this module quietly
    keeps using local SQLite -- see get_storage_status()."""
    global _supabase_url, _supabase_key
    if url and key:
        _supabase_url = url.rstrip("/")
        _supabase_key = key


def _is_supabase_configured():
    return bool(_supabase_url and _supabase_key)


def get_storage_status():
    """Returns (is_persistent: bool, message: str) -- app.py uses this to
    show an honest banner instead of silently risking data loss again."""
    if _is_supabase_configured():
        return True, "Supabase se connected -- trade history permanently safe hai, app restart/redeploy se bhi nahi mitegi."
    return False, ("⚠️ Persistent storage configure nahi hai -- trade history sirf is session ke liye hai, "
                    "app restart ya redeploy hone par MIT JAAYEGI. Supabase setup karo isse permanent banane ke liye.")


def _supabase_headers():
    return {
        "apikey": _supabase_key,
        "Authorization": f"Bearer {_supabase_key}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------
# LOCAL SQLITE BACKEND (fallback only -- NOT persistent across restarts)
# ---------------------------------------------------------------------
def _sqlite_init():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ai_trade_setups
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT, direction TEXT, strike INTEGER, option_type TEXT,
                  underlying_entry REAL, stop_loss REAL, target REAL,
                  confidence_pct REAL, factors_json TEXT,
                  status TEXT DEFAULT 'OPEN', exit_price REAL, exit_timestamp TEXT)''')
    conn.commit()
    conn.close()


def _sqlite_insert(row):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""INSERT INTO ai_trade_setups
                 (timestamp, direction, strike, option_type, underlying_entry,
                  stop_loss, target, confidence_pct, factors_json, status)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
              (row["timestamp"], row["direction"], row["strike"], row["option_type"],
               row["underlying_entry"], row["stop_loss"], row["target"],
               row["confidence_pct"], row["factors_json"]))
    setup_id = c.lastrowid
    conn.commit()
    conn.close()
    return setup_id


def _sqlite_open_signature():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT direction, strike FROM ai_trade_setups WHERE status='OPEN' ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row


def _sqlite_get_open():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, direction, underlying_entry, stop_loss, target FROM ai_trade_setups WHERE status='OPEN'")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "timestamp": r[1], "direction": r[2], "underlying_entry": r[3],
             "stop_loss": r[4], "target": r[5]} for r in rows]


def _sqlite_close(setup_id, status, exit_price):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE ai_trade_setups SET status=?, exit_price=?, exit_timestamp=? WHERE id=?",
              (status, exit_price, now, setup_id))
    conn.commit()
    conn.close()


def _sqlite_get_latest_open_full():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""SELECT id, timestamp, direction, strike, option_type, underlying_entry,
                        stop_loss, target, confidence_pct
                 FROM ai_trade_setups WHERE status='OPEN' ORDER BY id DESC LIMIT 1""")
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "timestamp": row[1], "direction": row[2], "strike": row[3],
            "option_type": row[4], "underlying_entry": row[5], "stop_loss": row[6],
            "target": row[7], "confidence_pct": row[8]}


def _sqlite_recent(limit):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""SELECT timestamp, direction, strike, option_type, underlying_entry,
                        stop_loss, target, confidence_pct, status, exit_price, exit_timestamp
                 FROM ai_trade_setups ORDER BY id DESC LIMIT ?""", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"timestamp": r[0], "direction": r[1], "strike": r[2], "option_type": r[3],
              "entry": r[4], "stop_loss": r[5], "target": r[6], "confidence_pct": r[7],
              "status": r[8], "exit_price": r[9], "exit_timestamp": r[10]} for r in rows]


def _sqlite_resolved_factors_status():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT factors_json, status FROM ai_trade_setups WHERE status IN ('WIN','LOSS')")
    rows = c.fetchall()
    conn.close()
    return rows


def _sqlite_status_counts():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    counts = {}
    for s in ("WIN", "LOSS", "EXPIRED"):
        c.execute("SELECT COUNT(*) FROM ai_trade_setups WHERE status=?", (s,))
        counts[s] = c.fetchone()[0]
    conn.close()
    return counts


# ---------------------------------------------------------------------
# SUPABASE BACKEND (permanent -- survives restarts/redeploys)
# ---------------------------------------------------------------------
def _sb_insert(row):
    url = f"{_supabase_url}/rest/v1/{TABLE}"
    resp = requests.post(url, headers={**_supabase_headers(), "Prefer": "return=representation"},
                          json=row, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[0]["id"] if data else None


def _sb_open_signature():
    url = f"{_supabase_url}/rest/v1/{TABLE}?status=eq.OPEN&order=id.desc&limit=1&select=direction,strike"
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return (data[0]["direction"], data[0]["strike"]) if data else None


def _sb_get_open():
    url = f"{_supabase_url}/rest/v1/{TABLE}?status=eq.OPEN&select=id,timestamp,direction,underlying_entry,stop_loss,target"
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _sb_close(setup_id, status, exit_price):
    url = f"{_supabase_url}/rest/v1/{TABLE}?id=eq.{setup_id}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resp = requests.patch(url, headers=_supabase_headers(),
                           json={"status": status, "exit_price": exit_price, "exit_timestamp": now}, timeout=10)
    resp.raise_for_status()


def _sb_get_latest_open_full():
    url = (f"{_supabase_url}/rest/v1/{TABLE}?status=eq.OPEN&order=id.desc&limit=1"
           f"&select=id,timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,confidence_pct")
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


def _sb_recent(limit):
    url = (f"{_supabase_url}/rest/v1/{TABLE}?order=id.desc&limit={limit}"
           f"&select=timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,"
           f"confidence_pct,status,exit_price,exit_timestamp")
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    return [{"timestamp": r["timestamp"], "direction": r["direction"], "strike": r["strike"],
             "option_type": r["option_type"], "entry": r["underlying_entry"], "stop_loss": r["stop_loss"],
             "target": r["target"], "confidence_pct": r["confidence_pct"], "status": r["status"],
             "exit_price": r["exit_price"], "exit_timestamp": r["exit_timestamp"]} for r in rows]


def _sb_resolved_factors_status():
    url = f"{_supabase_url}/rest/v1/{TABLE}?status=in.(WIN,LOSS)&select=factors_json,status"
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    return [(r["factors_json"], r["status"]) for r in resp.json()]


def _sb_status_counts():
    url = f"{_supabase_url}/rest/v1/{TABLE}?select=status"
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    counts = {"WIN": 0, "LOSS": 0, "EXPIRED": 0}
    for r in resp.json():
        if r["status"] in counts:
            counts[r["status"]] += 1
    return counts


# ---------------------------------------------------------------------
# PUBLIC API — same signatures regardless of backend; every function
# below tries Supabase first (if configured), and transparently falls
# back to local SQLite only on genuine failure or when not configured,
# so a temporary network hiccup doesn't crash the dashboard.
# ---------------------------------------------------------------------
def init_db():
    _sqlite_init()  # always available as the fallback path


def log_setup(direction, strike, option_type, underlying_entry, stop_loss, target,
              confidence_pct, factor_flags: dict):
    factors_json = json.dumps(factor_flags)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if _is_supabase_configured():
        try:
            existing = _sb_open_signature()
            if existing is not None and existing[0] == direction and existing[1] == strike:
                return None
            row = {"timestamp": timestamp, "direction": direction, "strike": strike,
                   "option_type": option_type, "underlying_entry": underlying_entry,
                   "stop_loss": stop_loss, "target": target, "confidence_pct": confidence_pct,
                   "factors_json": factors_json, "status": "OPEN"}
            return _sb_insert(row)
        except Exception as e:
            print(f"[trade_learning] Supabase insert failed, falling back to local: {e}")

    existing = _sqlite_open_signature()
    if existing is not None and existing[0] == direction and existing[1] == strike:
        return None
    return _sqlite_insert({"timestamp": timestamp, "direction": direction, "strike": strike,
                            "option_type": option_type, "underlying_entry": underlying_entry,
                            "stop_loss": stop_loss, "target": target,
                            "confidence_pct": confidence_pct, "factors_json": factors_json})


def resolve_open_setups(live_price):
    """Call every refresh with the current underlying price."""
    try:
        open_rows = _sb_get_open() if _is_supabase_configured() else _sqlite_get_open()
    except Exception as e:
        print(f"[trade_learning] Could not fetch open setups, using local fallback: {e}")
        open_rows = _sqlite_get_open()

    for row in open_rows:
        setup_id, ts, direction = row["id"], row["timestamp"], row["direction"]
        entry, sl, target = row["underlying_entry"], row["stop_loss"], row["target"]
        is_buy = direction == "BUY"
        outcome = None
        if is_buy:
            if live_price >= target:
                outcome = "WIN"
            elif live_price <= sl:
                outcome = "LOSS"
        else:
            if live_price <= target:
                outcome = "WIN"
            elif live_price >= sl:
                outcome = "LOSS"
        if outcome is None:
            try:
                opened_at = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                if datetime.now() - opened_at > timedelta(minutes=MAX_HOLD_MINUTES):
                    outcome = "EXPIRED"
            except Exception:
                pass
        if outcome:
            try:
                if _is_supabase_configured():
                    _sb_close(setup_id, outcome, live_price)
                else:
                    _sqlite_close(setup_id, outcome, live_price)
            except Exception as e:
                print(f"[trade_learning] Could not close setup {setup_id}: {e}")


def get_open_setup():
    if _is_supabase_configured():
        try:
            return _sb_get_latest_open_full()
        except Exception as e:
            print(f"[trade_learning] Supabase read failed, using local fallback: {e}")
    return _sqlite_get_latest_open_full()


def get_recent_setups(limit=20):
    if _is_supabase_configured():
        try:
            return _sb_recent(limit)
        except Exception as e:
            print(f"[trade_learning] Supabase read failed, using local fallback: {e}")
    return _sqlite_recent(limit)


def get_overall_track_record():
    try:
        counts = _sb_status_counts() if _is_supabase_configured() else _sqlite_status_counts()
    except Exception as e:
        print(f"[trade_learning] Supabase read failed, using local fallback: {e}")
        counts = _sqlite_status_counts()
    wins, losses, expired = counts["WIN"], counts["LOSS"], counts["EXPIRED"]
    resolved = wins + losses
    win_rate = round((wins / resolved) * 100, 1) if resolved > 0 else None
    return {"wins": wins, "losses": losses, "expired": expired, "win_rate": win_rate, "sample_size": resolved}


def get_factor_reliability():
    try:
        rows = _sb_resolved_factors_status() if _is_supabase_configured() else _sqlite_resolved_factors_status()
    except Exception as e:
        print(f"[trade_learning] Supabase read failed, using local fallback: {e}")
        rows = _sqlite_resolved_factors_status()

    result = {}
    for key in ALL_FACTOR_KEYS:
        true_wins, true_total = 0, 0
        for factors_json, status in rows:
            try:
                factors = json.loads(factors_json) if isinstance(factors_json, str) else factors_json
            except Exception:
                continue
            if factors.get(key):
                true_total += 1
                if status == "WIN":
                    true_wins += 1
        if true_total >= MIN_SAMPLES_FOR_LEARNING:
            result[key] = {"win_rate_when_true": round((true_wins / true_total) * 100, 1),
                            "samples_when_true": true_total, "label": FACTOR_LABELS[key]}
        else:
            result[key] = {"win_rate_when_true": None, "samples_when_true": true_total,
                            "label": FACTOR_LABELS[key]}
    return result


def compute_confidence(factor_flags: dict):
    """
    Blends rule-based scoring with LEARNED factor reliabilities wherever
    enough resolved history exists for that specific factor. Result is
    always clamped to [CONFIDENCE_FLOOR, CONFIDENCE_CEILING].
    Returns (confidence_pct, used_learning: bool, learned_factor_count: int)
    """
    reliability = get_factor_reliability()
    weighted_sum = 0.0
    weight_total = 0.0
    learned_factor_count = 0

    for key in ALL_FACTOR_KEYS:
        is_true = bool(factor_flags.get(key))
        rel = reliability[key]
        if rel["win_rate_when_true"] is not None:
            learned_factor_count += 1
            lift = (rel["win_rate_when_true"] - 50.0) / 50.0
            weighted_sum += (lift if is_true else 0.0)
            weight_total += 1.0
        else:
            weighted_sum += (0.5 if is_true else 0.0)
            weight_total += 1.0

    base_pct = 50.0 + (weighted_sum / weight_total) * 40.0 if weight_total > 0 else 50.0
    confidence = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, round(base_pct, 1)))
    return confidence, learned_factor_count > 0, learned_factor_count
