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
from app_logging import get_logger
logger = get_logger(__name__)

import sqlite3
import json
import requests
from datetime import datetime, timedelta

DB_NAME = "trade_learning.db"
TABLE = "ai_trade_setups"

CONFIDENCE_FLOOR = 32.0
CONFIDENCE_CEILING = 78.0
MIN_SAMPLES_FOR_LEARNING = 8
MIN_FACTORS_TRUE_TO_QUALIFY = 7
MAX_HOLD_MINUTES = 120
TRAIL_TRIGGER_PTS = 40      # once price moves this many pts in our favor, SL trails to breakeven (entry)

ALL_FACTOR_KEYS = [
    "main_signal_aligned", "ml_agrees", "level_pct_ge_65",
    "banknifty_no_divergence", "breadth_aligned", "global_aligned",
    "oi_aligned", "fvg_ob_confluence", "vwap_aligned",
    "htf_1h_aligned", "htf_15min_aligned", "round_number_level",
    "liquidity_sweep", "low_vix", "news_sentiment_aligned", "ladder_confluence_aligned",
    "sniper_setup_aligned", "away_from_max_pain", "global_research_aligned",
    "order_flow_aligned", "regime_aligned", "mtf_stack_aligned",
    "data_quality_pass", "strong_daily_level", "strong_opening_range_level",
    "entry_timing_safe", "no_opposite_transition", "immediate_reversal_risk_low",
    "nifty50_news_aligned", "nifty50_fundamentals_aligned",
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
    "global_research_aligned": "Structured global research agrees or is neutral",
    "away_from_max_pain": "Price moving away from Options Max Pain (less resistance)",
    "order_flow_aligned": "Live Nifty futures order-book imbalance aligned",
    "regime_aligned": "Market regime supports the trade direction/setup",
    "mtf_stack_aligned": "Multi-timeframe structure stack aligned",
    "data_quality_pass": "Live data/context quality passed",
    "strong_daily_level": "Strong previous-day / daily-close structural level",
    "strong_opening_range_level": "Strong opening-range structural level",
    "entry_timing_safe": "Entry timing is safe",
    "no_opposite_transition": "No immediate opposite trend transition",
    "immediate_reversal_risk_low": "Immediate post-entry reversal risk is low",
    "nifty50_news_aligned": "NIFTY 50-specific live news sentiment aligned (own RSS scoring)",
    "nifty50_fundamentals_aligned": "NIFTY 50 constituent-aggregate fundamentals tilt aligned/neutral",
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
    c.execute("SELECT id, timestamp, direction, underlying_entry, stop_loss, target, factors_json FROM ai_trade_setups WHERE status='OPEN'")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "timestamp": r[1], "direction": r[2], "underlying_entry": r[3],
             "stop_loss": r[4], "target": r[5]} for r in rows]


def _sqlite_update_factors(setup_id, factors_json):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE ai_trade_setups SET factors_json=? WHERE id=?", (factors_json, setup_id))
    conn.commit(); conn.close()


def _sqlite_update_sl(setup_id, new_sl):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE ai_trade_setups SET stop_loss=? WHERE id=?", (new_sl, setup_id))
    conn.commit()
    conn.close()


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
                        stop_loss, target, confidence_pct, status, exit_price, exit_timestamp,
                        factors_json
                 FROM ai_trade_setups ORDER BY id DESC LIMIT ?""", (limit,))
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        d = {"timestamp": r[0], "direction": r[1], "strike": r[2], "option_type": r[3],
             "entry": r[4], "stop_loss": r[5], "target": r[6], "confidence_pct": r[7],
             "status": r[8], "exit_price": r[9], "exit_timestamp": r[10]}
        d.update(_extract_entry_reasoning(r[11]))
        out.append(d)
    return out


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
    url = f"{_supabase_url}/rest/v1/{TABLE}?status=eq.OPEN&select=id,timestamp,direction,underlying_entry,stop_loss,target,factors_json"
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _sb_update_factors(setup_id, factors_json):
    url = f"{_supabase_url}/rest/v1/{TABLE}?id=eq.{setup_id}"
    resp = requests.patch(url, headers=_supabase_headers(), json={"factors_json": factors_json}, timeout=10)
    resp.raise_for_status()


def _sb_update_sl(setup_id, new_sl):
    url = f"{_supabase_url}/rest/v1/{TABLE}?id=eq.{setup_id}"
    resp = requests.patch(url, headers=_supabase_headers(),
                           json={"stop_loss": new_sl}, timeout=10)
    resp.raise_for_status()


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
           f"confidence_pct,status,exit_price,exit_timestamp,factors_json")
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    out = []
    for r in rows:
        d = {"timestamp": r["timestamp"], "direction": r["direction"], "strike": r["strike"],
             "option_type": r["option_type"], "entry": r["underlying_entry"], "stop_loss": r["stop_loss"],
             "target": r["target"], "confidence_pct": r["confidence_pct"], "status": r["status"],
             "exit_price": r["exit_price"], "exit_timestamp": r["exit_timestamp"]}
        d.update(_extract_entry_reasoning(r.get("factors_json")))
        out.append(d)
    return out


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


def _extract_entry_reasoning(factors_json):
    """
    Pull the human-readable "why this trade was taken" (playbook, score,
    top reasons) back out of the stored factors_json, so the Recent AI
    Setups table can show it for every PAST trade too -- not just the
    live moment it fired. This is what actually lets you verify, for any
    row, that it wasn't a random buy/sell.
    """
    try:
        factors = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
    except Exception:
        factors = {}
    return {
        "playbook": factors.get("_playbook_label", "-"),
        "setup_score": factors.get("_setup_score"),
        "reasons_short": " · ".join((factors.get("_reasons") or [])[:3]) or "-",
    }


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
            logger.exception("Broad exception caught; fallback path executed")
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
        logger.exception("Broad exception caught; fallback path executed")
        print(f"[trade_learning] Could not fetch open setups, using local fallback: {e}")
        open_rows = _sqlite_get_open()

    for row in open_rows:
        setup_id, ts, direction = row["id"], row["timestamp"], row["direction"]
        entry, sl, target = row["underlying_entry"], row["stop_loss"], row["target"]
        try:
            factors = json.loads(row.get("factors_json") or "{}") if isinstance(row, dict) else {}
        except Exception:
            factors = {}
        meta = factors.get("_learning_meta") if isinstance(factors.get("_learning_meta"), dict) else {}
        is_buy = direction == "BUY"
        favorable = (live_price - entry) if is_buy else (entry - live_price)
        adverse = (entry - live_price) if is_buy else (live_price - entry)
        meta["max_favorable_pts"] = round(max(float(meta.get("max_favorable_pts", 0.0)), float(favorable)), 2)
        meta["max_adverse_pts"] = round(max(float(meta.get("max_adverse_pts", 0.0)), float(adverse)), 2)
        meta["immediate_reversal_observed"] = bool(meta["max_adverse_pts"] >= max(0.35 * abs(entry - sl), 6.0))
        factors["_learning_meta"] = meta
        try:
            payload = json.dumps(factors)
            if _is_supabase_configured(): _sb_update_factors(setup_id, payload)
            else: _sqlite_update_factors(setup_id, payload)
        except Exception:
            logger.exception("Could not persist learning excursion metadata")

        # Trail SL to breakeven once TRAIL_TRIGGER_PTS moved in our favor,
        # so a target that never gets hit unwinds at ~0 instead of a full
        # loss -- but only ever tighten the SL, never loosen it.
        if favorable >= TRAIL_TRIGGER_PTS and not meta.get("trailed_to_breakeven"):
            new_sl = entry
            tightened = (new_sl > sl) if is_buy else (new_sl < sl)
            if tightened:
                try:
                    if _is_supabase_configured(): _sb_update_sl(setup_id, new_sl)
                    else: _sqlite_update_sl(setup_id, new_sl)
                    sl = new_sl
                    meta["trailed_to_breakeven"] = True
                    factors["_learning_meta"] = meta
                    payload = json.dumps(factors)
                    if _is_supabase_configured(): _sb_update_factors(setup_id, payload)
                    else: _sqlite_update_factors(setup_id, payload)
                except Exception:
                    logger.exception("Could not trail stop-loss to breakeven")

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
                logger.exception("Broad exception caught; fallback path executed")
                pass
        if outcome:
            try:
                if _is_supabase_configured():
                    _sb_close(setup_id, outcome, live_price)
                else:
                    _sqlite_close(setup_id, outcome, live_price)
            except Exception as e:
                logger.exception("Broad exception caught; fallback path executed")
                print(f"[trade_learning] Could not close setup {setup_id}: {e}")


def get_open_setup():
    if _is_supabase_configured():
        try:
            return _sb_get_latest_open_full()
        except Exception as e:
            logger.exception("Broad exception caught; fallback path executed")
            print(f"[trade_learning] Supabase read failed, using local fallback: {e}")
    return _sqlite_get_latest_open_full()


def get_recent_setups(limit=20):
    if _is_supabase_configured():
        try:
            return _sb_recent(limit)
        except Exception as e:
            logger.exception("Broad exception caught; fallback path executed")
            print(f"[trade_learning] Supabase read failed, using local fallback: {e}")
    return _sqlite_recent(limit)


def get_overall_track_record():
    try:
        counts = _sb_status_counts() if _is_supabase_configured() else _sqlite_status_counts()
    except Exception as e:
        logger.exception("Broad exception caught; fallback path executed")
        print(f"[trade_learning] Supabase read failed, using local fallback: {e}")
        counts = _sqlite_status_counts()
    wins, losses, expired = counts["WIN"], counts["LOSS"], counts["EXPIRED"]
    resolved = wins + losses
    win_rate = round((wins / resolved) * 100, 1) if resolved > 0 else None
    return {"wins": wins, "losses": losses, "expired": expired, "win_rate": win_rate, "sample_size": resolved}


def get_factor_reliability():
    rows = _resolved_learning_rows()

    result = {}
    for key in ALL_FACTOR_KEYS:
        true_wins, true_losses, true_total = 0, 0, 0
        for factors_json, status in rows:
            try:
                factors = json.loads(factors_json) if isinstance(factors_json, str) else factors_json
            except Exception:
                logger.exception("Broad exception caught; fallback path executed")
                continue
            if factors.get(key):
                true_total += 1
                if status == "WIN":
                    true_wins += 1
                elif status == "LOSS":
                    true_losses += 1
        result[key] = {
            "win_rate_when_true": round((true_wins / true_total) * 100, 1) if true_total >= MIN_SAMPLES_FOR_LEARNING else None,
            "samples_when_true": true_total,
            "wins_when_true": true_wins,
            "losses_when_true": true_losses,
            "label": FACTOR_LABELS[key],
        }
    return result


def _resolved_learning_rows():
    """Return resolved factor snapshots. Only WIN/LOSS records are used.

    EXPIRED trades are intentionally excluded: an expiry timeout is not a
    clean directional label and treating it as a loss would teach the engine
    the wrong lesson.

    A trade that trailed its SL to breakeven (moved 40+ pts in our favor,
    then came back to entry) is ALSO excluded from the LOSS side: the setup
    was right for a while but simply didn't reach the target this time --
    that is not the same failure as a clean stop-out, and counting it as a
    plain LOSS would wrongly punish the very factors that got it 40 pts in
    profit. It still shows as LOSS in the trade history/win-rate (real
    money-wise it was breakeven), but the self-learning ignores it.
    """
    try:
        rows = _sb_resolved_factors_status() if _is_supabase_configured() else _sqlite_resolved_factors_status()
    except Exception as e:
        logger.exception("Broad exception caught; fallback path executed")
        print(f"[trade_learning] learning read failed, using local fallback: {e}")
        rows = _sqlite_resolved_factors_status()

    clean = []
    for factors_json, status in rows:
        if status == "LOSS":
            try:
                factors = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
                meta = factors.get("_learning_meta") or {}
                if meta.get("trailed_to_breakeven"):
                    continue  # inconclusive breakeven exit -- not a clean loss signal
            except Exception:
                pass
        clean.append((factors_json, status))
    return clean


def get_learning_summary():
    """Human-readable state of the adaptive learner.

    The learner does not rewrite trading rules after one or two trades. It
    needs a minimum sample per factor, uses Laplace/Beta smoothing, and only
    changes the confidence weighting. This reduces overfitting to a tiny
    live sample while still allowing the engine to adapt automatically.
    """
    rows = _resolved_learning_rows()
    resolved = len(rows)
    learned = 0
    for key in ALL_FACTOR_KEYS:
        n = 0
        for factors_json, _status in rows:
            try:
                factors = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
            except Exception:
                logger.exception("Broad exception caught; fallback path executed")
                continue
            if key in factors:
                n += 1
        if n >= MIN_SAMPLES_FOR_LEARNING:
            learned += 1
    reversal_samples = reversal_wins = reversal_losses = 0
    for factors_json, status in rows:
        try:
            f = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
            if (f.get("_learning_meta") or {}).get("immediate_reversal_observed"):
                reversal_samples += 1
                reversal_wins += int(status == "WIN")
                reversal_losses += int(status == "LOSS")
        except Exception:
            continue
    return {
        "resolved_trades": resolved,
        "learned_factors": learned,
        "min_samples_per_factor": MIN_SAMPLES_FOR_LEARNING,
        "persistent": _is_supabase_configured(),
        "mode": "adaptive" if learned else "warming_up",
        "immediate_reversal": {"samples": reversal_samples, "wins": reversal_wins, "losses": reversal_losses,
                                "win_rate": round(100*reversal_wins/(reversal_wins+reversal_losses),1) if (reversal_wins+reversal_losses) else None},
    }


def _factor_bayesian_reliability(rows, key):
    """Smoothed P(WIN | factor=True) and P(WIN | factor=False).

    Beta(1,1) smoothing prevents a tiny sample from producing a 0%/100%
    reliability and therefore prevents a single lucky/unlucky trade from
    dominating future decisions.
    """
    true_wins = true_n = false_wins = false_n = 0
    for factors_json, status in rows:
        try:
            factors = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
        except Exception:
            logger.exception("Broad exception caught; fallback path executed")
            continue
        if key not in factors:
            continue
        if bool(factors.get(key)):
            true_n += 1
            true_wins += int(status == "WIN")
        else:
            false_n += 1
            false_wins += int(status == "WIN")
    # Beta(1,1) prior: posterior mean (wins+1)/(n+2).
    p_true = (true_wins + 1.0) / (true_n + 2.0) if true_n else None
    p_false = (false_wins + 1.0) / (false_n + 2.0) if false_n else None
    return p_true, true_n, p_false, false_n


def compute_confidence(factor_flags: dict):
    """Compute confidence using rule-based confluence + learned evidence.

    IMPORTANT: this is adaptive learning, not automatic rule mutation.
    Resolved WIN/LOSS outcomes update factor weights; the engine never
    changes its core safety gates (support/resistance, confirmation, R:R,
    missing-data handling) because of a single trade.
    """
    rows = _resolved_learning_rows()
    weighted = []
    learned_factor_count = 0

    # Base confluence: every aligned factor is evidence, but correlated
    # factors are not allowed to explode confidence.
    true_count = sum(bool(factor_flags.get(k)) for k in ALL_FACTOR_KEYS)
    base = 50.0 + min(18.0, true_count * 1.8)

    for key in ALL_FACTOR_KEYS:
        if not bool(factor_flags.get(key)):
            continue
        p_true, n_true, p_false, n_false = _factor_bayesian_reliability(rows, key)
        if p_true is None or n_true < MIN_SAMPLES_FOR_LEARNING:
            continue
        learned_factor_count += 1
        # Lift over the learned unconditional baseline. Clamp each factor so
        # one factor can never dominate the complete decision.
        baseline_parts = [p for p in (p_true, p_false) if p is not None]
        baseline = sum(baseline_parts) / len(baseline_parts) if baseline_parts else 0.5
        lift = max(-0.20, min(0.20, p_true - baseline))
        weighted.append(lift)

    if weighted:
        # Small adaptive correction; the rule-based confluence remains the
        # main driver and learned evidence is deliberately bounded.
        adaptive = (sum(weighted) / len(weighted)) * 35.0
        base += adaptive

    confidence = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, round(base, 1)))
    return confidence, learned_factor_count > 0, learned_factor_count
