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
from datetime import datetime, timedelta, time as _dtime

import edge_model

try:
    from zoneinfo import ZoneInfo as _ZI
    _IST = _ZI("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = None


def _now_ist():
    """Naive IST wall-clock.  Streamlit Cloud runs in UTC; trade timestamps, expiry and the
    end-of-day square-off must all use the exchange clock or they drift by 5.5 hours."""
    return datetime.now(_IST).replace(tzinfo=None) if _IST is not None else datetime.now()


def _now_str():
    return _now_ist().strftime("%Y-%m-%d %H:%M:%S")


DB_NAME = "trade_learning.db"
TABLE = "ai_trade_setups"

CONFIDENCE_FLOOR = 32.0
CONFIDENCE_CEILING = 78.0
MIN_SAMPLES_FOR_LEARNING = 8
MIN_FACTORS_TRUE_TO_QUALIFY = 7
MAX_HOLD_MINUTES = 120
TRAIL_TRIGGER_PTS = 40      # (unused: trailing stays disabled, see resolve_open_setups)
EOD_SQUAREOFF = _dtime(15, 20)   # open trades are closed (EXPIRED, mark-to-market) at this IST time
TIME_STOP_MIN = 60               # a trade that has NOT moved in our favour by this time is closed (frees the slot for a better setup)
TIME_STOP_MIN_FAV_R = 0.35       # ... 'moved in our favour' = best excursion >= this many R
EXPIRED_LEARN_MIN_R = 0.5        # an EXPIRED trade teaches the learner only if it moved >= this many R
LEARNING_WINDOW = 80             # most recent resolved trades used by the edge model
COOLDOWN_AFTER_LOSS_MIN = 10     # no re-entry in the SAME direction this soon after a stop-out (~2 candles)
LOSS_STREAK_PAUSE = None         # None = off.  Set e.g. 3 to pause after that many straight losses today ...
LOSS_STREAK_PAUSE_MIN = 45       # ... pause new entries for this long

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
    "nifty50_news_aligned", "nifty50_fundamentals_aligned", "external_ai_aligned",
    "entry_not_extended", "entry_confirmed_bounce", "pcr_velocity_aligned",
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
    "external_ai_aligned": "Independent Gemini dashboard research agrees",
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
             "stop_loss": r[4], "target": r[5], "factors_json": r[6]} for r in rows]


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
    now = _now_str()
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
    c.execute("SELECT factors_json, status FROM ai_trade_setups WHERE status IN ('WIN','LOSS','EXPIRED') ORDER BY id ASC")
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
# SPEED FIX: the 30s refresh used to fire ~8-10 separate Supabase GETs (each
# with a 10s timeout) every cycle. Reads are now cached for a few seconds and
# the cache is cleared immediately after ANY write (insert/update/close), so the
# engine never sees stale open-trade state after it changes something.
import time as _sb_time
import threading as _sb_threading
_SB_READ_CACHE = {}
_SB_READ_LOCK = _sb_threading.Lock()
_SB_READ_TTL = 12


def _sb_cache_clear():
    with _SB_READ_LOCK:
        _SB_READ_CACHE.clear()


def _sb_read_cached(fn):
    def wrapper(*args, **kwargs):
        key = (fn.__name__, args, tuple(sorted(kwargs.items())))
        now = _sb_time.time()
        with _SB_READ_LOCK:
            hit = _SB_READ_CACHE.get(key)
            if hit and now - hit[0] < _SB_READ_TTL:
                return hit[1]
        value = fn(*args, **kwargs)
        with _SB_READ_LOCK:
            _SB_READ_CACHE[key] = (now, value)
        return value
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _sb_insert(row):
    url = f"{_supabase_url}/rest/v1/{TABLE}"
    resp = requests.post(url, headers={**_supabase_headers(), "Prefer": "return=representation"},
                          json=row, timeout=10)
    resp.raise_for_status()
    _sb_cache_clear()
    data = resp.json()
    return data[0]["id"] if data else None


@_sb_read_cached
def _sb_open_signature():
    url = f"{_supabase_url}/rest/v1/{TABLE}?status=eq.OPEN&order=id.desc&limit=1&select=direction,strike"
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return (data[0]["direction"], data[0]["strike"]) if data else None


@_sb_read_cached
def _sb_get_open():
    url = f"{_supabase_url}/rest/v1/{TABLE}?status=eq.OPEN&select=id,timestamp,direction,underlying_entry,stop_loss,target,factors_json"
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _sb_update_factors(setup_id, factors_json):
    url = f"{_supabase_url}/rest/v1/{TABLE}?id=eq.{setup_id}"
    resp = requests.patch(url, headers=_supabase_headers(), json={"factors_json": factors_json}, timeout=10)
    resp.raise_for_status()
    _sb_cache_clear()


def _sb_update_sl(setup_id, new_sl):
    url = f"{_supabase_url}/rest/v1/{TABLE}?id=eq.{setup_id}"
    resp = requests.patch(url, headers=_supabase_headers(),
                           json={"stop_loss": new_sl}, timeout=10)
    resp.raise_for_status()
    _sb_cache_clear()


def _sb_close(setup_id, status, exit_price):
    url = f"{_supabase_url}/rest/v1/{TABLE}?id=eq.{setup_id}"
    now = _now_str()
    resp = requests.patch(url, headers=_supabase_headers(),
                           json={"status": status, "exit_price": exit_price, "exit_timestamp": now}, timeout=10)
    resp.raise_for_status()
    _sb_cache_clear()


@_sb_read_cached
def _sb_get_latest_open_full():
    url = (f"{_supabase_url}/rest/v1/{TABLE}?status=eq.OPEN&order=id.desc&limit=1"
           f"&select=id,timestamp,direction,strike,option_type,underlying_entry,stop_loss,target,confidence_pct")
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


@_sb_read_cached
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


@_sb_read_cached
def _sb_resolved_factors_status():
    url = f"{_supabase_url}/rest/v1/{TABLE}?status=in.(WIN,LOSS,EXPIRED)&order=id.asc&select=factors_json,status"
    resp = requests.get(url, headers=_supabase_headers(), timeout=10)
    resp.raise_for_status()
    return [(r["factors_json"], r["status"]) for r in resp.json()]


@_sb_read_cached
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
        "probe": bool(factors.get("_probe")),
        # Gemini review that approved this trade (None for trades logged before v37.3 -- not recorded then)
        "gemini": factors.get("_gemini") if isinstance(factors.get("_gemini"), dict) else None,
        # v18: which decision-logic version produced this trade. Used by recent_performance()
        # so a bad streak from an OLD, since-changed logic can never keep penalising a NEWER one.
        "logic_version": (factors.get("_entry_meta") or {}).get("logic_version"),
    }


def log_setup(direction, strike, option_type, underlying_entry, stop_loss, target,
              confidence_pct, factor_flags: dict):
    factors_json = json.dumps(factor_flags)
    timestamp = _now_str()
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


def _scan_bars(is_buy, sl, target, bars):
    """Walk completed candles (oldest first) and return (outcome, exit_price) for the first
    candle that touched SL or target, else (None, None).

    `bars` = list of (high, low).  If ONE candle touched BOTH levels we cannot know the order,
    so it is scored as a LOSS (pessimistic) -- the tracker must never flatter itself.
    The old logic only compared the 30-second snapshot price with SL/target, so a spike that hit
    the stop and came back between two refreshes was silently ignored.
    """
    for hi, lo in bars:
        try:
            hi, lo = float(hi), float(lo)
        except Exception:
            continue
        if is_buy:
            hit_sl, hit_tp = lo <= sl, hi >= target
        else:
            hit_sl, hit_tp = hi >= sl, lo <= target
        if hit_sl:                       # both-in-one-candle also lands here (pessimistic)
            return "LOSS", sl
        if hit_tp:
            return "WIN", target
    return None, None


def _bars_since(df, opened_at):
    """(high, low) of fully-formed candles that OPENED after the trade did.  Index may be tz-aware."""
    try:
        if df is None or len(df) == 0 or "High" not in df.columns or "Low" not in df.columns:
            return []
        idx = df.index
        try:
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
        except Exception:
            pass
        out = []
        # the last row is the still-forming candle: the live price check already covers it
        for t, hi, lo in list(zip(idx, df["High"], df["Low"]))[:-1]:
            try:
                tt = t.to_pydatetime()
                if tt > opened_at and tt.date() == opened_at.date():   # only the trade's own session
                    out.append((hi, lo))
            except Exception:
                continue
        return out
    except Exception:
        logger.exception("bars_since failed")
        return []


def resolve_open_setups(live_price, df=None):
    """Call every refresh with the current underlying price (and, ideally, the OHLC frame).

    Outcome rules (unchanged philosophy, now actually correct):
      * WIN / LOSS  -- target / original SL touched.  With `df`, candle highs/lows since entry are
                       scanned so intrabar touches between refreshes are not missed.
      * EXPIRED     -- MAX_HOLD_MINUTES elapsed, or the EOD square-off time passed, or the trade
                       survived into another day.  Stored with its mark-to-market R so the
                       learner can still learn from it (see _resolved_learning_rows).
    No trailing / breakeven stop: SL stays at the original level.
    """
    try:
        open_rows = _sb_get_open() if _is_supabase_configured() else _sqlite_get_open()
    except Exception as e:
        logger.exception("Broad exception caught; fallback path executed")
        print(f"[trade_learning] Could not fetch open setups, using local fallback: {e}")
        open_rows = _sqlite_get_open()

    now = _now_ist()
    for row in open_rows:
        setup_id, ts, direction = row["id"], row["timestamp"], row["direction"]
        entry, sl, target = row["underlying_entry"], row["stop_loss"], row["target"]
        try:
            factors = json.loads(row.get("factors_json") or "{}") if isinstance(row, dict) else {}
        except Exception:
            factors = {}
        if not isinstance(factors, dict):
            factors = {}
        meta = factors.get("_learning_meta") if isinstance(factors.get("_learning_meta"), dict) else {}
        is_buy = direction == "BUY"
        risk = max(abs(float(entry) - float(sl)), 1e-9)
        try:
            opened_at = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            opened_at = None

        favorable = (live_price - entry) if is_buy else (entry - live_price)
        adverse = (entry - live_price) if is_buy else (live_price - entry)
        meta["max_favorable_pts"] = round(max(float(meta.get("max_favorable_pts", 0.0)), float(favorable)), 2)
        meta["max_adverse_pts"] = round(max(float(meta.get("max_adverse_pts", 0.0)), float(adverse)), 2)
        meta["immediate_reversal_observed"] = bool(meta["max_adverse_pts"] >= max(0.35 * abs(entry - sl), 6.0))

        # 1) completed candles since entry (catches intrabar SL/target touches)
        outcome, exit_px = (None, None)
        if opened_at is not None:
            _bars = _bars_since(df, opened_at)
            # best / worst excursion must include candle highs/lows, not only the 30 s snapshots
            for _hi, _lo in _bars:
                try:
                    _fav = (float(_hi) - entry) if is_buy else (entry - float(_lo))
                    _adv = (entry - float(_lo)) if is_buy else (float(_hi) - entry)
                    meta["max_favorable_pts"] = round(max(float(meta["max_favorable_pts"]), _fav), 2)
                    meta["max_adverse_pts"] = round(max(float(meta["max_adverse_pts"]), _adv), 2)
                except Exception:
                    pass
            outcome, exit_px = _scan_bars(is_buy, sl, target, _bars)
        # 2) a trade that survived into another session is closed at the last price seen in ITS session --
        #    today's gap-open price is not a price the trade could ever have been filled at
        carried = opened_at is not None and opened_at.date() != now.date()
        if outcome is None and carried:
            outcome = "EXPIRED"
            exit_px = float(meta.get("last_seen_price", entry))
            meta["expired_reason"] = "carried_overnight"
        # 3) the live price right now
        if outcome is None:
            if is_buy:
                if live_price >= target: outcome, exit_px = "WIN", live_price
                elif live_price <= sl: outcome, exit_px = "LOSS", live_price
            else:
                if live_price <= target: outcome, exit_px = "WIN", live_price
                elif live_price >= sl: outcome, exit_px = "LOSS", live_price
        # 4) time based exits (same session)
        if outcome is None and opened_at is not None:
            stagnant = ((now - opened_at) >= timedelta(minutes=TIME_STOP_MIN)
                        and float(meta.get("max_favorable_pts", 0.0)) < TIME_STOP_MIN_FAV_R * risk)
            if now.time() >= EOD_SQUAREOFF or (now - opened_at) > timedelta(minutes=MAX_HOLD_MINUTES) or stagnant:
                outcome = "EXPIRED"
                exit_px = live_price
                meta["expired_reason"] = ("eod_squareoff" if now.time() >= EOD_SQUAREOFF
                                          else "max_hold" if (now - opened_at) > timedelta(minutes=MAX_HOLD_MINUTES)
                                          else "time_stop_no_progress")
        if outcome is None:
            meta["last_seen_price"] = round(float(live_price), 2)   # used if the trade is ever carried overnight
        if outcome == "EXPIRED":
            pnl = (exit_px - entry) if is_buy else (entry - exit_px)
            meta["expired_r"] = round(pnl / risk, 3)

        factors["_learning_meta"] = meta
        try:
            payload = json.dumps(factors)
            if _is_supabase_configured(): _sb_update_factors(setup_id, payload)
            else: _sqlite_update_factors(setup_id, payload)
        except Exception:
            logger.exception("Could not persist learning excursion metadata")

        if outcome:
            try:
                if _is_supabase_configured():
                    _sb_close(setup_id, outcome, exit_px)
                else:
                    _sqlite_close(setup_id, outcome, exit_px)
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
    # Honest rate: EXPIRED trades that ended clearly red / green are included (see _resolved_learning_rows)
    lab = _resolved_learning_rows()
    lw = sum(1 for _, st in lab if st == "WIN")
    ln = len(lab)
    return {"wins": wins, "losses": losses, "expired": expired, "win_rate": win_rate, "sample_size": resolved,
            "win_rate_incl_expired": round(100.0 * lw / ln, 1) if ln else None, "sample_incl_expired": ln}


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
    """Resolved factor snapshots as (factors_json, 'WIN'|'LOSS').

    * WIN / LOSS are used as they are.
    * EXPIRED trades are no longer thrown away.  Throwing them away made the win rate and the
      learner look better than reality (a trade that drifted -0.8R and timed out simply vanished).
      An EXPIRED trade is labelled by its mark-to-market result: >= +EXPIRED_LEARN_MIN_R -> WIN,
      <= -EXPIRED_LEARN_MIN_R -> LOSS, anything in between is genuinely inconclusive and skipped.
      Old EXPIRED rows without a stored R are skipped.
    """
    try:
        rows = _sb_resolved_factors_status() if _is_supabase_configured() else _sqlite_resolved_factors_status()
    except Exception as e:
        logger.exception("Broad exception caught; fallback path executed")
        print(f"[trade_learning] learning read failed, using local fallback: {e}")
        rows = _sqlite_resolved_factors_status()

    clean = []
    for factors_json, status in rows:
        try:
            factors = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
            meta = factors.get("_learning_meta") or {}
        except Exception:
            factors, meta = {}, {}
        if status == "EXPIRED":
            r = meta.get("expired_r")
            if r is None:
                continue
            if r >= EXPIRED_LEARN_MIN_R:
                status = "WIN"
            elif r <= -EXPIRED_LEARN_MIN_R:
                status = "LOSS"
            else:
                continue
        elif status == "LOSS" and meta.get("trailed_to_breakeven"):
            continue  # legacy rows from the old breakeven-trail version: inconclusive
        clean.append((factors_json, status))
    return clean


def _labelled_flag_rows(with_version=False):
    """[(flag_dict_without_private_keys, 1|0)] for the edge model (or triples with the logic_version when asked)."""
    out = []
    for factors_json, status in _resolved_learning_rows():
        if status not in ("WIN", "LOSS"):
            continue
        try:
            f = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
        except Exception:
            continue
        if not isinstance(f, dict):
            continue
        flags = {k: v for k, v in f.items() if not str(k).startswith("_")}
        label = 1 if status == "WIN" else 0
        if with_version:
            out.append((flags, label, (f.get("_entry_meta") or {}).get("logic_version")))
        else:
            out.append((flags, label))
    # Rolling window: only the most recent trades describe how the market behaves NOW.
    return out[-LEARNING_WINDOW:]


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


def _labelled_rows_full(include_shadow=True):
    """Everything the learners may study, OLDEST first: [{flags, label, weight, version, feats, source}].
    Real trades weigh 1.0.  Resolved SHADOW (paper) trades -- setups that were blocked but followed anyway -- weigh
    shadow_trades.SHADOW_WEIGHT.  Shadow rows come first so the newest real trades form the out-of-sample test set."""
    out = []
    if include_shadow:
        try:
            import shadow_trades
            for flags, y, feats, blocked_by in shadow_trades.labelled_rows()[-150:]:
                out.append({"flags": flags, "label": y, "weight": shadow_trades.SHADOW_WEIGHT, "version": LOGIC_VERSION,
                            "feats": feats, "source": "shadow", "blocked_by": blocked_by})
        except Exception:
            logger.exception("shadow rows unavailable (ignored)")
    real = []
    for factors_json, status in _resolved_learning_rows():
        if status not in ("WIN", "LOSS"):
            continue
        try:
            f = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
        except Exception:
            continue
        if not isinstance(f, dict):
            continue
        real.append({"flags": {k: v for k, v in f.items() if not str(k).startswith("_")}, "label": 1 if status == "WIN" else 0,
                     "weight": 1.0, "version": (f.get("_entry_meta") or {}).get("logic_version"),
                     "feats": f.get("_features") if isinstance(f.get("_features"), dict) else None, "source": "real"})
    return out + real[-LEARNING_WINDOW:]


def compute_edge(factor_flags: dict, rr: float, extra_penalty_logit: float = 0.0):
    """Calibrated win probability + expectancy (in R) for one candidate -- see edge_model.py.
    Factor lifts learn from all recent history (real + shadow); the win-rate calibration uses ONLY trades made by the CURRENT
    decision logic (LOGIC_VERSION) plus shadow trades, so an older logic's losing record can not freeze the new one."""
    full = _labelled_rows_full()
    rows = [(r["flags"], r["label"], r["weight"]) for r in full]
    calib = [(r["flags"], r["label"], r["weight"]) for r in full if r["version"] == LOGIC_VERSION or r["source"] == "shadow"]
    return edge_model.estimate_edge(factor_flags, rows, ALL_FACTOR_KEYS, rr=rr,
                                    extra_penalty_logit=extra_penalty_logit, calib_rows=calib)


def learned_adjustment(flags: dict, feats: dict):
    """Log-odds delta from the mistake memory + the self-validated neural net (see learner_nn.py).  Never raises."""
    try:
        import learner_nn
        return learner_nn.learned_adjustment(flags, feats, ALL_FACTOR_KEYS, _labelled_rows_full())
    except Exception:
        logger.exception("learned_adjustment failed (ignored)")
        return {"delta_logit": 0.0, "note": "", "nn": {}, "memory": {}}


def compute_confidence(factor_flags: dict):
    """Backwards-compatible wrapper: (confidence_pct, used_learning, learned_count).

    The number is now a calibrated win probability (not `50 + 1.8 * factors`).  Uses RR=1.5 only
    to fill the expectancy field; callers that know the real R:R should call compute_edge().
    """
    e = compute_edge(factor_flags, rr=1.5)
    conf = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, e["confidence_pct"]))
    return conf, e["learned_count"] > 0, e["learned_count"]


# =====================================================================
# SEGMENT LEARNING -- the engine learns WHICH kind of trade works.
# ---------------------------------------------------------------------
# Factor-level learning (above) says "is factor X predictive".  This adds a second
# layer: every trade is stored with an `_entry_meta` (playbook, level grade, time-of-day
# bucket, side, probe or not).  Once a segment has enough resolved trades and keeps
# losing, the engine stops taking that segment by itself; segments that win stay open.
# It only ever BLOCKS proven losers -- it never lowers a safety gate.
# =====================================================================
import time as _time

SEGMENT_MIN_SAMPLES = 8          # resolved trades before a single segment can be judged
SEGMENT_COMBO_MIN_SAMPLES = 6    # playbook+grade combo needs fewer (it is more specific)
SEGMENT_BLOCK_WINRATE = 30.0     # below this win-rate (with enough samples) the segment is paused
PROBE_MIN_SAMPLES = 6
PROBE_PAUSE_WINRATE = 35.0       # probes pause themselves below this win-rate
RECENT_WINDOW = 10               # last N resolved non-probe trades define "current form"
RECENT_BAD_WINRATE = 35.0
FORM_STALE_HOURS = 30            # v17: a 'bad form' streak older than this (wall-clock) stops penalising new entries
# v18: bump this any time the entry-gating logic changes materially. recent_performance() only
# counts trades tagged with the CURRENT version -- an old logic's loss streak must never keep
# tightening/penalising a logic that has since been rewritten (that was making the bar go up
# and stay up on stale, no-longer-relevant history).
LOGIC_VERSION = "v36_calibrated_edge"
MIN_NEW_LOGIC_SAMPLES = 6        # need at least this many CURRENT-logic resolved trades before 'bad form' can fire
_SEG_CACHE = {"ts": 0.0, "stats": None}
_SEG_CACHE_SECONDS = 60


def _entry_meta_rows():
    out = []
    for factors_json, status in _resolved_learning_rows():
        if status not in ("WIN", "LOSS"):
            continue
        try:
            f = json.loads(factors_json) if isinstance(factors_json, str) else (factors_json or {})
        except Exception:
            continue
        meta = f.get("_entry_meta")
        if isinstance(meta, dict):
            out.append((meta, status))
    return out


def get_segment_stats(force=False):
    """{'playbook=BOUNCE_BUY': {n,wins,losses,win_rate}, 'grade=A': ..., 'combo=BOUNCE_BUY|A': ...}"""
    now = _time.time()
    if not force and _SEG_CACHE["stats"] is not None and now - _SEG_CACHE["ts"] < _SEG_CACHE_SECONDS:
        return _SEG_CACHE["stats"]
    stats = {}

    def bump(k, win):
        d = stats.setdefault(k, {"n": 0, "wins": 0, "losses": 0})
        d["n"] += 1
        d["wins" if win else "losses"] += 1

    try:
        for meta, status in _entry_meta_rows():
            win = status == "WIN"
            for key in ("playbook", "grade", "hour_bucket", "side"):
                if meta.get(key):
                    bump(f"{key}={meta[key]}", win)
            bump(f"probe={bool(meta.get('probe'))}", win)
            if meta.get("playbook") and meta.get("grade"):
                bump(f"combo={meta['playbook']}|{meta['grade']}", win)
    except Exception:
        logger.exception("segment stats failed")
    for d in stats.values():
        d["win_rate"] = round(100.0 * d["wins"] / d["n"], 1) if d["n"] else None
    _SEG_CACHE["ts"], _SEG_CACHE["stats"] = now, stats
    return stats


def segment_gate(meta):
    """(blocked, reason).  Blocks a trade only when its segment has ENOUGH resolved trades
    and a clearly losing record.  Not enough history => allowed (that is how it learns)."""
    try:
        stats = get_segment_stats()
    except Exception:
        return False, ""
    checks = []
    for key in ("playbook", "grade", "hour_bucket", "side"):
        if meta.get(key):
            checks.append((f"{key}={meta[key]}", SEGMENT_MIN_SAMPLES, SEGMENT_BLOCK_WINRATE))
    if meta.get("playbook") and meta.get("grade"):
        checks.append((f"combo={meta['playbook']}|{meta['grade']}", SEGMENT_COMBO_MIN_SAMPLES, SEGMENT_BLOCK_WINRATE))
    if meta.get("probe"):
        checks.append(("probe=True", PROBE_MIN_SAMPLES, PROBE_PAUSE_WINRATE))
    for key, min_n, floor in checks:
        d = stats.get(key)
        if d and d["n"] >= min_n and d["win_rate"] is not None and d["win_rate"] < floor:
            return True, (f"segment '{key}' ka apna record kharab hai ({d['wins']}W/{d['losses']}L = "
                          f"{d['win_rate']}% < {floor:.0f}%) -- engine ne khud is tarah ke trade rok diye hain "
                          f"jab tak record sudhar na jaye")
    return False, ""


def recent_performance(window=RECENT_WINDOW):
    """Current form over the last resolved NON-probe trades: {'n', 'wins', 'win_rate', 'bad'}.
    v18: only trades tagged with the CURRENT logic_version are counted -- older trades (produced
    by since-changed gating logic) are skipped entirely so they can never be blamed on today's logic."""
    try:
        rows = get_recent_setups(limit=window * 8)
    except Exception:
        return {"n": 0, "wins": 0, "win_rate": None, "bad": False}
    all_resolved = [r for r in rows if r.get("status") in ("WIN", "LOSS") and not r.get("probe")]
    legacy_skipped = sum(1 for r in all_resolved if r.get("logic_version") != LOGIC_VERSION)
    res = [r for r in all_resolved if r.get("logic_version") == LOGIC_VERSION][:window]
    n = len(res)
    wins = sum(1 for r in res if r.get("status") == "WIN")
    wr = round(100.0 * wins / n, 1) if n else None
    # limit: not enough CURRENT-logic samples yet -> can't judge form, never mark bad off stale data
    bad = bool(n >= MIN_NEW_LOGIC_SAMPLES and wr is not None and wr < RECENT_BAD_WINRATE)
    # v17: the penalty must EXPIRE. Bad form -> higher bar -> fewer trades -> no new results -> form never updates
    # = the engine stays tight for days. If the newest counted trade is older than FORM_STALE_HOURS the old streak
    # no longer describes today's market, so the extra bar is dropped until fresh results say otherwise.
    stale = False
    if bad and res:
        try:
            from datetime import datetime as _dt
            ts = res[0].get("exit_timestamp") or res[0].get("timestamp")
            # timestamps are stored in IST; compare with IST "now" (server clock is UTC on Streamlit Cloud)
            age_h = (_now_ist() - _dt.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0
            if age_h > FORM_STALE_HOURS:
                bad, stale = False, True
        except Exception:
            pass
    return {"n": n, "wins": wins, "win_rate": wr, "bad": bad, "stale_bad": stale, "legacy_skipped": legacy_skipped}



# =====================================================================
# ENTRY COOLDOWN -- stops "stopped out, re-entered the same idea 2 candles later, stopped out again"
# =====================================================================
def entry_cooldown(direction, now=None):
    """(blocked, reason).  Two cheap, well-understood protections:

    1. After a LOSS in the SAME direction, wait COOLDOWN_AFTER_LOSS_MIN minutes.  A stop-out means
       the level just failed; the very next candle is the most likely place for a repeat loss.
    2. LOSS_STREAK_PAUSE consecutive losses today -> pause new entries LOSS_STREAK_PAUSE_MIN minutes
       (the market is clearly not behaving like the playbooks expect right now).
    Both expire by themselves, so they cannot freeze the engine for days.
    """
    now = now or _now_ist()
    try:
        rows = get_recent_setups(limit=12)
    except Exception:
        return False, ""

    def _t(r):
        try:
            return datetime.strptime(str(r.get("exit_timestamp") or r.get("timestamp"))[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    closed = [r for r in rows if r.get("status") in ("WIN", "LOSS", "EXPIRED")]
    # 1) same-direction cooldown after the latest closed trade of that direction
    for r in closed:
        if r.get("direction") != direction:
            continue
        t = _t(r)
        if r.get("status") == "LOSS" and t is not None and (now - t) < timedelta(minutes=COOLDOWN_AFTER_LOSS_MIN):
            left = COOLDOWN_AFTER_LOSS_MIN - int((now - t).total_seconds() // 60)
            return True, (f"Cooldown: last {direction} trade was stopped out {int((now - t).total_seconds() // 60)} min ago; "
                          f"waiting ~{left} more min so the engine does not re-enter the same failed idea.")
        break
    # 2) loss streak today
    streak = 0
    last_loss_t = None
    for r in closed:
        t = _t(r)
        if t is None or t.date() != now.date():
            break
        if r.get("status") == "LOSS":
            streak += 1
            last_loss_t = last_loss_t or t
        else:
            break
    if LOSS_STREAK_PAUSE and streak >= LOSS_STREAK_PAUSE and last_loss_t is not None and (now - last_loss_t) < timedelta(minutes=LOSS_STREAK_PAUSE_MIN):
        left = LOSS_STREAK_PAUSE_MIN - int((now - last_loss_t).total_seconds() // 60)
        return True, (f"Loss-streak pause: {streak} consecutive losses today; new entries resume in ~{left} min "
                      f"(market is not behaving like the playbooks expect).")
    return False, ""


# =====================================================================
# ONE SETUP, ONE TRADE -- never re-enter on a signal that has already been traded
# =====================================================================
def _parse_naive_ist(x):
    """Timestamp string / Timestamp (maybe tz-aware) -> naive IST datetime, or None."""
    if x is None:
        return None
    try:
        import pandas as _pd
        t = _pd.Timestamp(x)
        if t.tzinfo is not None:
            t = t.tz_convert("Asia/Kolkata").tz_localize(None)
        return t.to_pydatetime()
    except Exception:
        return None


def signal_already_traded(direction, formed_at):
    """(used, reason).  A strategy signal stays 'alive' for ~6 candles (30 min).  After a trade from it hit its target
    (or was stopped), the very next refresh used to see the SAME still-alive signal and open another trade on it --
    i.e. buy again at the top of the move that just paid out.  A new trade in the same direction now needs a signal
    that formed AFTER the previous same-direction trade was opened (a genuinely new setup + a fresh full research pass)."""
    ft = _parse_naive_ist(formed_at)
    if ft is None:
        return False, ""
    # `formed_at` is the OPEN time of the 5-min candle that completed the signal, i.e. the signal really exists
    # from the candle's close.  Without this shift a trade logged 30 s after that close would wrongly mark the very
    # NEXT candle's fresh signal as "already used".
    ft = ft + timedelta(minutes=5)
    try:
        rows = get_recent_setups(limit=12)
    except Exception:
        return False, ""
    for r in rows:                                     # newest first
        if r.get("direction") != direction:
            continue
        et = _parse_naive_ist(r.get("timestamp"))
        if et is not None and ft <= et:
            return True, (f"Same setup already traded: the {direction} strategy signal formed at {ft.strftime('%H:%M')} "
                          f"was already used by the trade opened {et.strftime('%H:%M')} ({r.get('status')}). "
                          f"Waiting for a NEW setup instead of re-entering after the move.")
        break
    return False, ""
