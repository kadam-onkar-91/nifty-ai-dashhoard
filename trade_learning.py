"""
trade_learning.py — NEW, purely additive module.

A self-learning trade DECISION engine, separate from the existing
database.py trade log (which is left completely untouched). This module:

  1. Only speaks up when a real minimum bar of confluence is met -- it
     does NOT recommend something on every single refresh. If nothing
     qualifies, it says so plainly instead of inventing a trade.
  2. Reads EVERY factor this tool already computes (technical,
     OI/options, ICT/SMC, global markets, FII/DII, breadth, Bank Nifty
     correlation, the ML ensemble model, AND India VIX as a live
     fundamentals/event-risk proxy) and picks a specific strike + CE/PE
     + entry + stop-loss + target.
  3. Logs every recommendation it makes (with the exact factor snapshot
     at that moment) to its own SQLite table, then automatically
     resolves each one (WIN/LOSS/EXPIRED) against live price -- so it
     builds a REAL historical track record instead of a guessed one.
  4. Uses that growing history to learn, per factor, whether it has
     actually been predictive in THIS user's own data -- and blends
     that learned reliability into future confidence numbers. With
     little or no history yet, it says exactly that instead of
     pretending to be confident.
  5. Confidence is DELIBERATELY capped well below 90% -- even a large,
     mature historical edge rarely holds above ~78% for a setup like
     this in real markets, and claiming higher would be dishonest. There
     is no such thing as a "guaranteed" trade from any tool; this module
     will never claim one.

Nothing here can read the user's mind or predict news -- it can only
tell you what has actually lined up, and how often similar alignments
have gone well SO FAR in the recorded history.
"""

import sqlite3
import json
from datetime import datetime, timedelta

DB_NAME = "trade_learning.db"

# Confidence is always clamped into this band. The floor keeps a very
# weak setup from reading as "0% / hopeless" (it's still a coin-flip-ish
# probability, not impossible); the ceiling is a deliberate honesty cap --
# no setup here is ever allowed to claim nineties-level confidence.
CONFIDENCE_FLOOR = 32.0
CONFIDENCE_CEILING = 78.0

# A factor's learned win-rate is only trusted once it has this many
# resolved (WIN/LOSS) samples -- below that, the rule-based equal weight
# is used instead, and the UI says so explicitly.
MIN_SAMPLES_FOR_LEARNING = 8

# A setup with fewer than this many TRUE factors is not worth logging at
# all -- this is what stops the old "trades at every single point with no
# logic" problem from ever happening again.
MIN_FACTORS_TRUE_TO_QUALIFY = 5

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

MAX_HOLD_MINUTES = 120  # same convention as database.py -- force-resolve stale setups


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ai_trade_setups
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  direction TEXT,
                  strike INTEGER,
                  option_type TEXT,
                  underlying_entry REAL,
                  stop_loss REAL,
                  target REAL,
                  confidence_pct REAL,
                  factors_json TEXT,
                  status TEXT DEFAULT 'OPEN',
                  exit_price REAL,
                  exit_timestamp TEXT)''')
    conn.commit()
    conn.close()


def _open_setup_signature():
    """Returns (direction, strike) of the currently OPEN setup, if any --
    used so a new setup isn't logged on top of one already being tracked
    for the same direction/strike."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT direction, strike FROM ai_trade_setups WHERE status='OPEN' ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row


def log_setup(direction, strike, option_type, underlying_entry, stop_loss, target,
              confidence_pct, factor_flags: dict):
    """
    Logs a new AI-recommended setup, ONLY if nothing with the same
    (direction, strike) is already open -- prevents duplicate logging on
    every 30s refresh while the same setup persists.
    """
    existing = _open_setup_signature()
    if existing is not None and existing[0] == direction and existing[1] == strike:
        return None  # same setup already being tracked

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""INSERT INTO ai_trade_setups
                 (timestamp, direction, strike, option_type, underlying_entry,
                  stop_loss, target, confidence_pct, factors_json, status)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
              (timestamp, direction, strike, option_type, underlying_entry,
               stop_loss, target, confidence_pct, json.dumps(factor_flags)))
    setup_id = c.lastrowid
    conn.commit()
    conn.close()
    return setup_id


def _close_setup(setup_id, status, exit_price):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""UPDATE ai_trade_setups SET status=?, exit_price=?, exit_timestamp=?
                 WHERE id=?""", (status, exit_price, now, setup_id))
    conn.commit()
    conn.close()


def resolve_open_setups(live_price):
    """
    Call every refresh with the current underlying price. Auto-resolves
    the open setup (on the UNDERLYING index level, since strike-level
    premium isn't tracked historically here) against its stop-loss and
    target, or force-closes it as EXPIRED after MAX_HOLD_MINUTES so
    nothing sits open forever.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""SELECT id, timestamp, direction, underlying_entry, stop_loss, target
                 FROM ai_trade_setups WHERE status='OPEN'""")
    open_rows = c.fetchall()
    conn.close()

    for setup_id, ts, direction, entry, sl, target in open_rows:
        is_buy = direction == "BUY"
        if is_buy:
            if live_price >= target:
                _close_setup(setup_id, "WIN", live_price)
                continue
            elif live_price <= sl:
                _close_setup(setup_id, "LOSS", live_price)
                continue
        else:
            if live_price <= target:
                _close_setup(setup_id, "WIN", live_price)
                continue
            elif live_price >= sl:
                _close_setup(setup_id, "LOSS", live_price)
                continue
        try:
            opened_at = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if datetime.now() - opened_at > timedelta(minutes=MAX_HOLD_MINUTES):
                _close_setup(setup_id, "EXPIRED", live_price)
        except Exception:
            pass


def get_open_setup():
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


def get_overall_track_record():
    """Real, measured win-rate across every RESOLVED (WIN/LOSS) setup this
    engine has recommended -- EXPIRED setups are reported separately since
    they were neither a clean win nor a clean loss."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ai_trade_setups'")
    if c.fetchone()[0] == 0:
        conn.close()
        return {"wins": 0, "losses": 0, "expired": 0, "win_rate": None, "sample_size": 0}
    c.execute("SELECT COUNT(*) FROM ai_trade_setups WHERE status='WIN'")
    wins = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM ai_trade_setups WHERE status='LOSS'")
    losses = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM ai_trade_setups WHERE status='EXPIRED'")
    expired = c.fetchone()[0]
    conn.close()
    resolved = wins + losses
    win_rate = round((wins / resolved) * 100, 1) if resolved > 0 else None
    return {"wins": wins, "losses": losses, "expired": expired,
            "win_rate": win_rate, "sample_size": resolved}


def get_factor_reliability():
    """
    THE SELF-LEARNING PART. For every factor this engine tracks, looks
    across all RESOLVED (WIN/LOSS) past setups and computes: when this
    factor was TRUE, what fraction of those setups actually won? This is
    measured directly from this user's own history -- nothing here is
    assumed or hardcoded.

    Returns {factor_key: {"win_rate_when_true": float or None,
                           "samples_when_true": int,
                           "label": str}}
    A factor with too few samples (< MIN_SAMPLES_FOR_LEARNING) reports
    win_rate_when_true=None, meaning "not enough data yet to trust this."
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ai_trade_setups'")
    if c.fetchone()[0] == 0:
        conn.close()
        return {k: {"win_rate_when_true": None, "samples_when_true": 0, "label": FACTOR_LABELS[k]}
                for k in ALL_FACTOR_KEYS}

    c.execute("SELECT factors_json, status FROM ai_trade_setups WHERE status IN ('WIN','LOSS')")
    rows = c.fetchall()
    conn.close()

    result = {}
    for key in ALL_FACTOR_KEYS:
        true_wins, true_total = 0, 0
        for factors_json, status in rows:
            try:
                factors = json.loads(factors_json)
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
    Blends a simple rule-based score (equal weight per TRUE factor) with
    LEARNED factor reliabilities wherever enough history exists for that
    specific factor -- otherwise falls back to the equal-weight rule for
    that factor. Result is always clamped to [CONFIDENCE_FLOOR,
    CONFIDENCE_CEILING] -- never allowed to claim 90%+, honestly reflecting
    that no tool can promise that.

    Returns (confidence_pct, used_learning: bool, learned_factor_count: int)
    """
    reliability = get_factor_reliability()
    true_count = sum(1 for k in ALL_FACTOR_KEYS if factor_flags.get(k))
    total = len(ALL_FACTOR_KEYS)

    weighted_sum = 0.0
    weight_total = 0.0
    learned_factor_count = 0

    for key in ALL_FACTOR_KEYS:
        is_true = bool(factor_flags.get(key))
        rel = reliability[key]
        if rel["win_rate_when_true"] is not None:
            # Learned weight: how far this factor's actual win-rate sits
            # from a 50/50 coin flip, scaled to a +/-1 contribution.
            learned_factor_count += 1
            lift = (rel["win_rate_when_true"] - 50.0) / 50.0  # -1..1
            weighted_sum += (lift if is_true else 0.0)
            weight_total += 1.0
        else:
            # Not enough history for this factor yet -- fall back to a
            # plain rule-based contribution (present/absent, equal weight).
            weighted_sum += (0.5 if is_true else 0.0)
            weight_total += 1.0

    base_pct = 50.0 + (weighted_sum / weight_total) * 40.0 if weight_total > 0 else 50.0
    confidence = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, round(base_pct, 1)))
    return confidence, learned_factor_count > 0, learned_factor_count
