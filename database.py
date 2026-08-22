import sqlite3
from datetime import datetime, timedelta

DB_NAME = "trading_logs.db"

# If a trade sits open longer than this with neither SL nor Target hit,
# it gets closed at current price so it doesn't block new signals forever
# and so it doesn't silently vanish from the sample.
MAX_HOLD_MINUTES = 120


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  signal TEXT,
                  entry_price REAL,
                  stop_loss REAL,
                  target_1 REAL,
                  target_2 REAL,
                  status TEXT DEFAULT 'OPEN',
                  exit_price REAL,
                  exit_timestamp TEXT,
                  pnl REAL)''')
    conn.commit()
    conn.close()


def check_open_position():
    """Returns the currently OPEN trade (if any) as a dict, else None."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""SELECT id, timestamp, signal, entry_price, stop_loss, target_1, target_2
                 FROM trades WHERE status = 'OPEN' ORDER BY id DESC LIMIT 1""")
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0], "timestamp": row[1], "signal": row[2],
        "entry_price": row[3], "stop_loss": row[4], "target_1": row[5], "target_2": row[6]
    }


def log_entry_safe(signal, entry_price, sl, target_1, target_2=None):
    """
    Only logs a new trade if no trade is currently OPEN. This is what
    prevents the database from filling up with overlapping/duplicate
    entries every time the app refreshes and the signal repeats.
    """
    if check_open_position() is not None:
        return None, "System already has an OPEN position — new signal ignored until it resolves."

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""INSERT INTO trades (timestamp, signal, entry_price, stop_loss, target_1, target_2, status)
                 VALUES (?, ?, ?, ?, ?, ?, 'OPEN')""",
              (timestamp, signal, entry_price, sl, target_1, target_2))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    return trade_id, "New trade logged."


def _close_trade(trade_id, entry_price, signal, exit_price, status):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if "BUY" in signal.upper() or signal == "1":
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""UPDATE trades SET exit_price = ?, exit_timestamp = ?, pnl = ?, status = ?
                 WHERE id = ?""", (exit_price, now, round(pnl, 2), status, trade_id))
    conn.commit()
    conn.close()


def update_stop_loss(trade_id, new_sl):
    """
    Updates the stop-loss of the currently open trade (used for trailing
    SL / breakeven-lock logic). This keeps the database as the single
    source of truth for trade state instead of a separate session_state
    system that can drift out of sync with it.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE trades SET stop_loss = ? WHERE id = ? AND status = 'OPEN'", (new_sl, trade_id))
    conn.commit()
    conn.close()


def check_and_update_open_trades(live_price):
    """
    Call this on every refresh with the current live price. It checks the
    open trade against its Target/SL and closes it automatically. If a
    trade has been open too long with neither level hit, it force-closes
    at the current price (TIME_EXIT) so trades never sit open forever and
    so new signals aren't blocked indefinitely.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""SELECT id, timestamp, signal, entry_price, stop_loss, target_1
                 FROM trades WHERE status = 'OPEN'""")
    open_trades = c.fetchall()
    conn.close()

    for trade_id, ts, signal, entry_price, sl, target_1 in open_trades:
        is_buy = "BUY" in signal.upper() or signal == "1"

        if is_buy:
            if live_price >= target_1:
                _close_trade(trade_id, entry_price, signal, live_price, "TARGET HIT")
                continue
            elif live_price <= sl:
                _close_trade(trade_id, entry_price, signal, live_price, "SL HIT")
                continue
        else:
            if live_price <= target_1:
                _close_trade(trade_id, entry_price, signal, live_price, "TARGET HIT")
                continue
            elif live_price >= sl:
                _close_trade(trade_id, entry_price, signal, live_price, "SL HIT")
                continue

        try:
            opened_at = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if datetime.now() - opened_at > timedelta(minutes=MAX_HOLD_MINUTES):
                _close_trade(trade_id, entry_price, signal, live_price, "TIME_EXIT")
        except Exception:
            pass


def fetch_performance_metrics():
    """
    Returns the REAL, measured track record. win_rate is calculated only
    from resolved TARGET HIT / SL HIT trades (a cleaner win-rate signal);
    TIME_EXIT trades are reported separately since they were neither a
    clean win nor a clean loss.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='trades'")
    if c.fetchone()[0] == 0:
        conn.close()
        return {"wins": 0, "losses": 0, "time_exits": 0, "win_rate": None, "sample_size": 0, "total_pnl": 0.0}

    c.execute("SELECT COUNT(*) FROM trades WHERE status = 'TARGET HIT'")
    wins = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE status = 'SL HIT'")
    losses = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM trades WHERE status = 'TIME_EXIT'")
    time_exits = c.fetchone()[0]
    c.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status != 'OPEN'")
    total_pnl = c.fetchone()[0]
    conn.close()

    resolved = wins + losses
    win_rate = round((wins / resolved) * 100, 1) if resolved > 0 else None

    return {
        "wins": wins, "losses": losses, "time_exits": time_exits,
        "win_rate": win_rate, "sample_size": resolved, "total_pnl": round(total_pnl, 2)
    }
