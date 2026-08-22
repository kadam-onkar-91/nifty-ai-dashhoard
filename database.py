import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "trading_terminal.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            signal_type TEXT,
            entry_price REAL,
            stop_loss REAL,
            target_1 REAL,
            status TEXT DEFAULT 'PENDING',
            exit_price REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_entry_safe(signal_type, entry_price, stop_loss, target_1):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO trades (timestamp, signal_type, entry_price, stop_loss, target_1, status)
            VALUES (?, ?, ?, ?, ?, 'PENDING')
        ''', (timestamp, signal_type, entry_price, stop_loss, target_1))
        conn.commit()
        t_id = cursor.lastrowid
        conn.close()
        return t_id, "Success"
    except Exception as e:
        return None, str(e)

def update_pending_outcomes(live_price):
    """Checks pending trades against current live price to mark WIN or LOSS dynamically."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, signal_type, entry_price, stop_loss, target_1 FROM trades WHERE status = 'PENDING'")
        pending_trades = cursor.fetchall()

        for t_id, sig, entry, sl, t1 in pending_trades:
            if "Bullish" in sig or "BUY" in sig or "LONG" in sig or "🟢" in sig:
                if live_price >= t1:
                    cursor.execute("UPDATE trades SET status = 'WIN', exit_price = ? WHERE id = ?", (live_price, t_id))
                elif live_price <= sl:
                    cursor.execute("UPDATE trades SET status = 'LOSS', exit_price = ? WHERE id = ?", (live_price, t_id))
            elif "Bearish" in sig or "SELL" in sig or "SHORT" in sig or "🔴" in sig:
                if live_price <= t1:
                    cursor.execute("UPDATE trades SET status = 'WIN', exit_price = ? WHERE id = ?", (live_price, t_id))
                elif live_price >= sl:
                    cursor.execute("UPDATE trades SET status = 'LOSS', exit_price = ? WHERE id = ?", (live_price, t_id))

        conn.commit()
        conn.close()
    except Exception:
        pass

def get_real_accuracy_stats():
    """Computes actual win rate based on historical resolved trades."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status IN ('WIN', 'LOSS')")
        total_resolved = cursor.fetchone()[0]

        if total_resolved == 0:
            conn.close()
            return {"win_rate": None, "total_resolved": 0, "wins": 0, "losses": 0}

        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'WIN'")
        wins = cursor.fetchone()[0]
        losses = total_resolved - wins
        win_rate = round((wins / total_resolved) * 100, 1)

        conn.close()
        return {"win_rate": win_rate, "total_resolved": total_resolved, "wins": wins, "losses": losses}
    except Exception:
        return {"win_rate": None, "total_resolved": 0, "wins": 0, "losses": 0}

def fetch_performance_metrics():
    stats = get_real_accuracy_stats()
    return stats['wins'], stats['losses'], (stats['win_rate'] if stats['win_rate'] is not None else 0.0)
