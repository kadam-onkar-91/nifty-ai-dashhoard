import sqlite3
from datetime import datetime
import os

DB_NAME = "trading_logs.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT, signal TEXT, entry_price REAL,
                      stop_loss REAL, target REAL, status TEXT,
                      exit_price REAL, pnl REAL)''')
        conn.commit()

def check_open_position():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT id, signal, entry_price FROM trades WHERE status = 'OPEN'")
        return c.fetchone()

def log_entry_safe(signal, entry_price, sl, target):
    if check_open_position():
        return None, "Blocked: Another trade is already OPEN."
        
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("""INSERT INTO trades (timestamp, signal, entry_price, stop_loss, target, status) 
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (timestamp, signal, entry_price, sl, target, 'OPEN'))
        conn.commit()
        return c.lastrowid, "Success: Trade Logged."

def log_outcome(trade_id, exit_price, status):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT entry_price, signal FROM trades WHERE id=?", (trade_id,))
        row = c.fetchone()
        if row:
            entry_price, signal = row
            pnl = (exit_price - entry_price) if signal.upper() == 'BUY' else (entry_price - exit_price)
            c.execute("UPDATE trades SET exit_price = ?, pnl = ?, status = ? WHERE id = ?",
                      (exit_price, round(pnl, 2), status, trade_id))
            conn.commit()

def check_and_update_open_trades(live_price):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT id, signal, stop_loss, target FROM trades WHERE status = 'OPEN'")
        open_trades = c.fetchall()
        
    for t_id, signal, sl, target in open_trades:
        if signal.upper() == 'BUY':
            if live_price >= target: log_outcome(t_id, live_price, 'TARGET HIT')
            elif live_price <= sl: log_outcome(t_id, live_price, 'SL HIT')
        elif signal.upper() == 'SELL':
            if live_price <= target: log_outcome(t_id, live_price, 'TARGET HIT')
            elif live_price >= sl: log_outcome(t_id, live_price, 'SL HIT')

def fetch_performance_metrics():
    if not os.path.exists(DB_NAME): return 0, 0, 0.0
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='trades'")
        if c.fetchone()[0] == 0: return 0, 0, 0.0
        
        c.execute("SELECT COUNT(*) FROM trades WHERE status = 'TARGET HIT'")
        wins = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM trades WHERE status = 'SL HIT'")
        losses = c.fetchone()[0]
        
    total_closed = wins + losses
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    return wins, losses, round(win_rate, 2)
