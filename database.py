import sqlite3
from datetime import datetime

def log_prediction_to_db(ltp, signal_val, sl, t1, t2):
    try:
        conn = sqlite3.connect('nifty_predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ltp REAL,
                signal TEXT,
                stop_loss REAL,
                target_1 REAL,
                target_2 REAL
            )
        ''')
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        signal_text = "BULLISH (BUY)" if signal_val == 1 else "BEARISH (SELL)"
        
        cursor.execute('''
            INSERT INTO signals (timestamp, ltp, signal, stop_loss, target_1, target_2)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (current_time, ltp, signal_text, sl, t1, t2))
        
        conn.commit()
        conn.close()
    except Exception:
        pass
