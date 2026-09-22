"""Stock-specific learning helpers for the multi-stock dashboard.

Learning is intentionally isolated by symbol. NIFTY option-trade learning is
not copied into individual equity profiles.
"""
from app_logging import get_logger
logger = get_logger(__name__)
import json
from datetime import datetime, timezone


def ensure_tables(supabase_url=None, supabase_key=None):
    """Return additive SQL for the stock-learning tables.

    The dashboard does not execute DDL through Supabase REST. The SQL is also
    included in supabase_schema.sql for a one-time additive migration.
    """
    return True


def profile_from_history(rows):
    wins = losses = 0
    for row in rows or []:
        status = str(row.get("status", "")).upper()
        if status == "WIN": wins += 1
        elif status == "LOSS": losses += 1
    total = wins + losses
    return {
        "resolved": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round((wins / total) * 100, 2) if total else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
