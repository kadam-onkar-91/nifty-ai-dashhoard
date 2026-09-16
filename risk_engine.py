from datetime import datetime

"""
PHASE 4 -- RISK ENGINE (spec section 30)
---------------------------------------------
Built entirely from the REAL trade log (database.fetch_all_closed_trades
+ database.check_open_position) -- not a simulation. Tracks today's
realized risk/PnL and consecutive losses, and tells the caller whether a
NEW entry should be blocked under the configured safety rules. This
module does not itself block anything -- app.py decides whether to act
on the recommendation, same pattern as the rest of this app's
"read-only advisory" modules.
"""

DEFAULT_MAX_TRADES_PER_DAY = 4
DEFAULT_MAX_CONSECUTIVE_LOSSES = 3
DEFAULT_MAX_DAILY_LOSS_PCT = 3.0


def _is_today(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").date() == datetime.now().date()
    except Exception:
        return False


def evaluate_risk_state(closed_trades, capital,
                         max_trades_per_day=DEFAULT_MAX_TRADES_PER_DAY,
                         max_consecutive_losses=DEFAULT_MAX_CONSECUTIVE_LOSSES,
                         max_daily_loss_pct=DEFAULT_MAX_DAILY_LOSS_PCT):
    """
    closed_trades: database.fetch_all_closed_trades() (real, resolved trades only).
    capital: account capital in Rs., for converting today's real P&L (in
    Nifty points) into a % of capital -- this assumes 1 lot per trade for
    the % figure unless the caller supplies actual position sizes, which
    this app's current trade log doesn't store; flagged in the output
    rather than silently assumed to be precise.
    """
    today_trades = [t for t in closed_trades if _is_today(t.get("exit_timestamp") or t.get("timestamp") or "")]
    today_pnl_pts = sum(t["pnl"] for t in today_trades if t["pnl"] is not None)
    today_trade_count = len(today_trades)

    # Consecutive losses: walk backwards from the most recent trade (any day)
    consecutive_losses = 0
    for t in reversed(closed_trades):
        if t["pnl"] is None:
            continue
        if t["pnl"] <= 0:
            consecutive_losses += 1
        else:
            break

    from position_sizing import NIFTY_LOT_SIZE
    today_pnl_rs_1lot = today_pnl_pts * NIFTY_LOT_SIZE
    today_pnl_pct_of_capital = round((today_pnl_rs_1lot / capital) * 100, 2) if capital and capital > 0 else None

    blocks = []
    if today_trade_count >= max_trades_per_day:
        blocks.append(f"Max trades/day reached ({today_trade_count}/{max_trades_per_day}) -- no new entries today.")
    if consecutive_losses >= max_consecutive_losses:
        blocks.append(f"{consecutive_losses} consecutive losses -- stop and review before the next entry (limit: {max_consecutive_losses}).")
    if today_pnl_pct_of_capital is not None and today_pnl_pct_of_capital <= -max_daily_loss_pct:
        blocks.append(f"Daily loss limit hit ({today_pnl_pct_of_capital}% of capital, limit -{max_daily_loss_pct}%) -- no new entries today.")

    return {
        "status": "NO_NEW_ENTRIES" if blocks else "OK_TO_TRADE",
        "today_trade_count": today_trade_count,
        "today_pnl_pts": round(today_pnl_pts, 2),
        "today_pnl_pct_of_capital_1lot_estimate": today_pnl_pct_of_capital,
        "today_pnl_estimate_note": "Estimated assuming 1 lot/trade -- the trade log doesn't store actual lots sized per trade.",
        "consecutive_losses": consecutive_losses,
        "limits": {
            "max_trades_per_day": max_trades_per_day,
            "max_consecutive_losses": max_consecutive_losses,
            "max_daily_loss_pct": max_daily_loss_pct,
        },
        "block_reasons": blocks,
    }
