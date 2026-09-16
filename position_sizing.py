"""
PHASE 4 -- POSITION SIZING ENGINE (spec section 28)
------------------------------------------------------
Pure calculation, no market data needed -- this is deliberately simple
and configurable, per the spec: "Never increase position size simply
because confidence is high."

NIFTY_LOT_SIZE is NSE's current official lot size and needs manual
updating if NSE revises it (exchanges do this periodically) -- it is
NOT fetched live because Upstox's instrument master row for the futures
contract already used elsewhere (market_data.py) doesn't cleanly expose
lot size in a way this app currently parses; flagged here rather than
silently risking a stale number without the user knowing where it comes from.
"""

NIFTY_LOT_SIZE = 75  # NSE Nifty 50 lot size as of the 2025 revision -- update if NSE changes it


def calculate_position_size(capital, risk_per_trade_pct, entry_price, stop_loss_price,
                             lot_size=NIFTY_LOT_SIZE, max_daily_risk_pct=3.0):
    """
    capital: total trading capital (Rs.)
    risk_per_trade_pct: % of capital willing to risk on this one trade (e.g. 1.0 = 1%)
    entry_price / stop_loss_price: in Nifty points
    Returns a dict with lots, risk amount, and warnings -- never silently
    rounds risk UP because confidence was high; only ever rounds DOWN to
    the nearest whole lot (a fraction of a lot can't be traded).
    """
    out = {"status": "OK", "warnings": []}
    try:
        sl_distance_pts = abs(entry_price - stop_loss_price)
        if sl_distance_pts <= 0:
            return {"status": "INVALID", "reason": "Entry and Stop-Loss can't be the same price -- no risk distance to size against."}
        if capital <= 0 or risk_per_trade_pct <= 0:
            return {"status": "INVALID", "reason": "Capital and risk % must both be positive."}

        risk_amount_rs = capital * (risk_per_trade_pct / 100.0)
        risk_per_lot_rs = sl_distance_pts * lot_size
        raw_lots = risk_amount_rs / risk_per_lot_rs
        lots = int(raw_lots)  # always round DOWN -- never round up because "confidence is high"

        if lots < 1:
            out["warnings"].append(
                f"Risk budget (₹{risk_amount_rs:,.0f}) is smaller than the risk of even 1 lot "
                f"(₹{risk_per_lot_rs:,.0f} at this SL distance) -- either reduce SL distance, "
                f"increase risk_per_trade_pct, or skip this trade."
            )
            lots = 0

        actual_risk_rs = lots * risk_per_lot_rs
        actual_risk_pct_of_capital = round((actual_risk_rs / capital) * 100, 2) if capital > 0 else None

        if actual_risk_pct_of_capital is not None and actual_risk_pct_of_capital > max_daily_risk_pct:
            out["warnings"].append(
                f"This single trade's risk ({actual_risk_pct_of_capital}% of capital) already exceeds "
                f"your configured max daily risk ({max_daily_risk_pct}%) -- reduce lots or SL distance."
            )

        out.update({
            "lots": lots,
            "quantity": lots * lot_size,
            "sl_distance_pts": round(sl_distance_pts, 2),
            "risk_per_lot_rs": round(risk_per_lot_rs, 2),
            "planned_risk_rs": round(risk_amount_rs, 2),
            "actual_risk_rs": round(actual_risk_rs, 2),
            "actual_risk_pct_of_capital": actual_risk_pct_of_capital,
            "lot_size_used": lot_size,
        })
        return out
    except Exception as e:
        return {"status": "ERROR", "reason": f"{type(e).__name__}: {e}"}
