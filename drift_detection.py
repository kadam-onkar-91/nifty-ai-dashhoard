import numpy as np

"""
PHASE 3 -- MODEL DRIFT DETECTION
------------------------------------
Compares the system's REAL win rate / expectancy over its most recent
trades against its own all-time baseline, using only actually-resolved
logged trades (database.fetch_all_closed_trades()). If recent
performance has meaningfully deteriorated, this flags it explicitly
rather than silently doing anything about it -- per the spec, drift
detection should surface the problem, not auto-retrain or hide it.

Needs a real minimum sample on both sides of the comparison; with too
few trades, a "drop" is just noise, so this reports NOT_ENOUGH_DATA
instead of a false alarm.
"""

MIN_BASELINE_TRADES = 20
ROLLING_WINDOW = 10
DRIFT_WIN_RATE_DROP_PTS = 15.0  # percentage-point drop vs baseline to flag drift


def check_for_drift(closed_trades, rolling_window=ROLLING_WINDOW):
    n = len(closed_trades)
    if n < MIN_BASELINE_TRADES:
        return {
            "status": "NOT_ENOUGH_DATA", "sample_size": n,
            "min_required": MIN_BASELINE_TRADES,
            "note": f"Only {n} resolved trades logged -- need at least {MIN_BASELINE_TRADES} "
                    f"before a meaningful baseline vs recent-window comparison is possible."
        }

    def _win_rate(trades):
        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        if not pnls:
            return None
        wins = sum(1 for p in pnls if p > 0)
        return round(wins / len(pnls) * 100, 1)

    def _expectancy(trades):
        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        return round(float(np.mean(pnls)), 2) if pnls else None

    baseline_trades = closed_trades  # all-time
    recent_trades = closed_trades[-rolling_window:]

    baseline_wr = _win_rate(baseline_trades)
    recent_wr = _win_rate(recent_trades)
    baseline_exp = _expectancy(baseline_trades)
    recent_exp = _expectancy(recent_trades)

    drift_flag = False
    drift_reasons = []
    if baseline_wr is not None and recent_wr is not None:
        drop = baseline_wr - recent_wr
        if drop >= DRIFT_WIN_RATE_DROP_PTS:
            drift_flag = True
            drift_reasons.append(f"Recent {rolling_window}-trade win rate ({recent_wr}%) is {round(drop,1)} pts "
                                  f"below the all-time baseline ({baseline_wr}%).")
    if baseline_exp is not None and recent_exp is not None and baseline_exp > 0 and recent_exp < 0:
        drift_flag = True
        drift_reasons.append(f"Recent {rolling_window}-trade expectancy has turned negative "
                              f"({recent_exp} pts/trade) vs a positive all-time baseline ({baseline_exp} pts/trade).")

    return {
        "status": "MODEL PERFORMANCE DRIFT DETECTED ⚠️" if drift_flag else "STABLE ✅",
        "sample_size": n,
        "rolling_window": rolling_window,
        "baseline_win_rate_pct": baseline_wr,
        "recent_win_rate_pct": recent_wr,
        "baseline_expectancy_pts": baseline_exp,
        "recent_expectancy_pts": recent_exp,
        "drift_reasons": drift_reasons,
        "note": "Compares the system's own real logged trades over time -- not a retrain signal, purely a flag for the user to review.",
    }
