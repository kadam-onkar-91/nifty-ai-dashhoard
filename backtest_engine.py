import numpy as np

"""
PHASE 3 -- BACKTESTING / PERFORMANCE REPORT ENGINE
-----------------------------------------------------
Honesty note up front: a full realistic options backtest (with real
historical option-chain prices, brokerage, exchange charges, slippage,
and actual fills) needs a historical option-chain database that Upstox's
free tier does not provide, and this app does not have access to one.
Building that on fabricated/simulated historical option prices would
violate this project's core rule ("never invent values in production
mode") -- so this engine does NOT attempt a synthetic options backtest.

What IS real and honest: this app already logs every actual signal it
generated into trading_logs.db (database.py) -- entry, SL, targets, and
the real resolved outcome (TARGET HIT / SL HIT / TIME_EXIT) with real
P&L in index points. That is genuine forward "paper" performance, not a
backtest on invented history, and it only grows more meaningful the
longer the app runs. This engine reports statistics from THAT real data,
and explicitly says "NOT ENOUGH DATA YET" below a minimum sample size
rather than presenting noisy small-sample stats as if they were reliable.

Brokerage/slippage: since these trades are logged in index POINTS (not
an actual executed order with a real fill), realistic Rs. brokerage
can't be deducted from something that was never actually filled -- this
engine instead reports gross P&L in points plus a clearly-labeled
"illustrative" cost knock-off using a configurable points-per-trade
estimate, so the user can see the order of magnitude without the report
pretending those costs are exact.
"""

MIN_TRADES_FOR_STATS = 20  # below this, sample is too small to be meaningful
ILLUSTRATIVE_COST_PTS_PER_TRADE = 2.0  # rough combined brokerage+slippage+STT estimate, in Nifty points


def _max_drawdown(equity_curve):
    peak = -np.inf
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = peak - v
        max_dd = max(max_dd, dd)
    return round(max_dd, 2)


def _longest_streak(results, target):
    longest = cur = 0
    for r in results:
        if r == target:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest


def generate_backtest_report(closed_trades):
    """
    closed_trades: output of database.fetch_all_closed_trades() -- real
    logged trades only, oldest-first. Returns a dict report. If the
    sample is too small, returns status='NOT_ENOUGH_DATA' with what
    little is known rather than computing misleading ratios on noise.
    """
    n = len(closed_trades)
    if n == 0:
        return {"status": "NOT_ENOUGH_DATA", "sample_size": 0,
                "note": "No resolved trades logged yet. This section fills in as the system's real signals play out."}

    pnls = [t["pnl"] for t in closed_trades if t["pnl"] is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = round(len(wins) / len(pnls) * 100, 1) if pnls else None

    r_multiples = []
    for t in closed_trades:
        try:
            risk = abs(t["entry_price"] - t["stop_loss"])
            if risk > 0 and t["pnl"] is not None:
                r_multiples.append(t["pnl"] / risk)
        except Exception:
            continue

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (float('inf') if gross_profit > 0 else None)
    expectancy_pts = round(np.mean(pnls), 2) if pnls else None
    avg_r = round(float(np.mean(r_multiples)), 2) if r_multiples else None

    equity_curve = np.cumsum(pnls).tolist() if pnls else []
    max_dd = _max_drawdown(equity_curve) if equity_curve else None

    win_loss_seq = ["W" if p > 0 else "L" for p in pnls]
    longest_losing_streak = _longest_streak(win_loss_seq, "L")

    # Trade-based Sharpe-like ratio: mean R / std R. This is NOT an
    # annualized market Sharpe ratio (no fixed time basis for intraday
    # index-point trades) -- labeled explicitly to avoid overclaiming.
    sharpe_like = None
    sortino_like = None
    if len(r_multiples) >= 2:
        std_r = float(np.std(r_multiples, ddof=1))
        if std_r > 0:
            sharpe_like = round(float(np.mean(r_multiples)) / std_r, 2)
        downside = [r for r in r_multiples if r < 0]
        if downside:
            downside_std = float(np.std(downside, ddof=1)) if len(downside) >= 2 else abs(downside[0])
            if downside_std > 0:
                sortino_like = round(float(np.mean(r_multiples)) / downside_std, 2)

    mfe_vals = [t["mfe"] for t in closed_trades if t.get("mfe") is not None]
    mae_vals = [t["mae"] for t in closed_trades if t.get("mae") is not None]
    avg_mfe = round(float(np.mean(mfe_vals)), 2) if mfe_vals else None
    avg_mae = round(float(np.mean(mae_vals)), 2) if mae_vals else None

    illustrative_net_pnl = round(sum(pnls) - n * ILLUSTRATIVE_COST_PTS_PER_TRADE, 2) if pnls else None

    report = {
        "status": "OK" if n >= MIN_TRADES_FOR_STATS else "NOT_ENOUGH_DATA",
        "sample_size": n,
        "min_required_for_reliable_stats": MIN_TRADES_FOR_STATS,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "expectancy_pts_per_trade": expectancy_pts,
        "avg_r_multiple": avg_r,
        "gross_pnl_pts": round(sum(pnls), 2) if pnls else None,
        "illustrative_net_pnl_pts": illustrative_net_pnl,
        "illustrative_cost_note": f"Net figure deducts an illustrative {ILLUSTRATIVE_COST_PTS_PER_TRADE} pts/trade for brokerage+slippage+STT -- NOT a real fill-based cost, since these are logged signals, not actual executed orders.",
        "max_drawdown_pts": max_dd,
        "avg_winner_pts": round(float(np.mean(wins)), 2) if wins else None,
        "avg_loser_pts": round(float(np.mean(losses)), 2) if losses else None,
        "largest_loss_pts": round(min(losses), 2) if losses else None,
        "largest_win_pts": round(max(wins), 2) if wins else None,
        "longest_losing_streak": longest_losing_streak,
        "sharpe_like_ratio": sharpe_like,
        "sortino_like_ratio": sortino_like,
        "sharpe_sortino_note": "Trade-based (mean R / std R of R-multiples), NOT an annualized market Sharpe -- there's no fixed time basis for intraday index-point trades.",
        "avg_mfe_pts": avg_mfe,
        "avg_mae_pts": avg_mae,
        "mfe_mae_note": "Sampled once per ~30s dashboard refresh while each trade was open, not tick-precision.",
    }
    if report["status"] == "NOT_ENOUGH_DATA":
        report["note"] = (f"Only {n} resolved trades logged so far -- need at least {MIN_TRADES_FOR_STATS} "
                           f"for these stats to be statistically meaningful rather than noise. Figures above "
                           f"are shown as-is from what's logged, but treat them as provisional until the sample grows.")
    return report
