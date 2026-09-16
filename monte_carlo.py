import numpy as np

"""
PHASE 3 -- MONTE CARLO ANALYSIS
-----------------------------------
This does NOT simulate the market or invent hypothetical trades. It
bootstrap-resamples (with replacement) the REAL sequence of resolved
R-multiples this system has actually logged, many times over, to show
the RANGE of drawdown/losing-streak outcomes that same edge (or lack of
one) could plausibly produce in a different order -- a standard,
honest use of Monte Carlo for an existing track record. It is never
presented as a prediction of future returns, only as a distribution
of what the system's OWN real historical R-multiples imply about risk.

Needs a minimum real sample (see MIN_TRADES) -- below that, resampling
a handful of trades hundreds of times just manufactures false precision
from noise, so this module reports NOT_ENOUGH_DATA instead.
"""

MIN_TRADES = 20
N_SIMULATIONS = 2000


def _max_drawdown_r(r_curve):
    peak = -np.inf
    max_dd = 0.0
    for v in r_curve:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    return max_dd


def _longest_losing_streak(sequence):
    longest = cur = 0
    for r in sequence:
        if r < 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest


def run_monte_carlo(closed_trades, risk_per_trade_pct=1.0, starting_capital_units=100.0, n_sims=N_SIMULATIONS, seed=42):
    """
    closed_trades: output of database.fetch_all_closed_trades().
    risk_per_trade_pct / starting_capital_units: purely for translating
    R-multiples into an illustrative equity-curve shape (e.g. "if you
    risked 1% of capital per trade") -- these are user-configurable
    assumptions, not measured facts, and are labeled as such in the output.
    """
    r_multiples = []
    for t in closed_trades:
        try:
            risk = abs(t["entry_price"] - t["stop_loss"])
            if risk > 0 and t["pnl"] is not None:
                r_multiples.append(t["pnl"] / risk)
        except Exception:
            continue

    n = len(r_multiples)
    if n < MIN_TRADES:
        return {
            "status": "NOT_ENOUGH_DATA", "sample_size": n, "min_required": MIN_TRADES,
            "note": f"Only {n} real resolved trades with valid R-multiples so far -- need at least "
                    f"{MIN_TRADES} before a Monte Carlo resample means anything beyond noise."
        }

    rng = np.random.default_rng(seed)
    r_arr = np.array(r_multiples)

    terminal_r_totals = []
    max_drawdowns_r = []
    losing_streaks = []

    for _ in range(n_sims):
        sample = rng.choice(r_arr, size=n, replace=True)
        equity_r = np.cumsum(sample)
        terminal_r_totals.append(float(equity_r[-1]))
        max_drawdowns_r.append(_max_drawdown_r(equity_r))
        losing_streaks.append(_longest_losing_streak(sample))

    terminal_r_totals = np.array(terminal_r_totals)
    max_drawdowns_r = np.array(max_drawdowns_r)
    losing_streaks = np.array(losing_streaks)

    equity_pct_change = terminal_r_totals * risk_per_trade_pct  # approx, assumes fixed % risk each trade
    risk_of_ruin_pct = round(float(np.mean(equity_pct_change <= -50.0)) * 100, 1)  # % of sims losing >=50% of capital

    return {
        "status": "OK",
        "sample_size": n,
        "n_simulations": n_sims,
        "assumptions_note": f"Assumes a fixed {risk_per_trade_pct}% of capital risked per trade (user-configurable), applied to the REAL historical R-multiple sequence resampled {n_sims} times -- not a market return prediction.",
        "terminal_r_median": round(float(np.median(terminal_r_totals)), 2),
        "terminal_r_p5": round(float(np.percentile(terminal_r_totals, 5)), 2),
        "terminal_r_p95": round(float(np.percentile(terminal_r_totals, 95)), 2),
        "max_drawdown_r_median": round(float(np.median(max_drawdowns_r)), 2),
        "max_drawdown_r_p95_worst_case": round(float(np.percentile(max_drawdowns_r, 95)), 2),
        "losing_streak_median": round(float(np.median(losing_streaks)), 1),
        "losing_streak_p95_worst_case": round(float(np.percentile(losing_streaks, 95)), 1),
        "risk_of_ruin_pct_of_sims": risk_of_ruin_pct,
        "risk_of_ruin_note": "% of simulated resamples where equity would have drawn down 50%+ of starting capital, at the assumed risk-per-trade -- illustrative, based on real historical R-multiples only.",
    }
