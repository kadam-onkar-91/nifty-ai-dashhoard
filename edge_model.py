"""
edge_model.py -- calibrated win-probability + expectancy for the AI trade engine.

WHY THIS EXISTS
---------------
The old confidence was `50 + 1.8 * (factors that are True)` (+/- a few points of
learning).  That number was never tied to how often trades actually won, and the
entry floor for strategy-led entries was 50, so almost any setup with a handful of
True flags passed.  A "62%" setup could really be a 35% setup and nothing noticed.

This module replaces it with a small Bayesian model:

  1. PRIOR     the rule-based confluence gives a *conservative* prior win
               probability (never above PRIOR_MAX).
  2. LEARNING  every resolved trade (WIN / LOSS, plus EXPIRED trades that clearly
               moved for or against us) updates a per-factor log-odds lift, with
               shrinkage so a few lucky/unlucky trades cannot dominate.
  3. CALIB     once enough trades exist, the final probability is pulled toward the
               engine's REAL overall win rate (reliability correction).
  4. EDGE      expectancy in R:  EV = p * RR - (1 - p).  The engine only trades when
               EV is positive by a margin.  A higher R:R therefore needs a lower
               win probability, a lower R:R needs a higher one.

Nothing here claims certainty.  With fewer than MIN_TRADES_FOR_CALIBRATION resolved
trades the model says so and stays on its conservative prior.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

PRIOR_MIN = 0.30
PRIOR_MAX = 0.56            # rules alone can never claim more than this
PRIOR_AT_ZERO_FACTORS = 0.34
PRIOR_PER_FACTOR = 0.012    # each extra aligned factor adds a little (diminishing)
PRIOR_FACTOR_CAP = 18       # factors beyond this count add nothing (they are correlated)

FACTOR_MIN_SAMPLES = 8      # per-factor samples (True AND False side) before it is trusted
FACTOR_PRIOR_STRENGTH = 6.0 # pseudo-trades pulling a factor lift back to zero
MAX_FACTOR_LOGIT = 0.45     # clamp for one factor's log-odds lift
MAX_TOTAL_LOGIT = 0.60      # clamp for the sum of all learned lifts (small samples must not swing a trade by 25 pts)

MIN_TRADES_FOR_CALIBRATION = 15
CALIBRATION_STRENGTH = 20.0 # pseudo-trades behind the prior when blending with real win rate

REFERENCE_RR = 1.5          # win probability is calibrated for trades with this reward:risk
UNCALIBRATED_CAP = 0.50      # with no track record the model may not claim better than a coin flip
PROB_FLOOR = 0.15
PROB_CEILING = 0.78


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rule_prior(true_count: int) -> float:
    """Conservative prior win-probability from the number of aligned factors."""
    n = max(0, min(int(true_count), PRIOR_FACTOR_CAP))
    # diminishing returns: sqrt shape
    p = PRIOR_AT_ZERO_FACTORS + PRIOR_PER_FACTOR * math.sqrt(n) * 4.0
    return min(PRIOR_MAX, max(PRIOR_MIN, p))


def factor_lifts(rows: Iterable[Tuple[Dict, int]], keys: List[str]) -> Dict[str, Dict]:
    """Per-factor log-odds lift learned from resolved trades.

    rows : iterable of (factor_flags_dict, label) where label is 1 (win) or 0 (loss).
    Lift = shrunk log-odds ratio of win-rate when the factor is True vs False.
    A factor that is (almost) always True or always False has nothing to learn from
    and gets lift 0 -- this stops constants like `main_signal_aligned` from
    pretending to be informative.
    """
    rows = list(rows)
    out: Dict[str, Dict] = {}
    for k in keys:
        tw = tn = fw = fn = 0
        for flags, label in rows:
            if k not in flags:
                continue
            if bool(flags.get(k)):
                tn += 1
                tw += label
            else:
                fn += 1
                fw += label
        if tn < FACTOR_MIN_SAMPLES or fn < FACTOR_MIN_SAMPLES:
            out[k] = {"lift": 0.0, "n_true": tn, "n_false": fn, "trusted": False}
            continue
        p_t = (tw + 1.0) / (tn + 2.0)
        p_f = (fw + 1.0) / (fn + 2.0)
        raw = _logit(p_t) - _logit(p_f)
        shrink = min(tn, fn) / (min(tn, fn) + FACTOR_PRIOR_STRENGTH)
        lift = max(-MAX_FACTOR_LOGIT, min(MAX_FACTOR_LOGIT, raw * shrink))
        out[k] = {"lift": lift, "n_true": tn, "n_false": fn, "trusted": True,
                  "p_true": p_t, "p_false": p_f}
    return out


def estimate_edge(factor_flags: Dict, rows: Iterable[Tuple[Dict, int]], keys: List[str],
                  rr: float, extra_penalty_logit: float = 0.0, calib_rows=None) -> Dict:
    """Return calibrated win probability + expectancy for one candidate trade.

    rows must be (flags_dict, 1|0) for already-resolved trades.
    extra_penalty_logit : caller can subtract evidence (soft flags, conflicts).
    """
    rows = list(rows)
    # Factor lifts may learn from ALL history (they describe market behaviour), but the RELIABILITY correction
    # ("how often does THIS engine really win") must only use trades made by the CURRENT decision logic -- otherwise a bad
    # record from an older, since-changed logic would pull every new trade's probability down and freeze the engine.
    calib = rows if calib_rows is None else list(calib_rows)
    true_count = sum(bool(factor_flags.get(k)) for k in keys)
    prior = rule_prior(true_count)
    logit = _logit(prior)

    lifts = factor_lifts(rows, keys)
    learned = []
    total = 0.0
    for k in keys:
        info = lifts.get(k)
        if not info or not info["trusted"]:
            continue
        # a lift only applies when the factor is True on this candidate; if the factor is
        # False on this candidate, its absence counts the other way (half weight).
        total += info["lift"] if bool(factor_flags.get(k)) else -0.5 * info["lift"]
        learned.append(k)
    total = max(-MAX_TOTAL_LOGIT, min(MAX_TOTAL_LOGIT, total))
    logit += total
    n = len(calib)
    if n < MIN_TRADES_FOR_CALIBRATION:
        # No measured track record yet: the rules alone may not claim better than UNCALIBRATED_CAP.  The cap is applied
        # BEFORE the caller's penalties, so conflicts / a nearby OI wall / choppy volatility still cost probability.
        logit = min(logit, _logit(UNCALIBRATED_CAP))
        logit = min(logit - float(extra_penalty_logit or 0.0), _logit(UNCALIBRATED_CAP))   # a bonus can not lift it above the cap either
    else:
        logit -= float(extra_penalty_logit or 0.0)
    p = _sigmoid(logit)

    wins = sum(label for _, label in calib)
    calibrated = False
    if n >= MIN_TRADES_FOR_CALIBRATION:
        emp = (wins + 1.0) / (n + 2.0)
        w = n / (n + CALIBRATION_STRENGTH)
        # Reliability correction: shift this trade's probability by how far the model's AVERAGE prediction on past
        # trades was from the engine's REAL win rate.  Shift (not replace) so per-trade factor information is kept.
        p = p + w * (emp - _mean_model_prob(calib, keys, lifts))
        calibrated = True
    p = max(PROB_FLOOR, min(PROB_CEILING, p))
    p_ref = p

    rr = max(float(rr or 0.0), 0.0)
    # A farther target is harder to reach.  `p` is calibrated for a REFERENCE_RR trade; for a different
    # reward:risk, scale it the way a driftless price path behaves (P(hit target first) ~ 1/(1+RR)).
    # Without this a far OI wall (RR 5+) would look like a huge "edge" and be over-traded.
    if rr > 0:
        p = min(PROB_CEILING, p_ref * (1.0 + REFERENCE_RR) / (1.0 + rr))
    ev = p * rr - (1.0 - p)
    breakeven = 1.0 / (1.0 + rr) if rr > 0 else 1.0
    return {
        "win_probability": round(p, 4),
        "win_probability_ref_rr": round(p_ref, 4),
        "confidence_pct": round(100.0 * p, 1),
        "expectancy_r": round(ev, 3),
        "breakeven_probability": round(breakeven, 4),
        "prior": round(prior, 4),
        "learned_factors": learned,
        "learned_count": len(learned),
        "calibrated_on_trades": n if calibrated else 0,
        "empirical_win_rate": round(100.0 * wins / n, 1) if n else None,
        "sample_size": n,
    }


def _mean_model_prob(rows, keys, lifts) -> float:
    """Average probability the model would have assigned to the historical trades.
    Used to correct systematic over/under-confidence against the real win rate."""
    if not rows:
        return 0.5
    s = 0.0
    for flags, _ in rows:
        tc = sum(bool(flags.get(k)) for k in keys)
        lg = _logit(rule_prior(tc))
        tot = 0.0
        for k in keys:
            info = lifts.get(k)
            if info and info["trusted"]:
                tot += info["lift"] if bool(flags.get(k)) else -0.5 * info["lift"]
        lg += max(-MAX_TOTAL_LOGIT, min(MAX_TOTAL_LOGIT, tot))
        s += _sigmoid(lg)
    return s / len(rows)


def required_probability(rr: float, min_ev_r: float) -> float:
    """Win probability needed for EV >= min_ev_r at this reward:risk."""
    rr = max(float(rr), 1e-9)
    return (1.0 + min_ev_r) / (1.0 + rr)
