"""
macro_confluence.py -- MODEL B: the top-down "macro / global-led" read, used TOGETHER with the structure + strategy engine (Model A).

WHY (v50): the older, better-behaved engine (v41) took a trade only when the global research was not strongly against it.  v42/v43 turned that hard
veto into a 0.20 probability penalty, waived the win-rate bar in a "momentum lane", and capped the learned penalties -- and the engine then took five
losing trades in a row.  This module brings the old idea back as an independent second model that scores every group that is NOT price structure:

    global research (strong/moderate) . world equity average . India news . Nifty-50 stock news . Nifty-50 fundamentals
    breadth (advancers vs decliners) . order-flow pressure . PCR velocity . India VIX

It returns a signed score toward the proposed direction and one verdict:
    STRONG_ALIGNED / ALIGNED / NEUTRAL / OPPOSED / STRONG_OPPOSED
How the engine uses it (no new hard gate except the old, documented global-research veto):
    * agreement of both models  -> small probability bonus, and the only way to earn the "momentum lane" / early-opening trades
    * Model B opposed           -> probability penalty, no lane, learned penalties are NOT capped (old strict behaviour)
    * global research STRONG against the trade -> veto (as in v29..v41)
Pure function, never raises, no I/O.
"""
from __future__ import annotations

ENABLED = True

W_GLOBAL_STRONG = 3.0
W_GLOBAL_MODERATE = 1.5
W_EQUITY_AVG_BIG = 1.0      # world equity average beyond +/-0.3 %
W_EQUITY_AVG_SMALL = 0.5    # beyond +/-0.1 %
W_NEWS = 1.0                # India news / Nifty-50 stock news, each
W_FUNDAMENTALS = 1.0
W_BREADTH_BIG = 1.5         # |adv-dec| / (adv+dec) >= 0.30
W_BREADTH_SMALL = 0.75      # >= 0.10
W_ORDER_FLOW = 1.0
W_PCR = 0.5
W_VIX = 0.75                # VIX >= 20 hurts BUY, helps SELL; VIX < 14 helps BUY

STRONG_AT = 4.0
ALIGNED_AT = 2.0

PENALTY_LOGIT = {"STRONG_ALIGNED": -0.15, "ALIGNED": -0.08, "NEUTRAL": 0.0, "OPPOSED": 0.15, "STRONG_OPPOSED": 0.30}


def _sgn(direction):
    return 1.0 if direction == "BUY" else -1.0


def _lean(text):
    t = str(text or "").upper()
    if "BULL" in t or t == "BUY":
        return 1.0
    if "BEAR" in t or t == "SELL":
        return -1.0
    return 0.0


def _num(x):
    try:
        v = float(x)
        return None if v != v else v
    except Exception:
        return None


def assess(direction, global_research=None, global_avg_change=None, india_news=None, nifty50_news=None, nifty50_fundamentals=None,
           breadth_advances=None, breadth_declines=None, order_pressure="", pcr_bias="NEUTRAL", vix=None):
    out = {"verdict": "NEUTRAL", "score": 0.0, "aligned_groups": [], "opposed_groups": [], "penalty_logit": 0.0,
           "global_strong_opposed": False, "global_strong_aligned": False, "note": "macro model off"}
    if not ENABLED or direction not in ("BUY", "SELL"):
        return out
    try:
        s = _sgn(direction)
        parts = []            # (label, signed contribution toward `direction`)

        gr = global_research if isinstance(global_research, dict) else {}
        if str(gr.get("status", "")).upper() == "AVAILABLE":
            lean = _lean(gr.get("directional_bias"))
            strength = str(gr.get("strength", "")).upper()
            w = W_GLOBAL_STRONG if strength == "STRONG" else (W_GLOBAL_MODERATE if strength == "MODERATE" else 0.0)
            if lean and w:
                parts.append((f"global research {strength.lower()} {gr.get('directional_bias')}", lean * s * w))
                if strength == "STRONG":
                    out["global_strong_opposed"] = lean * s < 0
                    out["global_strong_aligned"] = lean * s > 0

        g = _num(global_avg_change)
        if g is not None:
            if abs(g) >= 0.30:
                parts.append((f"world markets avg {g:+.2f}%", (1 if g > 0 else -1) * s * W_EQUITY_AVG_BIG))
            elif abs(g) >= 0.10:
                parts.append((f"world markets avg {g:+.2f}%", (1 if g > 0 else -1) * s * W_EQUITY_AVG_SMALL))

        for label, val in (("India news", india_news), ("Nifty-50 stock news", nifty50_news)):
            l = _lean(val)
            if l:
                parts.append((label, l * s * W_NEWS))
        l = _lean(nifty50_fundamentals)
        if l:
            parts.append(("Nifty-50 fundamentals", l * s * W_FUNDAMENTALS))

        adv, dec = _num(breadth_advances), _num(breadth_declines)
        if adv is not None and dec is not None and (adv + dec) > 0:
            r = (adv - dec) / (adv + dec)
            if abs(r) >= 0.30:
                parts.append((f"breadth {int(adv)}/{int(dec)}", (1 if r > 0 else -1) * s * W_BREADTH_BIG))
            elif abs(r) >= 0.10:
                parts.append((f"breadth {int(adv)}/{int(dec)}", (1 if r > 0 else -1) * s * W_BREADTH_SMALL))

        op = str(order_pressure or "").upper()
        if op == "BUYING PRESSURE":
            parts.append(("order flow", s * W_ORDER_FLOW))
        elif op == "SELLING PRESSURE":
            parts.append(("order flow", -s * W_ORDER_FLOW))
        pb = _lean(pcr_bias)
        if pb:
            parts.append(("PCR velocity", pb * s * W_PCR))
        v = _num(vix)
        if v is not None:
            if v >= 20.0:
                parts.append((f"VIX {v:.1f} high", -s * W_VIX))     # fear: bad for BUY, supportive for SELL
            elif v < 14.0:
                parts.append((f"VIX {v:.1f} calm", s * W_VIX * (1.0 if direction == "BUY" else 0.0)))

        score = round(sum(p[1] for p in parts), 2)
        aligned = [p[0] for p in parts if p[1] > 0]
        opposed = [p[0] for p in parts if p[1] < 0]
        if score >= STRONG_AT:
            verdict = "STRONG_ALIGNED"
        elif score >= ALIGNED_AT:
            verdict = "ALIGNED"
        elif score <= -STRONG_AT:
            verdict = "STRONG_OPPOSED"
        elif score <= -ALIGNED_AT:
            verdict = "OPPOSED"
        else:
            verdict = "NEUTRAL"
        out.update({"verdict": verdict, "score": score, "aligned_groups": aligned, "opposed_groups": opposed,
                    "penalty_logit": PENALTY_LOGIT[verdict],
                    "note": (f"Macro model {verdict.replace('_', ' ')} for {direction} (score {score:+.1f}): "
                             f"for [{', '.join(aligned) or '-'}] against [{', '.join(opposed) or '-'}]")})
    except Exception:
        out["note"] = "macro model failed (ignored)"
    return out
