"""
pcr_velocity.py -- how FAST the put/call ratio is moving (a static PCR says little; its slope says who is being aggressive).

The dashboard refreshes every ~30 s; each refresh calls update(pcr).  History lives in memory (per server process) and is
trimmed to the last 90 minutes.  bias():  PCR rising fast  = put writing / support building  -> bullish
                                         PCR falling fast = call writing / resistance building -> bearish
"""
from __future__ import annotations

import time
from collections import deque

_HIST = deque(maxlen=400)       # (epoch_seconds, pcr)
WINDOW_MIN = 20
MIN_SPAN_MIN = 8                # need at least this much history before giving an opinion
THRESH_PER_15 = 0.05            # |change per 15 min| at/above this = a real move


def update(pcr, now=None):
    try:
        v = float(pcr)
    except Exception:
        return
    if not (0.0 < v < 10.0):
        return
    t = time.time() if now is None else float(now)
    if _HIST and t - _HIST[-1][0] < 20:          # ignore re-runs within 20 s
        _HIST[-1] = (t, v)
        return
    _HIST.append((t, v))
    while _HIST and t - _HIST[0][0] > 90 * 60:
        _HIST.popleft()


def reset():
    _HIST.clear()


def bias(now=None):
    """-> {"bias": "BUY"|"SELL"|"NEUTRAL", "per15": float|None, "span_min": float, "note": str}"""
    t = time.time() if now is None else float(now)
    pts = [(ts, v) for ts, v in _HIST if t - ts <= WINDOW_MIN * 60]
    if len(pts) < 3:
        return {"bias": "NEUTRAL", "per15": None, "span_min": 0.0, "note": "PCR velocity: not enough history yet"}
    span = (pts[-1][0] - pts[0][0]) / 60.0
    if span < MIN_SPAN_MIN:
        return {"bias": "NEUTRAL", "per15": None, "span_min": round(span, 1), "note": "PCR velocity: collecting history"}
    per15 = (pts[-1][1] - pts[0][1]) / span * 15.0
    if per15 >= THRESH_PER_15:
        b, txt = "BUY", "PCR rising fast (put writing / support building)"
    elif per15 <= -THRESH_PER_15:
        b, txt = "SELL", "PCR falling fast (call writing / resistance building)"
    else:
        b, txt = "NEUTRAL", "PCR flat"
    return {"bias": b, "per15": round(per15, 3), "span_min": round(span, 1),
            "note": f"{txt}: {per15:+.3f} per 15 min over {span:.0f} min"}
