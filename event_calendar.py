"""
event_calendar.py -- "is today / right now a dangerous time to open a NEW trade?"

Why: a model cannot know that an RBI announcement, a Fed night or the Budget is coming.  On such days price jumps on news, not on
the OI / S-R levels the engine reads, so a perfectly "valid" setup loses.  This module is a small, EDITABLE calendar.

Levels
  BLOCK    -> no new entries inside the window (the engine returns "no setup: event risk").
  CAUTION  -> entries allowed, but the win probability is reduced by `penalty_logit`.
Add your own events in EXTRA_EVENTS below, or in a file `events_extra.json` next to app.py:
  [{"date": "2026-11-12", "start": "09:15", "end": "15:30", "level": "CAUTION", "label": "US CPI night (gap risk)"}]

Dates below were checked against the official RBI MPC schedule (press release of 23 Mar 2026) and the Fed's FOMC calendar.
Union Budget day is NOT confirmed yet -- it is usually 1 Feb; check it when it is announced.

Also exposes expiry-day flags (Nifty weekly expiry is Tuesday; the monthly one is the last Tuesday) so the learner can find out
by itself whether expiry days lose for this engine.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, time, timedelta

logger = logging.getLogger(__name__)

CAUTION_PENALTY_LOGIT = 0.25

# (date, start, end, level, label)
EVENTS = [
    # RBI policy decision is announced at 10:00 IST on the LAST day of the MPC meeting.
    ("2026-12-04", "09:15", "11:30", "BLOCK", "RBI policy announcement (10:00 IST)"),
    ("2026-12-04", "11:30", "15:30", "CAUTION", "RBI policy day (after-announcement volatility)"),
    ("2027-02-05", "09:15", "11:30", "BLOCK", "RBI policy announcement (10:00 IST)"),
    ("2027-02-05", "11:30", "15:30", "CAUTION", "RBI policy day (after-announcement volatility)"),
    # Fed decision comes at ~23:30 IST, so the Indian session NEXT morning gaps / reacts.
    ("2026-10-29", "09:15", "11:00", "CAUTION", "Reaction to the US Fed decision (28 Oct night)"),
    ("2026-12-10", "09:15", "11:00", "CAUTION", "Reaction to the US Fed decision (9 Dec night)"),
    # Budget: date not yet confirmed -> caution only, whole day
    ("2027-02-01", "09:15", "15:30", "CAUTION", "Union Budget day (confirm the date)"),
]


def _load_extra():
    out = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "events_extra.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for e in json.load(fh):
                    out.append((str(e["date"]), str(e.get("start", "09:15")), str(e.get("end", "15:30")),
                                str(e.get("level", "CAUTION")).upper(), str(e.get("label", "event"))))
    except Exception:
        logger.exception("events_extra.json could not be read (ignored)")
    return out


def _t(s):
    h, m = str(s).split(":")[:2]
    return time(int(h), int(m))


def _last_tuesday(d: date) -> bool:
    return d.weekday() == 1 and (d + timedelta(days=7)).month != d.month


def event_risk(now: datetime | None = None):
    """-> {"level": "NONE"|"CAUTION"|"BLOCK", "reason": str, "penalty_logit": float, "expiry_day": bool, "monthly_expiry": bool}"""
    out = {"level": "NONE", "reason": "", "penalty_logit": 0.0, "expiry_day": False, "monthly_expiry": False}
    try:
        now = now or datetime.now()
        d, t = now.date(), now.time()
        out["expiry_day"] = d.weekday() == 1
        out["monthly_expiry"] = _last_tuesday(d)
        worst = None
        for ds, st, en, level, label in list(EVENTS) + _load_extra():
            if ds != d.isoformat():
                continue
            if _t(st) <= t <= _t(en):
                if worst is None or (level == "BLOCK" and worst[0] != "BLOCK"):
                    worst = (level, label)
        if worst:
            out["level"], out["reason"] = worst
            out["penalty_logit"] = CAUTION_PENALTY_LOGIT if worst[0] == "CAUTION" else 0.0
    except Exception:
        logger.exception("event_risk failed (ignored)")
    return out
