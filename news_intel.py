"""news_intel.py -- fundamentals verdict, news sentiment verdict and an upcoming-events planner (stocks AND Nifty 50).

Design rules (kept on purpose):
  * Verdicts are computed locally from data the page already has -> adds ~0 ms, never a network call, never slows load/refresh.
  * The upcoming-events table needs Google Search through Gemini, so it only runs when the person taps the button (cached 30 min).
  * Everything is labelled as rule-based / AI-estimated.  It is a planning aid, not a prediction and not an order trigger.
"""
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

IST = timezone(timedelta(hours=5, minutes=30))
EVENTS_TTL_S = 1800

# ----------------------------------------------------------------------------------------------------------------- fundamentals
def _f(v):
    try:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.replace(",", "").replace("%", "").strip()
            if v in ("", "N/A", "None", "-"):
                return None
        x = float(v)
        return x if x == x else None
    except Exception:
        return None


def _pick(d, key, index_mode):
    v = _f(d.get(key + "_mean")) if index_mode else _f(d.get(key))
    if v is None and not index_mode:
        v = _f(d.get(key + "_mean"))
    return v


def fundamentals_verdict(f, is_index=False):
    """-> {"verdict": POSITIVE|NEGATIVE|NEUTRAL|UNCLEAR, "score": float, "points": [(label, text, +1/0/-1)]}"""
    f = f or {}
    pts = []
    if f.get("source") == "Upstox Company Fundamentals" and isinstance(f.get("ratio_map"), dict):
        rm = f["ratio_map"]
        g = lambda k: _f((rm.get(k) or {}).get("company_value"))
        s = lambda k: _f((rm.get(k) or {}).get("sector_value"))
        pe, spe, roe, sroe = g("P/E"), s("P/E"), g("ROE"), s("ROE")
        if pe is not None and spe:
            pts.append(("P/E vs sector", f"{pe:.1f} vs {spe:.1f}", 1 if pe <= spe * 0.9 else -1 if pe >= spe * 1.4 else 0))
        elif pe is not None:
            pts.append(("P/E", f"{pe:.1f}", -1 if pe < 0 else 1 if pe <= 25 else -1 if pe > 50 else 0))
        if roe is not None:
            ref = sroe if sroe else 12.0
            pts.append(("ROE", f"{roe:.1f}%", 1 if roe >= max(15.0, ref) else -1 if roe < 8 else 0))
        for k in ("ROCE", "ROA"):
            v = g(k)
            if v is not None:
                pts.append((k, f"{v:.1f}%", 1 if v >= (12 if k == "ROCE" else 2) else -1 if v < (6 if k == "ROCE" else 0.5) else 0))
    else:
        pe, fpe = _pick(f, "trailingPE", is_index), _pick(f, "forwardPE", is_index)
        roe, pm = _pick(f, "returnOnEquity", is_index), _pick(f, "profitMargins", is_index)
        rg, eg = _pick(f, "revenueGrowth", is_index), _pick(f, "earningsGrowth", is_index)
        de, pb = _pick(f, "debtToEquity", is_index), _pick(f, "priceToBook", is_index)
        if pe is not None and not is_index:                       # mean P/E of 50 stocks is distorted by outliers -> not scored for the index
            pts.append(("P/E", f"{pe:.1f}", -1 if pe <= 0 else 1 if pe <= 25 else -1 if pe > 50 else 0))
        if fpe is not None and pe is not None and pe > 0 and fpe > 0 and not is_index:
            pts.append(("Forward P/E vs P/E", f"{fpe:.1f} vs {pe:.1f}", 1 if fpe < pe * 0.85 else -1 if fpe > pe * 1.1 else 0))
        if pb is not None and not is_index:
            pts.append(("P/B", f"{pb:.2f}", 1 if 0 < pb <= 3 else -1 if pb > 8 or pb <= 0 else 0))
        if roe is not None:
            pts.append(("ROE", f"{roe*100:.1f}%", 1 if roe >= 0.15 else -1 if roe < 0.08 else 0))
        if pm is not None:
            pts.append(("Profit margin", f"{pm*100:.1f}%", 1 if pm >= 0.12 else -1 if pm < 0.04 else 0))
        if rg is not None:
            pts.append(("Revenue growth", f"{rg*100:.1f}%", 1 if rg >= 0.10 else -1 if rg < 0 else 0))
        if eg is not None:
            pts.append(("Earnings growth", f"{eg*100:.1f}%", 1 if eg >= 0.10 else -1 if eg < 0 else 0))
        sector = str(f.get("sector") or "").lower()
        if de is not None and not is_index and "financial" not in sector and "bank" not in sector:
            pts.append(("Debt/Equity", f"{de:.0f}%", 1 if de <= 60 else -1 if de > 150 else 0))
    if len(pts) < 3:
        return {"verdict": "UNCLEAR", "score": 0.0, "points": pts}
    score = sum(p[2] for p in pts) / len(pts)
    verdict = "POSITIVE" if score >= 0.25 else "NEGATIVE" if score <= -0.25 else "NEUTRAL"
    return {"verdict": verdict, "score": round(score, 2), "points": pts}


# ------------------------------------------------------------------------------------------------------------------------- news
_POS = ("surge", "rally", "rallies", "jump", "jumps", "soar", "gain", "gains", "rise", "rises", "rose", "higher", "record high", "upgrade", "upgraded",
        "beat", "beats", "strong", "boost", "inflow", "inflows", "profit rises", "profit jumps", "bullish", "outperform", "buy rating", "order win",
        "wins order", "bags order", "approval", "approved", "recovery", "rebound", "positive", "growth", "dividend", "bonus", "top gainer", "upper circuit",
        "snap losing", "extend rally", "all-time high", "lower crude", "cooling inflation", "rate cut")
_NEG = ("crash", "crashes", "plunge", "plunges", "slump", "tumble", "fall", "falls", "fell", "slip", "slips", "drop", "drops", "lower circuit", "loss",
        "losses", "downgrade", "downgraded", "probe", "fraud", "penalty", "fine", "ban", "bars", "rejects", "rejected", "weak", "cut", "cuts", "selloff",
        "sell-off", "default", "bearish", "underperform", "sell rating", "resigns", "resignation", "raid", "scam", "debt", "layoff", "shutdown", "war",
        "tariff", "top loser", "snap winning", "profit falls", "profit plunges", "margin pressure", "rate hike", "outflow", "outflows", "warning")


def _age_hours(published, now=None):
    now = now or datetime.now(timezone.utc)
    if not published:
        return None
    try:
        dt = parsedate_to_datetime(str(published))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 3600.0)
    except Exception:
        return None


def _headline_score(title):
    t = " " + str(title or "").lower() + " "
    pos = sum(1 for w in _POS if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t))
    neg = sum(1 for w in _NEG if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t))
    m = re.search(r"(crash|plung|tumbl|slump|fall|fell|drop|slip|slid|dip)\w*\s+(?:by\s+)?(\d+(?:\.\d+)?)\s?%", t)
    if m and float(m.group(2)) >= 3:
        neg += 1
    return pos - neg


def news_verdict(items, now=None):
    """-> {"verdict", "score", "rows": [(sentiment, age_text, title, source, link)], "fresh": n}"""
    rows, wsum, ssum, fresh = [], 0.0, 0.0, 0
    for n in (items or [])[:15]:
        title = n.get("title") or ""
        sc = _headline_score(title)
        age = _age_hours(n.get("published"), now)
        w = 0.0 if age is None else 1.0 if age <= 24 else 0.6 if age <= 72 else 0.3 if age <= 168 else 0.0
        if w >= 0.6:
            fresh += 1
        wsum += w
        ssum += w * max(-1, min(1, sc))
        label = "🟢 Positive" if sc > 0 else "🔴 Negative" if sc < 0 else "⚪ Neutral"
        age_txt = "time unknown" if age is None else (f"{age:.0f}h purani" if age < 48 else f"{age/24:.0f} din purani")
        if w == 0.0 and age is not None:
            age_txt += " (verdict me nahi gini)"
        rows.append((label, age_txt, title, n.get("source") or "", n.get("link") or ""))
    if wsum == 0:
        return {"verdict": "NEUTRAL", "score": 0.0, "rows": rows, "fresh": 0, "note": "Koi fresh (7 din ke andar) news nahi mili."}
    score = ssum / wsum
    verdict = "POSITIVE" if score >= 0.25 else "NEGATIVE" if score <= -0.25 else "NEUTRAL"
    return {"verdict": verdict, "score": round(score, 2), "rows": rows, "fresh": fresh, "note": ""}


# ----------------------------------------------------------------------------------------------------------------------- events
def _us_open_ist(now_ist):
    try:
        from zoneinfo import ZoneInfo
        d = now_ist.astimezone(ZoneInfo("America/New_York")).replace(hour=9, minute=30, second=0, microsecond=0)
        return d.astimezone(IST).strftime("%H:%M")
    except Exception:
        return "19:00-20:00"


def fixed_calendar(now_ist=None):
    """Deterministic items (no network): weekly expiry, US open, data-release pattern."""
    now_ist = now_ist or datetime.now(IST)
    d = now_ist.date()
    days = (1 - d.weekday()) % 7                      # Tuesday = Nifty weekly expiry (as shown on the Upstox header)
    exp = d + timedelta(days=days)
    out = [{"event": "Nifty weekly expiry", "date_ist": exp.isoformat(), "time_ist": "15:30", "type": "expiry", "expected_effect": "VOLATILE",
            "why": "Expiry din option gamma badhta hai; 14:00-15:30 me tez, jhatke wali moves aati hain.", "reaction_window": "14:00-15:30 IST"},
           {"event": "US cash market open (US data/Fed-linked moves yahin se Nifty tak aate hain)", "date_ist": d.isoformat(), "time_ist": _us_open_ist(now_ist),
            "type": "global", "expected_effect": "VOLATILE", "why": "US open ke baad global risk-mood badalta hai, raat ko India futures/GIFT Nifty pe dikhta hai.",
            "reaction_window": "open ke baad pehle 30 min"}]
    return out


TYPICAL_REACTION = [
    ("RBI policy / Budget", "Announcement ke turant baad 5-15 min me bada spike, phir 1-2 ghante me direction settle hoti hai"),
    ("India CPI/WPI (12/14 tareekh, ~17:30 IST)", "Market band hone ke baad aata hai -> asar agle din ke open (gap) me"),
    ("US CPI / Jobs / FOMC (raat ~19:00-00:30 IST)", "Nifty pe asar agle din subah ke open ya GIFT Nifty se; badi surprise pe gap"),
    ("Company results (aksar market ke baad)", "Agle din open par gap; pehle 15-30 min whipsaw, phir trend"),
    ("Court / regulatory order", "Aane ke turant baad tez move; aksar 1-3 candles me pura reaction nikal jata hai"),
]


def _extract_json(txt):
    txt = re.sub(r"```(?:json)?", "", txt or "").strip()
    a, b = txt.find("["), txt.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        data = json.loads(txt[a:b + 1])
        return [x for x in data if isinstance(x, dict) and x.get("event")][:10]
    except Exception:
        return []


def fetch_events(keys, subject, now_ist=None):
    """Google-Search grounded Gemini call.  Returns (events, error_text)."""
    now_ist = now_ist or datetime.now(IST)
    try:
        import requests
        import gemini_pool
        import ai_trade_decision as atd
    except Exception as exc:
        return [], f"modules load nahi hue: {exc}"[:160]
    prompt = (f"Aaj {now_ist.strftime('%Y-%m-%d %H:%M')} IST hai. Google Search se agle 7 din ke SCHEDULED, verifiable market-moving events batao jo {subject} ko "
              "asar kar sakte hain (RBI/Fed/CPI/jobs data, results/board meeting/AGM, court hearings, index events, global cues). Sirf wahi jo sach me confirm hon; "
              "invent mat karo. Agar exact time confirm nahi to time_ist 'unknown' likho. Sirf JSON array do (max 8), koi aur text nahi. Format: "
              '[{"event":"..","date_ist":"YYYY-MM-DD","time_ist":"HH:MM ya unknown","type":"macro|company|global|regulatory",'
              '"expected_effect":"UP|DOWN|VOLATILE|SIDEWAYS|UNCLEAR","why":"1 line","reaction_window":"e.g. release ke baad pehle 15 min"}]')
    body = {"contents": [{"parts": [{"text": prompt}]}], "tools": [{"google_search": {}}], "generationConfig": {"maxOutputTokens": 900}}
    try:                                            # share the trade-review quota cooldown: if Gemini search is rate-limited, do not add more calls
        if time.time() < atd._SEARCH_COOL["until"]:
            return [], "Gemini search abhi quota cooldown me hai (trade review ko quota bachane ke liye) -- thodi der baad try karo."
    except Exception:
        pass
    models = sorted(list(atd.REVIEW_MODELS), key=lambda m: 0 if "lite" in m else 1)     # cheapest models first

    def _one(key, model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = requests.post(url, headers={"Content-Type": "application/json", "x-goog-api-key": key}, json=body, timeout=40)
        if r.status_code != 200:
            raise gemini_pool.PoolHTTPError(r.status_code, r.text[:200])
        cand = (r.json().get("candidates") or [{}])[0]
        txt = "".join(p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []) if not p.get("thought"))
        ev = _extract_json(txt)
        if not ev and "[]" not in txt:
            raise gemini_pool.PoolHTTPError(502, "no parsable events")
        return ev

    try:
        ev, _i, _m = gemini_pool.run(keys, models, _one, max_attempts=4, deadline_s=40.0)
        return ev, ""
    except Exception as exc:
        return [], f"Gemini events nahi de paya: {exc}"[:200]


# ---------------------------------------------------------------------------------------------------------------------- render
_BADGE = {"POSITIVE": "🟢 POSITIVE", "NEGATIVE": "🔴 NEGATIVE", "NEUTRAL": "⚪ NEUTRAL", "UNCLEAR": "⚪ UNCLEAR (data kam)"}
_EFFECT = {"UP": "🟢 UP", "DOWN": "🔴 DOWN", "VOLATILE": "🟠 VOLATILE (tez jhatke)", "SIDEWAYS": "⚪ SIDEWAYS", "UNCLEAR": "⚪ unclear"}


def _sig(c):
    return "🟢" if c > 0 else "🔴" if c < 0 else "⚪"


def render_fundamentals(st, fundamentals, is_index=False):
    """One compact line INSIDE the existing Fundamentals expander (no new boxes)."""
    try:
        fv = fundamentals_verdict(fundamentals, is_index=is_index)
        st.markdown(f"**Verdict: {_BADGE[fv['verdict']]}**")
        if fv["points"]:
            st.caption(" · ".join(f"{a} {b} {_sig(c)}" for a, b, c in fv["points"]) + "  — rule-based andaza, advice nahi.")
    except Exception:
        pass


def render_news(st, subject, news_items, key="x"):
    """Verdict + upcoming events INSIDE the existing News expander (no new boxes)."""
    try:
        nv = news_verdict(news_items)
        sc = [r[0] for r in nv["rows"]]
        st.markdown(f"**News verdict: {_BADGE[nv['verdict']]}**  ·  🟢 {sum('Positive' in x for x in sc)}  🔴 {sum('Negative' in x for x in sc)}  ⚪ {sum('Neutral' in x for x in sc)}")
        st.caption(nv["note"] or "Sirf 7 din ke andar ki news ginti me; headline ke neeche har ek ka sentiment: " +
                   " | ".join(f"{r[0].split()[0]} {r[2][:38]}…" for r in nv["rows"][:4]))
        ck = f"_ni_events_{key}"
        if st.button("🗓️ Agle events (kab, kitna asar) — Google se", key=f"ni_btn_{key}"):
            with st.spinner("Events dhoondh raha hoon..."):
                ev, err = fetch_events(st.session_state.get("_news_intel_keys") or [], subject)
            st.session_state[ck] = (time.time(), ev, err)
        rows = fixed_calendar()
        cached = st.session_state.get(ck)
        if cached and time.time() - cached[0] < EVENTS_TTL_S:
            if cached[2]:
                st.warning(cached[2])
            rows = rows + list(cached[1])
        rows = sorted(rows, key=lambda r: (str(r.get("date_ist")), str(r.get("time_ist"))))
        st.dataframe([{"Date": r.get("date_ist"), "Time IST": r.get("time_ist"), "Event": r.get("event"),
                       "Asar": _EFFECT.get(str(r.get("expected_effect")).upper(), r.get("expected_effect")),
                       "Kab tak": r.get("reaction_window")} for r in rows], hide_index=True, width="stretch")
        st.caption("Events AI/Google-estimate hain, date-time khud verify karo; asar ek aam pattern hai, guarantee nahi. "
                   "Result aksar market ke baad -> agle din gap; RBI/Budget -> 5-15 min spike; US data -> agla open/GIFT Nifty.")
    except Exception:
        pass
