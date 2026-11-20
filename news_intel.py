"""news_intel.py -- fundamentals verdict, news sentiment verdict and an upcoming-events planner (stocks AND Nifty 50).

Design rules (kept on purpose):
  * Verdicts are computed locally from data the page already has -> adds ~0 ms, never a network call, never slows load/refresh.
  * v66: the events table is built in BACKGROUND threads only (RSS headlines -> NVIDIA model; Gemini+Google only as fallback).  The page just reads a dict.
    Rows that are already over (and whose market reaction is finished) are dropped on every render, so only UPCOMING / still-reacting events are shown.
  * Everything is labelled as rule-based / AI-estimated.  It is a planning aid, not a prediction and not an order trigger.
"""
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

IST = timezone(timedelta(hours=5, minutes=30))
EVENTS_TTL_S = 7200

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


_DAYS = ["Somvar", "Mangalvar", "Budhvar", "Guruvar", "Shukravar", "Shanivar", "Ravivar"]


def _next_trading_day(d):
    d = d + timedelta(days=1)
    while d.weekday() >= 5:                       # Sat/Sun (exchange holidays are not known offline)
        d += timedelta(days=1)
    return d


def _label(d, today):
    if d == today:
        return "Aaj"
    if d == today + timedelta(days=1):
        return "Kal"
    return f"{_DAYS[d.weekday()]} {d.strftime('%d-%b')}"


def india_impact(date_str, time_str, now_ist=None):
    """Event ka asar INDIAN market (09:15-15:30 IST) me kab dikhega, plain Hinglish text."""
    now_ist = now_ist or datetime.now(IST)
    today = now_ist.date()
    try:
        d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    except Exception:
        return "date confirm nahi -- khud check karo"
    try:
        hh, mm = [int(x) for x in str(time_str).split(":")[:2]]
        mins = hh * 60 + mm
    except Exception:
        mins = None
    if d.weekday() >= 5:
        nd = _next_trading_day(d - timedelta(days=1))
        return f"Market band (weekend) -> {_label(nd, today)} 09:15 open par gap"
    if mins is None:
        return f"{_label(d, today)} -- time unknown, din bhar dhyan rakho"
    if mins < 9 * 60 + 15:
        return f"{_label(d, today)} 09:15 open par gap (event market khulne se pehle)"
    if mins <= 15 * 60 + 30:
        end = min(mins + 30, 15 * 60 + 30)
        return f"{_label(d, today)} {hh:02d}:{mm:02d} se {end//60:02d}:{end%60:02d} tak (market chalu -- turant asar)"
    nd = _next_trading_day(d)
    return f"India band hone ke baad -> {_label(nd, today)} 09:15 open par gap"


def fixed_calendar(now_ist=None):
    """Deterministic items (no network): weekly expiry, US open, data-release pattern."""
    now_ist = now_ist or datetime.now(IST)
    d = now_ist.date()
    days = (1 - d.weekday()) % 7                      # Tuesday = Nifty weekly expiry (as shown on the Upstox header)
    exp = d + timedelta(days=days)
    out = [{"event": "Nifty weekly expiry", "date_ist": exp.isoformat(), "time_ist": "15:30", "type": "expiry", "expected_effect": "VOLATILE", "impact": 4 if exp == d else 3, "india_text": f"{_label(exp, d)} 14:00 se 15:30 tak (market chalu, aakhri ghante me tez jhatke)",
            "why": "Expiry din option gamma badhta hai; 14:00-15:30 me tez, jhatke wali moves aati hain.", "reaction_window": "14:00-15:30 IST"},
           {"event": "US market open (India me tab band hota hai)", "date_ist": d.isoformat(), "time_ist": _us_open_ist(now_ist),
            "type": "global", "expected_effect": "VOLATILE", "impact": 3, "why": "US ka mood raat ko GIFT Nifty me dikhta hai.", "reaction_window": ""}]
    if d.weekday() >= 5:                                                   # US is closed at the weekend -> do not list its 'open'
        out = out[:1]
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
        return [x for x in data if isinstance(x, dict) and x.get("event")][:16]
    except Exception:
        return []


import threading

_EVENTS = {}            # key -> {"ts", "status": running|done|error, "rows", "err"}   (filled by a background thread, never touches st.*)
_EV_LOCK = threading.Lock()


def fetch_events(keys, subject, headlines=None, now_ist=None):
    """Google-Search grounded Gemini call (blocking; run it in the background thread).  Returns (rows, error_text).
    Covers what HAPPENED in the last ~48h and what is COMING in the next 7 days, with where/when it can hit the Indian market."""
    now_ist = now_ist or datetime.now(IST)
    try:
        import requests
        import gemini_pool
        import ai_trade_decision as atd
    except Exception as exc:
        return [], f"modules load nahi hue: {exc}"[:160]
    try:
        if time.time() < getattr(atd, "_SEARCH_COOL", {"until": 0})["until"]:
            return [], "Gemini search abhi quota cooldown me hai (trade review ko quota bachane ke liye) -- thodi der baad try karo."
    except Exception:
        pass
    heads = "\n".join(f"- {h}" for h in (headlines or [])[:10])
    prompt = (f"Aaj {now_ist.strftime('%Y-%m-%d %H:%M')} IST hai. Google Search se INDIAN market ({subject}) ke liye: (1) pichhle 48 ghante me jo market-moving "
              "news/event HO CHUKA, aur (2) agle 7 din me jo SCHEDULED event AANE WALA hai -- dono batao. Cover karo: RBI/monetary policy, sarkari policy/budget/GST, India data (CPI/WPI/GDP/IIP), FII/DII flows, Fed/US data, crude/dollar, geopolitics, sector news, bade stocks/results. Har item me bata: kya hai, kab (IST), kis par asar "
              "(Nifty / Bank Nifty / IT / PSU banks / specific stock), direction, aur Indian market (09:15-15:30 IST) me asar kab aur kahan padta hai. "
              "Sirf verifiable cheezein; invent mat karo; time confirm nahi to 'unknown'. Ye headlines bhi context hain:\n" + heads +
              "\nSirf JSON array do (max 14), koi aur text nahi. Format: "
              '[{"event":"..","status":"happened|upcoming","date_ist":"YYYY-MM-DD","time_ist":"HH:MM ya unknown","type":"RBI|Policy|Data|Global|Sector|Stock|Results","affects":"kis par asar","duration":"asar kitni der (e.g. 30 min / pura din / 2-3 din)",'
              '"expected_effect":"UP|DOWN|VOLATILE|SIDEWAYS|UNCLEAR","why":"1 line","india_note":"India market me asar kab/kahan, 1 line"}]')
    body = {"contents": [{"parts": [{"text": prompt}]}], "tools": [{"google_search": {}}], "generationConfig": {"maxOutputTokens": 2600}}
    try:                                                      # v66: ai_trade_decision.REVIEW_MODELS was removed -> this line crashed the whole events thread
        models = sorted(list(getattr(atd, "REVIEW_MODELS", None) or ("gemini-2.5-flash-lite", "gemini-2.5-flash")), key=lambda m: 0 if "lite" in m else 1)
    except Exception:
        models = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

    def _one(key, model):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = requests.post(url, headers={"Content-Type": "application/json", "x-goog-api-key": key}, json=body, timeout=40)
        if r.status_code != 200:
            raise gemini_pool.PoolHTTPError(r.status_code, r.text[:200])
        cand = (r.json().get("candidates") or [{}])[0]
        txt = "".join(p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []) if not p.get("thought"))
        ev = _extract_json(txt)
        if not ev:
            raise gemini_pool.PoolHTTPError(502, "no parsable events")
        return ev

    try:
        ev, _i, _m = gemini_pool.run(keys, models, _one, max_attempts=4, deadline_s=45.0)
        return ev, ""
    except Exception as exc:
        return [], f"Gemini events nahi de paya: {exc}"[:200]


def start_events(key, keys, subject, headlines=None):
    """Start the lookup in a background thread and return at once -- the page never waits, so the live-refresh fragment can not cancel it."""
    with _EV_LOCK:
        cur = _EVENTS.get(key)
        if cur and cur["status"] == "running" and time.time() - cur["ts"] < 90:
            return
        _EVENTS[key] = {"ts": time.time(), "status": "running", "rows": [], "err": ""}

    def _work():
        rows, err = fetch_events(keys, subject, headlines)
        with _EV_LOCK:
            _EVENTS[key] = {"ts": time.time(), "status": "error" if (err and not rows) else "done", "rows": rows, "err": err}

    threading.Thread(target=_work, daemon=True).start()


# ---------------------------------------------------------------------------------------------------------------------- v66: BEST-ONLY events
# Every page (Nifty 50 and every single stock) gets ITS OWN list.  Pipeline (all in a BACKGROUND thread -- the page only reads a dict, so load/refresh speed is unchanged):
#   1. real dated Google-News RSS headlines (parallel, 6 s timeout, ~1-3 s total)      2. NVIDIA NIM model (NOT Gemini) turns them into upcoming / still-reacting events + impact 1-5
#   3. only if NVIDIA fails: Gemini+Google fallback, then a pure rule-based headline reader (so the table is never empty because an API is down)
# The LLM is called only when the headline set CHANGED (fingerprint) -> fresh news is noticed within ~10 min, quota is not burnt.
# On every render rows that are already over (and whose market reaction is finished) are dropped -> "ho chuka" never clutters the table.

RECHECK_S = 600            # headlines re-read every 10 min (cheap RSS); the LLM only runs again if they changed
RETRY_FAIL_S = 180         # after a failure retry sooner
LLM_MAX_AGE_S = 3 * 3600   # even with identical headlines, re-ask the model every 3 h (dates move closer)
_MON = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}

_STOCK_STOP = re.compile(r"\b(ltd|limited|co|corp|corporation|company|ven|ventures?|industries|inds?)\b\.?", re.I)


def stock_queries(symbol, company="", sector=""):
    """-> [(label, google-news query)] -- all about THIS stock (+ its sector).  Forward-looking query first: it is what finds 'results on 21 Nov', 'record date', 'AGM'."""
    sym = re.sub(r"[^A-Za-z0-9&\-]", "", str(symbol or ""))
    comp = re.sub(r"\s+", " ", str(company or "").strip())
    short = re.sub(r"\s+", " ", _STOCK_STOP.sub(" ", comp)).strip() or sym
    who = f'"{comp}"' if comp else sym
    sec = re.sub(r"\s+", " ", str(sector or "").strip())
    q = [("Company", f'{who} OR "{short}" share when:3d'),
         ("Aane wala", f'"{short}" "board meeting" OR "to announce" OR "results on" OR "record date" OR "ex-dividend" OR AGM OR "to consider" when:21d'),
         ("Results/Board", f'"{short}" results OR earnings OR dividend OR guidance when:10d'),
         ("Orders/Deals", f'"{short}" order OR contract OR acquisition OR "block deal" OR stake OR fundraise OR QIP when:10d'),
         ("Regulator/Court", f'"{short}" SEBI OR NSE OR court OR NCLT OR promoter OR pledge OR rating OR upgrade OR downgrade when:10d')]
    if sec and sec.lower() not in ("unknown", "n/a", "none", "nan"):
        q.append(("Sector", f"{sec} sector India stocks policy OR demand OR prices when:5d"))
    return q


def nifty_queries():
    """Index-level queries: policy, global cues, flows, data, heavyweights, expiry -- nothing about one single stock."""
    return [("RBI/Policy", "RBI OR MPC OR \"repo rate\" OR SEBI India markets when:7d"),
            ("US Fed/Data", "Fed OR FOMC OR Powell OR \"US CPI\" OR \"US jobs\" OR \"Wall Street\" when:4d"),
            ("FII/DII", "FII OR DII flows Nifty Sensex when:3d"),
            ("Crude/Dollar", "crude oil OR \"dollar index\" OR rupee OR \"bond yield\" India market when:3d"),
            ("India data", "India CPI OR WPI OR GDP OR IIP OR GST OR budget OR \"trade deficit\" when:14d"),
            ("Heavyweights", "Reliance OR \"HDFC Bank\" OR Infosys OR TCS OR ICICI OR Bharti OR ITC results OR earnings OR \"board meeting\" when:10d"),
            ("Expiry/NSE", "Nifty expiry OR \"F&O\" OR \"index rebalancing\" OR \"NSE holiday\" OR \"market holiday\" when:14d"),
            ("Geopolitics", "war OR tariff OR sanctions OR geopolitical markets Asia India when:3d")]


def _gather(queries, per_query=6, max_age_days=14, cap=26):
    out, seen = [], set()
    try:
        import news_rss
    except Exception:
        return out
    from concurrent.futures import ThreadPoolExecutor

    def _one(lq):
        try:
            return lq[0], news_rss.parse_items(news_rss._fetch(lq[1], timeout=6), max_age_h=max_age_days * 24, limit=per_query)
        except Exception:
            return lq[0], []
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(queries)))) as ex:
            res = list(ex.map(_one, queries))
    except Exception:
        return out
    now = datetime.now(IST)
    for label, items in res:
        for age_min, title, src in items:
            k = re.sub(r"\W+", "", title.lower())[:60]
            if k in seen:
                continue
            seen.add(k)
            when = now - timedelta(minutes=age_min)
            out.append({"label": label, "title": title, "src": src or "news", "age_h": round(age_min / 60.0, 1), "date": when.strftime("%Y-%m-%d %H:%M")})
    out.sort(key=lambda r: r["age_h"])
    out = out[:cap]
    for i, r in enumerate(out, 1):
        r["n"] = i
    return out


def gather_stock_headlines(symbol, company="", sector="", per_query=6, max_age_days=21):
    """Real, dated headlines for ONE stock (parallel, short timeouts, never raises).  -> [{'n','label','title','src','age_h','date'}]"""
    return _gather(stock_queries(symbol, company, sector), per_query=per_query, max_age_days=max_age_days, cap=26)


def gather_nifty_headlines(per_query=5, max_age_days=14):
    return _gather(nifty_queries(), per_query=per_query, max_age_days=max_age_days, cap=30)


def _fingerprint(heads):
    import hashlib
    t = "|".join(sorted(re.sub(r"\W+", "", h["title"].lower())[:50] for h in (heads or [])[:20]))
    return hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()


_RULES = ("Rules: SIRF headlines me jo likha hai wahi (ya uska seedha asar). Har row me 'evidence' = headline ka number. Date/time headline me na ho to 'unknown' likho -- date invent mat karo. "
          "SIRF kaam ki cheezein do: (a) AANE WALI event (results / board meeting / dividend record date / ex-date / AGM / order / policy / data release / court-regulator decision) jiska price par asar padega, "
          "(b) pichhle 24 ghante ki badi news JISKA ASAR MARKET ME ABHI TAK POORA NAHI AAYA (jaise market band hone ke baad aayi). Jo news/event ho chuka aur uska asar nikal chuka -- MAT do. "
          "Sirf best/high-impact rows do (max 8); kam important ya generic cheez mat do; khaali list bhi theek hai. 'impact' 1-5 (5 = bahut bada move, 3 = dhyan dene layak, 1-2 = mat do).\n")
_FMT = ('Sirf JSON array do, koi aur text nahi. Format:\n'
        '[{"event":"..","status":"upcoming|happened","date_ist":"YYYY-MM-DD ya unknown","time_ist":"HH:MM ya unknown","type":"Results|Order|Deal|Regulatory|Corporate|Sector|Rating|Policy|Data|Global|News",'
        '"affects":"kis par asar","expected_effect":"UP|DOWN|VOLATILE|SIDEWAYS|UNCLEAR","impact":1-5,"impact_start":"asar kab se shuru (e.g. results ke baad agle din 09:15 open par, turant, 1-2 din me)",'
        '"duration":"asar kitni der/din","why":"1 line, kyun asar padega","evidence":<headline number ya null>}]')


def _head_lines(heads):
    return "\n".join(f"[{h['n']}] ({h['date']} IST, {h['label']}, {h['src']}) {h['title']}" for h in heads) or "(koi headline nahi mili)"


def _stock_prompt(symbol, company, sector, heads, extra_titles, now_ist):
    old = "\n".join(f"- {t}" for t in (extra_titles or [])[:8])
    return (f"Aaj {now_ist.strftime('%Y-%m-%d %A %H:%M')} IST hai (Indian market 09:15-15:30 IST). Tum Indian equity research analyst ho. Stock: {symbol} ({company or symbol}), sector: {sector or 'unknown'}.\n"
            "Neeche is stock ki REAL headlines hain (number, time, source ke saath). Inhe padhkar sirf wahi events batao jo IS STOCK ke price ko aage hilayengi, aur batao asar KAB padega.\n"
            + _RULES + "Nifty ke generic events (expiry, US open) MAT do.\n\n"
            f"HEADLINES:\n{_head_lines(heads)}\n\nDashboard ki purani headlines (context):\n{old}\n\n" + _FMT)


def _nifty_prompt(heads, extra_titles, now_ist):
    old = "\n".join(f"- {t}" for t in (extra_titles or [])[:8])
    return (f"Aaj {now_ist.strftime('%Y-%m-%d %A %H:%M')} IST hai (Indian market 09:15-15:30 IST). Tum Indian index (NIFTY 50) ke analyst ho.\n"
            "Neeche index-level REAL headlines hain. Sirf wahi events batao jo poore NIFTY 50 / Bank Nifty / bade sectors ko hilayengi: RBI/policy, Fed/US data, India data (CPI/GDP/IIP), FII/DII, crude/dollar, geopolitics, "
            "heavyweights (Reliance, HDFC Bank, Infosys, TCS, ICICI...) ke results/board meeting. Kisi ek chhote stock ki news MAT do.\n"
            + _RULES + "Har row ke 'affects' me batao: Nifty / Bank Nifty / IT / PSU banks / Reliance jaisa kaun sa hissa.\n\n"
            f"HEADLINES:\n{_head_lines(heads)}\n\nDashboard ki purani headlines (context):\n{old}\n\n" + _FMT)


def _clean_stock_events(rows, heads):
    by_n = {h["n"]: h for h in heads}
    out = []
    for r in rows or []:
        if not isinstance(r, dict) or not r.get("event"):
            continue
        ev = r.get("evidence")
        try:
            ev = int(ev)
        except Exception:
            ev = None
        h = by_n.get(ev)
        r = dict(r)
        r["evidence_text"] = (f"[{ev}] {h['title'][:90]} ({h['src']}, {h['date'][:10]})" if h else "AI andaza -- headline se match nahi hua")
        if not h:
            r["event"] = str(r["event"])[:150] + " (AI andaza)"
        d = str(r.get("date_ist") or "unknown")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            r["date_ist"] = "unknown"
        try:
            r["impact"] = max(1, min(5, int(float(r.get("impact")))))
        except Exception:
            r["impact"] = 3
        out.append(r)
    out.sort(key=lambda x: (str(x.get("status", "")).lower().startswith("up"), str(x.get("date_ist"))))
    return out[:14]


# ---- rule-based reader: the last safety net (no LLM) -- picks headlines that name a FUTURE date together with an event word
_KW = ((r"results|earnings|q[1-4]\s?fy", "Results"), (r"board meeting|to consider|fund ?rais|qip|buyback", "Corporate"), (r"record date|ex-?dividend|ex-?date|dividend", "Corporate"),
       (r"\bagm\b|\begm\b", "Corporate"), (r"\brbi\b|\bmpc\b|policy|repo", "Policy"), (r"\bfed\b|fomc|powell", "Global"), (r"\bcpi\b|\bwpi\b|\bgdp\b|\biip\b", "Data"))
_DATE_RE = (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I),
            re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I))


def heuristic_events(heads, now_ist=None):
    now_ist = now_ist or datetime.now(IST)
    today = now_ist.date()
    out, seen = [], set()
    for h in heads or []:
        t = h["title"]
        kind = next((k for pat, k in _KW if re.search(pat, t, re.I)), None)
        if not kind:
            continue
        d = None
        for rx in _DATE_RE:
            m = rx.search(t)
            if not m:
                continue
            a, b = m.group(1), m.group(2)
            day, mon = (int(a), _MON[b[:3].lower()]) if a.isdigit() else (int(b), _MON[a[:3].lower()])
            try:
                d = datetime(today.year, mon, day).date()
                if d < today - timedelta(days=30):
                    d = datetime(today.year + 1, mon, day).date()
            except Exception:
                d = None
            break
        if not d or d < today or d > today + timedelta(days=21) or t[:40] in seen:
            continue
        seen.add(t[:40])
        out.append({"event": t[:140] + " (headline se, AI nahi)", "status": "upcoming", "date_ist": d.isoformat(), "time_ist": "unknown", "type": kind, "affects": "headline ke hisaab se",
                    "expected_effect": "UNCLEAR", "impact": 3, "impact_start": "event ke din / agle open par", "duration": "", "why": "Headline me aane wali date likhi hai",
                    "evidence_text": f"[{h['n']}] {t[:90]} ({h['src']}, {h['date'][:10]})"})
    return out[:6]


def _llm_rows(prompt, heads, nvidia_keys):
    """-> (rows, err, model).  NVIDIA only."""
    nk = [k for k in (nvidia_keys or []) if k]
    if not nk:
        return [], "NVIDIA_API_KEY secrets me nahi mili -- Gemini fallback chala", ""
    try:
        import nvidia_llm
        msgs = [{"role": "system", "content": "You are a careful Indian equity research analyst. Reply with ONLY a JSON array. Never invent facts or dates."},
                {"role": "user", "content": prompt}]
        txt, model = nvidia_llm.chat_robust(nk, msgs, max_tokens=2000, temperature=0.1, deadline_s=70.0, thinking=False)
        return _clean_stock_events(_extract_json(txt), heads), "", model
    except Exception as exc:
        return [], f"NVIDIA: {type(exc).__name__}: {exc}"[:200], ""


def _finish(rows, err, heads, gem_subject, keys, headlines, now_ist):
    """NVIDIA failed / no key -> Gemini+Google, then the rule-based headline reader (so the table is not empty just because an API is down)."""
    gerr = ""
    if keys:
        g_rows, gerr = fetch_events(keys, gem_subject, [h["title"] for h in heads[:10]] or headlines, now_ist)
        if g_rows:
            return _clean_stock_events(g_rows, heads), ""
    hr = heuristic_events(heads, now_ist)
    if hr:
        return hr, ""
    return [], " | ".join(x for x in (err, gerr) if x)[:260]


def fetch_stock_events(keys, subject, headlines=None, now_ist=None, stock=None, nvidia_keys=None, heads=None):
    """Blocking (background thread).  -> (rows, error_text).  NVIDIA first; Gemini-with-search, then rule-based reader only as fallbacks."""
    now_ist = now_ist or datetime.now(IST)
    stock = stock or {}
    sym, comp, sec = stock.get("symbol") or subject, stock.get("company") or "", stock.get("sector") or ""
    heads = heads if heads is not None else gather_stock_headlines(sym, comp, sec)
    rows, err, model = _llm_rows(_stock_prompt(sym, comp, sec, heads, headlines, now_ist), heads, nvidia_keys)
    for r in rows:
        r["_model"] = model
    if rows:
        return rows, ""
    if not err:                                                    # the model answered with an empty list: nothing important -> say so, do not spend Gemini quota
        return [], ""
    out, e2 = _finish(rows, err, heads, f"{sym} ({comp or sym}, sector {sec or 'unknown'}) -- sirf isi stock aur uske sector ke events", keys, headlines, now_ist)
    return out, e2


def fetch_nifty_events(keys, headlines=None, now_ist=None, nvidia_keys=None, heads=None):
    now_ist = now_ist or datetime.now(IST)
    heads = heads if heads is not None else gather_nifty_headlines()
    rows, err, model = _llm_rows(_nifty_prompt(heads, headlines, now_ist), heads, nvidia_keys)
    for r in rows:
        r["_model"] = model
    if rows:
        return rows, ""
    if not err:
        return [], ""
    return _finish(rows, err, heads, "NIFTY 50 / poora Indian market", keys, headlines, now_ist)


# ---- state machine (never touches st.*)
def _start(key, worker, force=False):
    with _EV_LOCK:
        cur = dict(_EVENTS.get(key) or {})
        if cur.get("status") == "running" and time.time() - cur.get("started", 0) < 150:
            return
        _EVENTS[key] = {**cur, "status": "running", "started": time.time(), "ts": cur.get("ts", 0.0), "rows": cur.get("rows") or [], "err": cur.get("err", "")}
        if len(_EVENTS) > 60:                                                   # keep memory small when many stocks were opened
            for k in sorted(_EVENTS, key=lambda x: _EVENTS[x].get("ts", 0))[:20]:
                if k != key and _EVENTS[k].get("status") != "running":
                    _EVENTS.pop(k, None)

    def _w():
        t0 = time.time()
        try:
            heads = worker["gather"]()
            fp = _fingerprint(heads)
            same = cur.get("fp") == fp and cur.get("rows_ok") and (time.time() - cur.get("llm_ts", 0) < LLM_MAX_AGE_S) and not force
            if same:
                rows, err, llm_ts, ok = cur.get("rows") or [], "", cur.get("llm_ts", 0), True
            else:
                rows, err = worker["fetch"](heads)
                llm_ts, ok = time.time(), bool(rows) or not err
                if not rows and cur.get("rows") and err:                         # keep the last good rows if this run failed
                    rows = cur.get("rows")
        except Exception as exc:
            rows, err, fp, llm_ts, ok = cur.get("rows") or [], f"{type(exc).__name__}: {exc}"[:200], cur.get("fp"), cur.get("llm_ts", 0), False
        with _EV_LOCK:
            _EVENTS[key] = {"ts": time.time(), "status": "error" if (err and not rows) else "done", "rows": rows, "err": err, "fp": fp, "llm_ts": llm_ts, "rows_ok": ok, "took": round(time.time() - t0, 1)}

    threading.Thread(target=_w, daemon=True).start()


def start_stock_events(key, keys, subject, headlines=None, stock=None, nvidia_keys=None, force=False):
    stock = stock or {}
    _start(key, {"gather": lambda: gather_stock_headlines(stock.get("symbol") or subject, stock.get("company") or "", stock.get("sector") or ""),
                 "fetch": lambda heads: fetch_stock_events(keys, subject, headlines, stock=stock, nvidia_keys=nvidia_keys, heads=heads)}, force=force)


def start_nifty_events(key, keys, headlines=None, nvidia_keys=None, force=False):
    _start(key, {"gather": gather_nifty_headlines, "fetch": lambda heads: fetch_nifty_events(keys, headlines, nvidia_keys=nvidia_keys, heads=heads)}, force=force)


_BG_NEWS = {"rows": None, "running": False, "ts": 0.0}


def start_bg_news(fn):
    """Fetch headlines in a background thread (never blocks the page)."""
    with _EV_LOCK:
        if _BG_NEWS["running"] and time.time() - _BG_NEWS["ts"] < 90:
            return
        _BG_NEWS["running"], _BG_NEWS["ts"] = True, time.time()

    def _w():
        try:
            rows = fn()
        except Exception:
            rows = None
        with _EV_LOCK:
            _BG_NEWS["rows"], _BG_NEWS["running"] = (rows or None), False

    threading.Thread(target=_w, daemon=True).start()


def take_bg_news():
    with _EV_LOCK:
        rows, _BG_NEWS["rows"] = _BG_NEWS["rows"], None
    return rows


def events_state(key):
    with _EV_LOCK:
        return dict(_EVENTS.get(key) or {})


# ---------------------------------------------------------------------------------------------------------- relevance filter (runs on every render, pure python, ~0 ms)
def _parse_dt(date_s, time_s):
    try:
        d = datetime.strptime(str(date_s)[:10], "%Y-%m-%d")
    except Exception:
        return None, False
    try:
        hh, mm = [int(x) for x in str(time_s).split(":")[:2]]
        return d.replace(hour=hh, minute=mm, tzinfo=IST), True
    except Exception:
        return d.replace(hour=15, minute=30, tzinfo=IST), False


def _in_session(dt):
    m = dt.hour * 60 + dt.minute
    return dt.weekday() < 5 and 9 * 60 + 15 <= m <= 15 * 60 + 30


def _next_open_after(dt):
    d = dt.date()
    if dt.weekday() < 5 and dt.hour * 60 + dt.minute < 9 * 60 + 15:
        pass
    else:
        d = _next_trading_day(d)
    return datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)


def event_phase(row, now=None):
    """-> 'upcoming' | 'live' | 'undated' | None (drop).  'live' = already happened but the market has not finished reacting (e.g. news after 15:30, reaction at next 09:15)."""
    now = now or datetime.now(IST)
    dt, _known_time = _parse_dt(row.get("date_ist"), row.get("time_ist"))
    if dt is None:
        return "undated" if str(row.get("status", "")).lower().startswith("up") else None
    if dt >= now - timedelta(minutes=5):
        return "upcoming" if dt <= now + timedelta(days=30) else None
    end = (dt + timedelta(minutes=60)) if _in_session(dt) else (_next_open_after(dt) + timedelta(minutes=60))
    return "live" if now <= end else None


def _eta(dt, now):
    secs = (dt - now).total_seconds()
    if secs < 0:
        return "abhi"
    if secs < 3600:
        return f"{max(1, int(secs // 60))} min baad"
    if secs < 86400:
        h, m = int(secs // 3600), int(secs % 3600 // 60)
        return f"{h} ghante {m} min baad"
    return f"{int(secs // 86400)} din baad"


def best_rows(rows, now=None, limit=8):
    """Only what still matters: upcoming + still-reacting, impact >= 3, live first, then by time.  -> [(phase, row)]"""
    now = now or datetime.now(IST)
    out = []
    for r in rows or []:
        ph = event_phase(r, now)
        if not ph:
            continue
        try:
            imp = int(float(r.get("impact", 3)))
        except Exception:
            imp = 3
        if imp < 3 or (ph == "undated" and imp < 4):
            continue
        dt, _ = _parse_dt(r.get("date_ist"), r.get("time_ist"))
        out.append((0 if ph == "live" else 1 if ph == "upcoming" else 2, -imp, dt or (now + timedelta(days=99)), ph, r))
    out.sort(key=lambda x: (x[0], x[2], x[1]))                  # live first, then soonest first (impact is shown in its own column)
    return [(x[3], x[4]) for x in out[:limit]]



# ------------------------------------------------------------------------------------------------------------ v68: headline -> market impact (always shown)
# Pure rules, no network, no API: the table below the news box is NEVER empty just because an LLM key / quota is missing.
# (category regex, label, impact 3-5, "kab asar", "kitni der")
_IMPACT_RULES = (
    (r"\b(rbi|repo rate|monetary policy|mpc|budget)\b", "RBI / Budget", 5, "Announcement ke turant baad 5-15 min me spike; phir 1-2 ghante me direction settle", "1-2 ghante (policy din poora din volatile)"),
    (r"\b(fed|fomc|powell|us cpi|nonfarm|jobs report|treasury yield|us inflation)\b", "US Fed / US data", 5, "Raat ko aata hai -> asar GIFT Nifty se agle din ke 09:15 gap me", "agle din ka pehla 1 ghanta"),
    (r"\b(crude|brent|opec|oil price)\b", "Crude oil", 4, "Market chalu me turant (OMC, paint, aviation, tyre sector pe zyada)", "kuch ghante - poora din, agar crude 2%+ hila"),
    (r"\b(fii|fpi|dii|foreign investor|institutional)\b", "FII / DII flow", 4, "Aaj ka flow aaj ki direction banata hai; data shaam ko aata hai -> asar agle din", "1-3 din ka trend"),
    (r"\b(war|missile|attack|ceasefire|sanction|tariff|geopolit|border|iran|israel|russia|china)\b", "Geopolitics / Tariff", 4, "Global market ke saath turant; Asia open par gap", "kuch ghante se kuch din (khabar badalne tak)"),
    (r"\b(q[1-4]\s?fy|results?|earnings|profit|net loss|revenue|ebitda|guidance)\b", "Company results", 4, "Results aksar market ke baad aate hain -> agle din 09:15 gap; pehle 15-30 min whipsaw", "pehle 30 min whipsaw, phir 1-2 din trend"),
    (r"\b(sebi|court|supreme court|nclt|tribunal|probe|raid|fraud|penalty|ban)\b", "SEBI / Court / Regulatory", 4, "Aate hi tez move; aksar 1-3 candles (5-15 min) me reaction poora", "15-60 min (bada order ho to 1-2 din)"),
    (r"\b(order win|bags order|wins order|contract|deal|acquisition|merger|stake|buyback|dividend|bonus|split)\b", "Order / Deal / Corporate action", 3, "Khabar aate hi us stock me; sector peers me halka asar", "1 din (record date tak halka support)"),
    (r"\b(upgrade|downgrade|target price|brokerage|rating)\b", "Brokerage rating", 3, "Agle open / khabar aate hi us stock me", "1-2 din"),
    (r"\b(gdp|cpi|wpi|inflation|iip|pmi|trade deficit|gst collection)\b", "India macro data", 4, "Data ~17:30 IST aata hai (market band) -> asar agle din 09:15 gap me", "agle din ka pehla 1 ghanta"),
    (r"\b(expiry|f&o|derivative|margin)\b", "Expiry / F&O rules", 3, "Expiry din 14:00-15:30 me tez jhatke", "aakhri 1.5 ghanta"),
)


def headline_impacts(news_items, limit=6, now=None):
    """Top fresh headlines -> [{"Khabar", "Sentiment", "Impact", "Category", "Market me asar kab", "Kitni der", "Umar"}] (rule-based, <1 ms)."""
    out = []
    for n in (news_items or [])[:25]:
        title = (n.get("title") or "").strip()
        if not title:
            continue
        age = _age_hours(n.get("published"), now)
        if age is not None and age > 72:
            continue
        low = title.lower()
        cat = next(((lab, imp, kab, dur) for rx, lab, imp, kab, dur in _IMPACT_RULES if re.search(rx, low)), None)
        sc = _headline_score(title)
        if cat is None and sc == 0:
            continue                                                       # neither a known market driver nor a clear positive/negative -> noise
        lab, imp, kab, dur = cat or ("General market news", 3, "Khabar ke hisaab se din me; confirm karne ke liye price reaction dekho", "kuch ghante")
        if sc != 0 and abs(sc) >= 2:
            imp = min(5, imp + 1)
        old = age is not None and age > 24
        out.append((-(imp if not old else imp - 1), age if age is not None else 99.0, {
            "Khabar": title[:140], "Sentiment": "🟢 Positive" if sc > 0 else "🔴 Negative" if sc < 0 else "⚪ Neutral",
            "Impact": _IMP.get(imp, "🟡 MEDIUM") if "_IMP" in globals() else str(imp), "Category": lab,
            "Market me asar kab": kab if not old else "Khabar purani (>24h) -- asar zyadatar price me aa chuka hoga",
            "Kitni der": dur, "Umar": "time unknown" if age is None else (f"{age:.0f}h" if age < 48 else f"{age/24:.0f} din")}))
    out.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in out[:limit]]

# ---------------------------------------------------------------------------------------------------------------------------------- render
_BADGE = {"POSITIVE": "🟢 POSITIVE", "NEGATIVE": "🔴 NEGATIVE", "NEUTRAL": "⚪ NEUTRAL", "UNCLEAR": "⚪ UNCLEAR (data kam)"}
_EFFECT = {"UP": "🟢 UP", "DOWN": "🔴 DOWN", "VOLATILE": "🟠 VOLATILE (tez jhatke)", "SIDEWAYS": "⚪ SIDEWAYS", "UNCLEAR": "⚪ unclear"}
_IMP = {5: "🔥 BAHUT BADA", 4: "🔥 BADA", 3: "🟡 MEDIUM"}


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


def _row_to_table(phase, r, now):
    dt, known = _parse_dt(r.get("date_ist"), r.get("time_ist"))
    if dt is None:
        kab, eta = "date pata nahi", ""
    else:
        kab = f"{dt.strftime('%d-%b')} {dt.strftime('%H:%M') if known else '(time pata nahi)'}"
        eta = "🔴 ABHI ASAR CHAL RAHA" if phase == "live" else _eta(dt, now)
    if phase == "live":
        mk = r.get("impact_start") or "Event aa chuki hai, market ka reaction abhi poora nahi hua -- live chart dekho"
    elif dt is not None and known:
        mk = r.get("india_text") or india_impact(r.get("date_ist"), r.get("time_ist"), now)
    else:
        mk = r.get("india_text") or r.get("impact_start") or r.get("india_note") or "time confirm nahi -- us din dhyan rakho"
    try:
        imp = int(float(r.get("impact", 3)))
    except Exception:
        imp = 3
    return {"Kab (IST)": f"{kab}  ·  {eta}".strip(" ·"), "Kya": r.get("event"), "Impact": _IMP.get(imp, "🟡 MEDIUM"), "Direction": _EFFECT.get(str(r.get("expected_effect")).upper(), r.get("expected_effect")),
            "Kis par asar": r.get("affects") or "Nifty / market", "Market me asar kab / kahan": mk, "Kitni der": r.get("duration") or r.get("reaction_window") or "", "Kyun": r.get("why") or "",
            "Sabut (headline)": r.get("evidence_text") or ("rule-based calendar" if r.get("type") == "expiry" else "")}


def render_news(st, subject, news_items, key="x", stock=None):
    """Verdict + BEST upcoming events INSIDE the existing News expander (no new boxes).  Everything slow runs in a background thread."""
    try:
        nv = news_verdict(news_items)
        sc = [r[0] for r in nv["rows"]]
        st.markdown(f"**News verdict: {_BADGE[nv['verdict']]}**  ·  🟢 {sum('Positive' in x for x in sc)}  🔴 {sum('Negative' in x for x in sc)}  ⚪ {sum('Neutral' in x for x in sc)}")
        st.caption(nv["note"] or "Sirf 7 din ke andar ki news ginti me; headline ke neeche har ek ka sentiment: " +
                   " | ".join(f"{r[0].split()[0]} {r[2][:38]}…" for r in nv["rows"][:4]))
        heads = [r[2] for r in nv["rows"]]
        _ks = st.session_state.get("_news_intel_keys") or []
        _nk = st.session_state.get("_nvidia_keys") or []
        _es = events_state(key)
        _age = time.time() - _es.get("ts", 0) if _es else 1e9
        _failed = bool(_es) and _es.get("status") == "error"
        _stale = (not _es) or (_es.get("status") in ("done", "error") and _age > (RETRY_FAIL_S if _failed else RECHECK_S))
        _label = (stock.get("symbol") or subject) if stock else "NIFTY 50"
        _go = lambda force: (start_stock_events(key, _ks, subject, heads, stock=stock, nvidia_keys=_nk, force=force) if stock
                             else start_nifty_events(key, _ks, heads, nvidia_keys=_nk, force=force))
        if _stale:
            _go(False)                                                       # automatic, background, LLM only if the headlines changed
        if st.button(f"🔄 {_label} ke best events abhi dobara nikalo", key=f"ni_btn_{key}"):
            _go(True)
        stt = events_state(key)
        now = datetime.now(IST)
        shown = best_rows(stt.get("rows") or [], now, limit=8 if stock else 10)
        pre = [("upcoming", r) for r in fixed_calendar(now)] if not stock else []
        pre = [(ph, r) for ph, r in pre if event_phase(r, now) == "upcoming" and int(r.get("impact", 3)) >= 3]
        table = [_row_to_table(ph, r, now) for ph, r in (pre + shown)]
        table.sort(key=lambda x: (0 if "ABHI" in x["Kab (IST)"] else 1))
        try:                                                                # v68: always-on, rule-based "badi khabar -> market asar" table
            _hi = headline_impacts(news_items, limit=6 if stock else 8)
            if _hi:
                st.markdown("**📰 Badi khabar → market par asar (headline se, turant)**")
                st.dataframe(_hi, hide_index=True, width="stretch")
        except Exception:
            pass
        if stt.get("status") == "running" and not table:
            st.info("⏳ Fresh news padhi ja rahi hai (5-30 sec). Page ko mat chhedo -- neeche apne aap dikh jayega.")
        if table:
            st.dataframe(table, hide_index=True, width="stretch")
            if stt.get("status") == "running":
                st.caption("🔄 Naya check background me chal raha hai.")
        elif stt.get("status") != "running":
            st.caption("Abhi koi bada aane wala event nahi mila -- jo ho chuka aur jiska asar nikal chuka wo jaan-boojh kar nahi dikhaya." + (f"  ({stt['err']})" if stt.get("err") else ""))
        if stt.get("err") and table:
            st.caption(f"⚠️ {stt['err']}")
        st.caption("Sirf aane wale ya abhi asar chalu events. Date/time headline se hain, AI/rule-based andaza hai -- khud verify karo, guarantee nahi. "
                   "Indian market: 09:15 open - 15:30 close (IST). Exchange holiday offline pata nahi chalta.")
    except Exception:
        pass
