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
EVENTS_TTL_S = 7200
AUTO_START_GRACE_S = 240    # no AUTOMATIC research in the first 4 minutes after a server start (the page's cold start goes first); buttons work at once
_BOOT_T = time.time()
STOCK_EVENTS_TTL_S = 2400   # stock events are re-researched every 40 min (background thread, the page never waits)

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
    out = [{"event": "Nifty weekly expiry", "date_ist": exp.isoformat(), "time_ist": "15:30", "type": "expiry", "expected_effect": "VOLATILE", "india_text": f"{_label(exp, d)} 14:00 se 15:30 tak (market chalu, aakhri ghante me tez jhatke)",
            "why": "Expiry din option gamma badhta hai; 14:00-15:30 me tez, jhatke wali moves aati hain.", "reaction_window": "14:00-15:30 IST"},
           {"event": "US market open (India me tab band hota hai)", "date_ist": d.isoformat(), "time_ist": _us_open_ist(now_ist),
            "type": "global", "expected_effect": "VOLATILE", "why": "US ka mood raat ko GIFT Nifty me dikhta hai.", "reaction_window": ""}]
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
        if time.time() < atd._SEARCH_COOL["until"]:
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
    models = sorted(list(atd.REVIEW_MODELS), key=lambda m: 0 if "lite" in m else 1)

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


# ---------------------------------------------------------------------------------------------------------------------- v65: PER-STOCK events
# Each stock gets ITS OWN research: several targeted Google-News RSS searches (company results / orders / regulator / promoter / sector policy),
# then NVIDIA NIM (NOT Gemini -- Gemini is already used heavily by the trade review) turns those REAL headlines into a dated happened/upcoming
# table with "when will it hit the price".  Every row must cite one of the supplied headlines; rows that cannot are flagged "AI andaza".
# If NVIDIA gives no answer, the old Gemini+Google-Search path is the fallback (with a stock-specific prompt).

_STOCK_STOP = re.compile(r"\b(ltd|limited|co|corp|corporation|company|ven|ventures?|industries|inds?)\b\.?", re.I)


def stock_queries(symbol, company="", sector=""):
    """-> [(label, google-news query)] -- all about THIS stock (+ its sector)."""
    sym = re.sub(r"[^A-Za-z0-9&\-]", "", str(symbol or ""))
    comp = re.sub(r"\s+", " ", str(company or "").strip())
    short = re.sub(r"\s+", " ", _STOCK_STOP.sub(" ", comp)).strip() or sym
    who = f'"{comp}"' if comp else sym
    sec = re.sub(r"\s+", " ", str(sector or "").strip())
    q = [("Company", f'{who} OR "{short}" share when:7d'),
         ("Results/Board", f'"{short}" results OR earnings OR "board meeting" OR dividend OR "record date" when:30d'),
         ("Orders/Deals", f'"{short}" order OR contract OR acquisition OR "block deal" OR stake when:14d'),
         ("Regulator/Court", f'"{short}" SEBI OR NSE OR court OR NCLT OR promoter OR pledge OR rating when:14d')]
    if sec and sec.lower() not in ("unknown", "n/a", "none", "nan"):
        q.append(("Sector", f"{sec} sector India stocks policy OR demand OR prices when:7d"))
    return q


def gather_stock_headlines(symbol, company="", sector="", per_query=6, max_age_days=14):
    """Real, dated headlines for ONE stock (parallel, short timeouts, never raises).  -> [{'n','label','title','src','age_h','date'}]"""
    out, seen = [], set()
    try:
        import news_rss
    except Exception:
        return out
    from concurrent.futures import ThreadPoolExecutor
    qs = stock_queries(symbol, company, sector)

    def _one(lq):
        try:
            return lq[0], news_rss.parse_items(news_rss._fetch(lq[1], timeout=7), max_age_h=max_age_days * 24, limit=per_query)
        except Exception:
            return lq[0], []
    try:
        with ThreadPoolExecutor(max_workers=5) as ex:
            res = list(ex.map(_one, qs))
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
    out = out[:24]
    for i, r in enumerate(out, 1):
        r["n"] = i
    return out



# ---- MARKET-WIDE (India + global) headlines: fetched ONCE per 20 minutes and shared by every stock (no per-stock cost) ---------------------------
_MACRO = {"t": 0.0, "rows": []}
_MACRO_LOCK = threading.Lock()
MACRO_TTL_S = 1200
_MACRO_QUERIES = [("Macro-India", 'Sensex OR Nifty market outlook RBI OR FII OR inflation OR "rate cut" when:2d'),
                  ("Macro-Global", 'Wall Street OR "US Fed" OR "Asian markets" OR "crude oil" OR tariff OR dollar when:1d')]


def gather_macro_headlines(per_query=6):
    """India + world headlines that move the whole market (and therefore the stock).  Cached 20 min, shared.  Never raises."""
    with _MACRO_LOCK:
        if time.time() - _MACRO["t"] < MACRO_TTL_S and _MACRO["rows"]:
            return list(_MACRO["rows"])
        try:
            import news_rss
            from concurrent.futures import ThreadPoolExecutor

            def _one(lq):
                try:
                    return lq[0], news_rss.parse_items(news_rss._fetch(lq[1], timeout=7), max_age_h=48, limit=per_query)
                except Exception:
                    return lq[0], []
            with ThreadPoolExecutor(max_workers=2) as ex:
                res = list(ex.map(_one, _MACRO_QUERIES))
            now = datetime.now(IST)
            out = []
            for label, items in res:
                for age_min, title, src in items:
                    out.append({"label": label, "title": title, "src": src or "news", "age_h": round(age_min / 60.0, 1),
                                "date": (now - timedelta(minutes=age_min)).strftime("%Y-%m-%d %H:%M")})
            _MACRO["t"], _MACRO["rows"] = time.time(), out
        except Exception:
            pass
        return list(_MACRO["rows"])


_CAL_CACHE = {}              # symbol -> (time, rows, ok)
_CAL_LOCK = threading.Lock()


def stock_calendar_rows(symbol, now_ist=None):
    """Cached wrapper: Yahoo is asked at most once per 12 h per stock (1 h after a failure) and only one call at a time, so this can never
    hold the shared yfinance session that the page's own market-data steps also use."""
    sym = str(symbol)
    now = time.time()
    with _CAL_LOCK:
        c = _CAL_CACHE.get(sym)
        if c and now - c[0] < (43200 if c[2] else 3600):
            return list(c[1])
        if not _auto_ok():
            return []
        rows = _stock_calendar_rows_uncached(symbol, now_ist)
        _CAL_CACHE[sym] = (time.time(), rows, bool(rows))
        return list(rows)


def _stock_calendar_rows_uncached(symbol, now_ist=None):
    """REAL scheduled events from Yahoo's calendar (results date, ex-dividend date) -> event rows.  8 s limit, never raises, [] if unavailable."""
    now_ist = now_ist or datetime.now(IST)
    box = {}

    def _w():
        try:
            import yfinance as yf
            box["cal"] = yf.Ticker(re.sub(r"[^A-Za-z0-9&\-]", "", str(symbol)) + ".NS").calendar
        except Exception:
            box["cal"] = None
    t = threading.Thread(target=_w, daemon=True)
    t.start()
    t.join(5.0)
    cal = box.get("cal")
    rows = []
    try:
        if not isinstance(cal, dict):
            return rows
        today = now_ist.date()

        def _dates(v):
            v = v if isinstance(v, (list, tuple)) else [v]
            o = []
            for x in v:
                try:
                    d = x.date() if hasattr(x, "date") else datetime.strptime(str(x)[:10], "%Y-%m-%d").date()
                    if d >= today:
                        o.append(d)
                except Exception:
                    pass
            return sorted(o)
        for d in _dates(cal.get("Earnings Date"))[:1]:
            rows.append({"event": "Quarterly results (company ki scheduled date)", "status": "upcoming", "date_ist": d.isoformat(), "time_ist": "unknown", "type": "Results",
                         "affects": "is stock", "expected_effect": "VOLATILE", "impact_start": "results aate hi (market ke baad aaye to agle din 09:15 open par gap)",
                         "duration": "1-2 din", "why": "Results par stock aam taur par tez hilta hai", "importance": 5, "evidence_text": "Yahoo Finance calendar (asli scheduled date)"})
        for d in _dates(cal.get("Ex-Dividend Date"))[:1]:
            rows.append({"event": "Ex-dividend date", "status": "upcoming", "date_ist": d.isoformat(), "time_ist": "09:15", "type": "Corporate",
                         "affects": "is stock", "expected_effect": "DOWN", "impact_start": "us din open par (price dividend jitna adjust hota hai)",
                         "duration": "us din", "why": "Ex-date par bhav dividend ke barabar neeche khulta hai", "importance": 2, "evidence_text": "Yahoo Finance calendar (asli scheduled date)"})
    except Exception:
        return []
    return rows


def _stock_prompt(symbol, company, sector, heads, extra_titles, now_ist):
    lines = "\n".join(f"[{h['n']}] ({h['date']} IST, {h['label']}, {h['src']}) {h['title']}" for h in heads) or "(koi headline nahi mili)"
    old = "\n".join(f"- {t}" for t in (extra_titles or [])[:8])
    return (f"Aaj {now_ist.strftime('%Y-%m-%d %H:%M')} IST ({now_ist.strftime('%A')}) hai. Tum Indian equity research analyst ho. Stock: {symbol} ({company or symbol}), sector: {sector or 'unknown'}.\n"
            "Neeche REAL headlines hain (number, time, source ke saath): is stock ki apni news + sector + poore India/global market ki news (Macro-India / Macro-Global).\n"
            "Kaam: inme se SIRF sabse zyada KAAM ki 4 se 8 cheezein chuno jo is stock ka bhav hila sakti hain:\n"
            "  (1) abhi/pichhle ~48 ghante me HO CHUKI badi news (company, sector, ya India/global macro jo is stock par asar daalti hai),\n"
            "  (2) AANE WALI event (results/board meeting/dividend/order/AGM/regulatory/court/sector policy/RBI/Fed/budget jo is stock par asar dale).\n"
            "Har row me batao ki asar INDIA MARKET me KAB padega: market 09:15-15:30 IST khula rehta hai. Agar news market ke andar aayi to asar 'turant' (aaj kitne baje tak); "
            "market band hone ke baad / weekend par aayi to 'agle trading din 09:15 open par gap'. Aane wali event ke liye date + kis samay asar (e.g. 'aaj 14:00 baje', 'kal open par', 'results ke baad agle din').\n"
            "Rules: SIRF headlines me jo likha hai wahi (ya uska seedha asar). 'evidence' = headline ka number. Date/time headline me na ho to 'unknown' -- date invent mat karo. "
            "Kam kaam ki / duplicate / generic cheezein (price-chart pages, 'stocks to watch' list) MAT do. Jo sabse zyada asar wali ho wo importance 5, sabse kam 1. Khaali list bhi theek hai.\n\n"
            f"HEADLINES:\n{lines}\n\nDashboard ki purani headlines (context):\n{old}\n\n"
            "Sirf JSON array do (max 8), koi aur text nahi. Format:\n"
            '[{"event":"..","status":"happened|upcoming","date_ist":"YYYY-MM-DD ya unknown","time_ist":"HH:MM ya unknown","type":"Results|Order|Deal|Regulatory|Corporate|Sector|Rating|Macro-India|Macro-Global|News",'
            '"affects":"is stock / sector / poora market","expected_effect":"UP|DOWN|VOLATILE|SIDEWAYS|UNCLEAR","impact_time_ist":"YYYY-MM-DD HH:MM ya unknown (asar shuru hone ka samay)",'
            '"impact_start":"asar kab se (e.g. turant aaj 14:00 tak, kal 09:15 open par gap, results ke agle din)","duration":"asar kitni der/din","why":"1 line, is stock ke liye kyun",'
            '"importance":<1-5>,"evidence":<headline number ya null>}]')


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
        out.append(r)
    for r in out:
        try:
            r["importance"] = max(1, min(5, int(float(r.get("importance") or 3))))
        except Exception:
            r["importance"] = 3
    # upcoming first (nearest date first, unknown dates last), then what already happened (most important, newest first)
    def _k(x):
        up = str(x.get("status", "")).lower().startswith("up")
        d = str(x.get("date_ist"))
        return (0 if up else 1, (d if (up and d != "unknown") else "9999") if up else "", -x["importance"], "" if up else "".join(chr(255 - ord(c)) for c in d))
    out.sort(key=_k)
    return out[:10]


def _merge_calendar(rows, cal_rows):
    """Real calendar rows first (they are facts); drop a model row that repeats the same event/date."""
    out = list(cal_rows or [])
    have = {(str(r.get("type")), str(r.get("date_ist"))) for r in out}
    for r in rows or []:
        if (str(r.get("type")), str(r.get("date_ist"))) in have and str(r.get("type")) in ("Results", "Corporate"):
            continue
        out.append(r)
    return out[:10]


def fetch_stock_events(keys, subject, headlines=None, now_ist=None, stock=None, nvidia_keys=None):
    """Blocking (run in a background thread).  -> (rows, error_text).  NVIDIA first, Gemini-with-search only as the fallback."""
    now_ist = now_ist or datetime.now(IST)
    stock = stock or {}
    sym, comp, sec = stock.get("symbol") or subject, stock.get("company") or "", stock.get("sector") or ""
    heads = gather_stock_headlines(sym, comp, sec)
    try:                                           # + market-wide India/global headlines (shared 20-min cache) -> numbered after the stock's own
        for h in gather_macro_headlines()[:10]:
            heads.append(dict(h, n=len(heads) + 1))
    except Exception:
        pass
    try:
        cal_rows = stock_calendar_rows(sym, now_ist)         # real scheduled results / ex-dividend date (8 s limit)
    except Exception:
        cal_rows = []
    err = ""
    nk = [k for k in (nvidia_keys or []) if k]
    if nk:
        try:
            import nvidia_llm
            msgs = [{"role": "system", "content": "You are a careful Indian equity research analyst. Reply with ONLY a JSON array. Never invent facts or dates."},
                    {"role": "user", "content": _stock_prompt(sym, comp, sec, heads, headlines, now_ist)}]
            txt, model = nvidia_llm.chat_robust(nk, msgs, max_tokens=2400, temperature=0.1, deadline_s=80.0, thinking=False)
            rows = _merge_calendar(_clean_stock_events(_extract_json(txt), heads), cal_rows)
            if rows or heads:
                for r in rows:
                    r["_model"] = model
                return rows, ("" if rows else "Is stock ki last 14 din ki headlines me koi badi/aane wali event nahi mili.")
            err = "NVIDIA ne khaali jawab diya"
        except Exception as exc:
            err = f"NVIDIA: {type(exc).__name__}: {exc}"[:200]
    else:
        err = "NVIDIA_API_KEY secrets me nahi mili -- Gemini fallback chala"
    # fallback: the old Google-grounded Gemini call, but with THIS stock as the subject (not 'Nifty')
    _g_rows, gerr = fetch_events(keys, f"{sym} ({comp or sym}, sector {sec or 'unknown'}) -- sirf isi stock aur uske sector ke events", [h["title"] for h in heads[:10]] or headlines, now_ist)
    rows = _merge_calendar(_g_rows or [], cal_rows)
    return rows, (err if rows else (err + " | " + gerr)).strip(" |")[:260]


_NIFTY_QUERIES = [("Nifty", 'Nifty 50 OR Sensex market today when:1d'),
                  ("Heavyweights", 'Reliance OR "HDFC Bank" OR Infosys OR "ICICI Bank" OR TCS OR "Bharti Airtel" results OR order OR rating when:2d'),
                  ("Upcoming", 'RBI policy OR "Fed meeting" OR "GDP data" OR "CPI inflation" OR "F&O expiry" OR "Union Budget" OR "FII" India market when:7d')]


def gather_nifty_headlines(per_query=7, max_age_days=7):
    """Real, dated headlines that move the Nifty 50 (index news + heavyweights + scheduled macro) + the shared India/global macro list."""
    out, seen = [], set()
    try:
        import news_rss
        from concurrent.futures import ThreadPoolExecutor

        def _one(lq):
            try:
                return lq[0], news_rss.parse_items(news_rss._fetch(lq[1], timeout=7), max_age_h=max_age_days * 24, limit=per_query)
            except Exception:
                return lq[0], []
        with ThreadPoolExecutor(max_workers=3) as ex:
            res = list(ex.map(_one, _NIFTY_QUERIES))
        now = datetime.now(IST)
        rows = [(lab, a, t, sr) for lab, items in res for a, t, sr in items]
        for lab, age_min, title, src in rows:
            k = re.sub(r"\W+", "", title.lower())[:60]
            if k in seen:
                continue
            seen.add(k)
            out.append({"label": lab, "title": title, "src": src or "news", "age_h": round(age_min / 60.0, 1), "date": (now - timedelta(minutes=age_min)).strftime("%Y-%m-%d %H:%M")})
    except Exception:
        pass
    for h in gather_macro_headlines():
        k = re.sub(r"\W+", "", h["title"].lower())[:60]
        if k not in seen:
            seen.add(k)
            out.append(dict(h))
    out.sort(key=lambda r: r["age_h"])
    out = out[:26]
    for i, r in enumerate(out, 1):
        r["n"] = i
    return out


def fetch_nifty_events(keys, subject, headlines=None, now_ist=None, nvidia_keys=None):
    """Blocking (background thread).  NVIDIA on real headlines first (same ranked, timed table as the stocks), old Gemini+Google path as fallback."""
    now_ist = now_ist or datetime.now(IST)
    heads = gather_nifty_headlines()
    err = ""
    nk = [k for k in (nvidia_keys or []) if k]
    if nk and heads:
        try:
            import nvidia_llm
            msgs = [{"role": "system", "content": "You are a careful Indian equity market analyst. Reply with ONLY a JSON array. Never invent facts or dates."},
                    {"role": "user", "content": _stock_prompt("NIFTY 50", "Nifty 50 index (poora India market)", "Index", heads, headlines, now_ist)}]
            txt, model = nvidia_llm.chat_robust(nk, msgs, max_tokens=2400, temperature=0.1, deadline_s=80.0, thinking=False)
            rows = _clean_stock_events(_extract_json(txt), heads)
            if rows:
                for r in rows:
                    r["_model"] = model
                return rows, ""
            err = "NVIDIA ne koi badi event nahi chuni"
        except Exception as exc:
            err = f"NVIDIA: {type(exc).__name__}: {exc}"[:200]
    rows, gerr = fetch_events(keys, subject, headlines, now_ist)
    return rows, ("" if rows else (err + " | " + gerr).strip(" |")[:260])


def start_nifty_events(key, keys, subject, headlines=None, nvidia_keys=None):
    if not _slot_acquire():      # another research thread is running -> try again on a later refresh, cost nothing now
        return
    with _EV_LOCK:
        cur = _EVENTS.get(key)
        if cur and cur["status"] == "running" and time.time() - cur["ts"] < 120:
            _EV_SLOT.release()
            return
        _EVENTS[key] = {"ts": time.time(), "status": "running", "rows": [], "err": ""}

    def _work():
        try:
            rows, err = fetch_nifty_events(keys, subject, headlines, nvidia_keys=nvidia_keys)
        except Exception as exc:
            rows, err = [], f"{type(exc).__name__}: {exc}"[:200]
        _EV_SLOT.release()
        with _EV_LOCK:
            _EVENTS[key] = {"ts": time.time(), "status": "error" if (err and not rows) else "done", "rows": rows, "err": err}

    threading.Thread(target=_work, daemon=True).start()


def start_stock_events(key, keys, subject, headlines=None, stock=None, nvidia_keys=None):
    """Background thread, same state machine as start_events()."""
    if not _slot_acquire():      # another research thread is running -> try again on a later refresh, cost nothing now
        return
    with _EV_LOCK:
        cur = _EVENTS.get(key)
        if cur and cur["status"] == "running" and time.time() - cur["ts"] < 120:
            _EV_SLOT.release()
            return
        _EVENTS[key] = {"ts": time.time(), "status": "running", "rows": [], "err": ""}

    def _work():
        try:
            rows, err = fetch_stock_events(keys, subject, headlines, stock=stock, nvidia_keys=nvidia_keys)
        except Exception as exc:
            rows, err = [], f"{type(exc).__name__}: {exc}"[:200]
        _EV_SLOT.release()
        with _EV_LOCK:
            _EVENTS[key] = {"ts": time.time(), "status": "error" if (err and not rows and "nahi mili" not in err) else "done", "rows": rows, "err": err}

    threading.Thread(target=_work, daemon=True).start()


_EV_SLOT = threading.Semaphore(1)     # ONE events-research thread in the whole app at a time (Nifty or a stock): nothing can pile up behind the page


_EV_SLOT_T = {"t": 0.0}


def _slot_acquire():
    """Non-blocking.  A research thread that somehow hangs for more than 4 minutes is abandoned (its slot is handed on), never blocks the feature for ever."""
    global _EV_SLOT
    if _EV_SLOT.acquire(blocking=False):
        _EV_SLOT_T["t"] = time.time()
        return True
    if time.time() - _EV_SLOT_T["t"] > 240:
        _EV_SLOT = threading.Semaphore(1)
        _EV_SLOT.acquire()
        _EV_SLOT_T["t"] = time.time()
        return True
    return False


def _auto_ok():
    return time.time() - _BOOT_T > AUTO_START_GRACE_S


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


def render_news(st, subject, news_items, key="x", stock=None):
    """Verdict + upcoming events INSIDE the existing News expander (no new boxes)."""
    try:
        nv = news_verdict(news_items)
        sc = [r[0] for r in nv["rows"]]
        st.markdown(f"**News verdict: {_BADGE[nv['verdict']]}**  ·  🟢 {sum('Positive' in x for x in sc)}  🔴 {sum('Negative' in x for x in sc)}  ⚪ {sum('Neutral' in x for x in sc)}")
        st.caption(nv["note"] or "Sirf 7 din ke andar ki news ginti me; headline ke neeche har ek ka sentiment: " +
                   " | ".join(f"{r[0].split()[0]} {r[2][:38]}…" for r in nv["rows"][:4]))
        heads = [r[2] for r in nv["rows"]]
        _ks = st.session_state.get("_news_intel_keys") or []
        _es = events_state(key)
        _nk = st.session_state.get("_nvidia_keys") or []
        _ttl = STOCK_EVENTS_TTL_S if (stock or key == "nifty50") else EVENTS_TTL_S
        rich = bool(stock) or key == "nifty50"
        _stale = (not _es) or (_es.get("status") in ("done", "error") and time.time() - _es.get("ts", 0) > _ttl)
        if stock:
            if _stale and _auto_ok():
                start_stock_events(key, _ks, subject, heads, stock=stock, nvidia_keys=_nk)       # this stock's OWN research, automatic, once per 2 h, in the background
            if st.button(f"🔄 {stock.get('symbol') or subject} ke events dobara nikalo (NVIDIA research)", key=f"ni_btn_{key}"):
                start_stock_events(key, _ks, subject, heads, stock=stock, nvidia_keys=_nk)
        else:
            if key == "nifty50" and (_ks or _nk) and _stale and _auto_ok():
                start_nifty_events(key, _ks, subject, heads, nvidia_keys=_nk)      # automatic BACKGROUND lookup every 40 min (page never waits)
            if st.button("🗓️ Aaj/kal kya hua + agle events (NVIDIA + Google news)", key=f"ni_btn_{key}"):
                start_nifty_events(key, st.session_state.get("_news_intel_keys") or [], subject, heads, nvidia_keys=_nk)
        stt = events_state(key)
        rows = []
        if stt.get("status") == "running":
            st.info("⏳ Google se news/events aa rahe hain (15-40 sec). Page ko mat chhedo -- neeche apne aap dikh jayenge.")
        elif stt and time.time() - stt.get("ts", 0) < _ttl:
            if stt.get("err"):
                st.warning(stt["err"])
            rows = list(stt.get("rows") or [])
        today = datetime.now(IST).date()
        table = []
        for r in (rows if stock else fixed_calendar() + rows):
            up = str(r.get("status", "upcoming")).lower().startswith("up")
            table.append({"Status": "🕒 Aane wala" if up else "✅ Ho chuka", "Type": r.get("type") or "", "Kab (IST)": f"{r.get('date_ist')} {r.get('time_ist')}",
                          "Kya": r.get("event"), "Kis par asar": r.get("affects") or "Nifty / market",
                          "Direction": _EFFECT.get(str(r.get("expected_effect")).upper(), r.get("expected_effect")),
                          "India market me kab / kahan": ((("⏱ " + str(r.get("impact_time_ist")) + " · ") if str(r.get("impact_time_ist") or "unknown") != "unknown" else "") if rich else "") + (r.get("india_text") or (india_impact(r.get("date_ist"), r.get("time_ist")) if (up and str(r.get("date_ist")) != "unknown") else (r.get("impact_start") or r.get("india_note") or ("date pata nahi" if up else "ho chuka")))),
                          "Kitni der asar": r.get("duration") or r.get("reaction_window") or "", "Kyun": r.get("why") or "", **({"Mahatva": "⭐" * int(r.get("importance") or 3), "Asar kab se": r.get("impact_start") or "", "Sabut (headline)": r.get("evidence_text") or ""} if rich else {}), "_k": str(r.get("date_ist")) + str(r.get("time_ist"))})
        if key == "nifty50":                            # upcoming first (nearest first), then what already happened (newest first)
            _up = sorted([x for x in table if x["Status"].startswith("🕒")], key=lambda x: x["_k"])
            _dn = sorted([x for x in table if not x["Status"].startswith("🕒")], key=lambda x: x["_k"], reverse=True)
            table = _up + _dn
        elif not stock:
            table.sort(key=lambda x: x["_k"])
        # (stock rows are already ranked: upcoming first, then by importance)
        for x in table:
            x.pop("_k")
        if stock and not table and stt.get("status") != "running":
            st.caption("Is stock ke liye abhi koi badi/aane wali event nahi mili (ya research abhi chal raha hai). Poore market ki India/global news bhi dekhi gayi.")
        elif table:
            st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Events AI/Google-estimate hain, date-time khud verify karo; asar ek aam pattern hai, guarantee nahi. "
                   "Indian market: 09:15 open - 15:30 close (IST). Exchange holiday offline pata nahi chalta.")
    except Exception:
        pass
