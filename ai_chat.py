from app_logging import get_logger
logger = get_logger(__name__)
import streamlit as st
import re
import io
import asyncio
import json
import time
import base64
import hashlib
import threading

import requests

try:
    import edge_tts
    HAS_TTS = True
except ImportError:
    edge_tts = None
    HAS_TTS = False

# Kept only so older code that does `from ai_chat import HAS_GENAI` never breaks.
# All Gemini calls now go through the REST router below (no SDK needed).
HAS_GENAI = True

from sniper_setup_framework import SNIPER_SETUP_FRAMEWORK


# =====================================================================
# GEMINI ROUTER  (shared by the chat box AND the trade-engine review)
# ---------------------------------------------------------------------
# Why chat "kabhi jawab deta hai kabhi nahi":
#   * 503 UNAVAILABLE = the MODEL is overloaded (same for all 5 keys), but the
#     old code raised on it immediately -> no retry, no other model.
#   * Every request used ONE model, so all 5 keys hit the same per-model quota.
#   * Every request re-sent everything (500 candles + full framework + web search
#     grounding) -> tokens/quota burned in a few questions.
# Fixes here: retry 503 -> fall to the NEXT model (each model has its own
# quota per key), remember exhausted key+model pairs (cooldown) so they are not
# hammered again, never burn 5 keys on a non-quota error.
# State is module-level, so it is shared by every session and by the engine.
# =====================================================================
GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_STATE_LOCK = threading.Lock()
_COOLDOWN = {}        # (key_id, model) -> unix time until which we skip this pair
_MODEL_BUSY = {}      # model -> unix time until which we skip it (503 overload)
_MODEL_DEAD = {}      # model -> unix time until which we skip it (404 / not available)
_RR = {"i": 0}        # round-robin start so load spreads across keys


class GeminiUnavailable(Exception):
    """kind: 'quota' | 'busy' | 'auth' | 'other'. retry_after in seconds (best guess)."""
    def __init__(self, kind, detail="", retry_after=0):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.retry_after = int(retry_after or 0)


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(sig in text for sig in ["429", "quota", "rate limit", "resource_exhausted", "resource exhausted"])


def _kid(key: str) -> str:
    return key[-8:] if key else ""


def _cooldown_seconds(resp_text: str) -> int:
    """How long to skip a key+model after a 429. Uses Google's own retryDelay if present."""
    t = resp_text or ""
    m = re.search(r'retryDelay"?\s*[:=]\s*"?(\d+(?:\.\d+)?)s', t)
    if m:
        return int(min(max(float(m.group(1)) + 2, 5), 600))
    low = t.lower()
    if "perday" in low.replace(" ", "") or "per day" in low or "daily" in low:
        return 3 * 3600          # daily quota gone -> stop probing for 3 hours
    return 60                     # per-minute style limit


def _post_once(model, key, body, timeout):
    """One REST call. Returns (kind, payload, detail).
    kind: ok | quota | busy | auth | missing | other"""
    try:
        resp = requests.post(
            _GEMINI_URL.format(model=model),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json=body, timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        return "busy", None, f"network {type(exc).__name__}"
    sc = resp.status_code
    if sc == 200:
        try:
            data = resp.json()
            cand = (data.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        except Exception as exc:
            return "other", None, f"unreadable reply ({type(exc).__name__})"
        if not text:
            return "other", None, f"empty reply (finish={cand.get('finishReason') if isinstance(cand, dict) else '?'})"
        return "ok", text, ""
    body_txt = resp.text[:600]
    if sc == 429:
        return "quota", _cooldown_seconds(body_txt), f"HTTP 429 {body_txt[:120]}"
    if sc in (500, 502, 503, 504):
        return "busy", None, f"HTTP {sc}"
    if sc in (401, 403):
        return "auth", None, f"HTTP {sc} {body_txt[:120]}"
    if sc == 404:
        return "missing", None, f"HTTP 404 model {model}"
    return "other", None, f"HTTP {sc} {body_txt[:160]}"


def gemini_generate(api_keys, prompt, image=None, want_json=False, temperature=None,
                    max_output_tokens=None, models=None, budget_s=55):
    """Send one prompt through key+model rotation. Returns (text, meta).
    image = {"mime_type": "...", "data": bytes} or None.
    Raises GeminiUnavailable when nothing could answer."""
    keys = []
    for k in (api_keys or []):
        if k and k not in keys:
            keys.append(k)
    if not keys:
        raise GeminiUnavailable("auth", "no Gemini API key configured")

    parts = [{"text": prompt}]
    if image:
        parts.append({"inline_data": {"mime_type": image["mime_type"],
                                      "data": base64.b64encode(image["data"]).decode()}})
    body = {"contents": [{"role": "user", "parts": parts}]}
    gen = {}
    if temperature is not None:
        gen["temperature"] = temperature
    if want_json:
        gen["responseMimeType"] = "application/json"
    if max_output_tokens:
        gen["maxOutputTokens"] = max_output_tokens
    if gen:
        body["generationConfig"] = gen

    models = list(models or GEMINI_MODELS)
    deadline = time.time() + budget_s
    saw = {"quota": 0, "busy": 0, "auth": 0, "other": 0}
    last_detail = ""
    n = len(keys)
    with _STATE_LOCK:
        start = _RR["i"] % n

    for model in models:
        now = time.time()
        if _MODEL_DEAD.get(model, 0) > now or _MODEL_BUSY.get(model, 0) > now:
            if _MODEL_BUSY.get(model, 0) > now:
                saw["busy"] += 1
            continue
        model_busy = False
        for off in range(n):
            if time.time() > deadline:
                break
            ki = (start + off) % n
            key = keys[ki]
            now = time.time()
            if _COOLDOWN.get((_kid(key), model), 0) > now or _COOLDOWN.get((_kid(key), "*"), 0) > now:
                continue
            for attempt in range(2):
                timeout = max(6, min(40, deadline - time.time()))
                kind, payload, detail = _post_once(model, key, body, timeout)
                if kind == "ok":
                    with _STATE_LOCK:
                        _RR["i"] = ki          # keep using the key that works
                    return payload, {"model": model, "key_index": ki, "keys": n}
                last_detail = f"{model}/key#{ki+1}: {detail}"
                if kind == "busy":
                    saw["busy"] += 1
                    if attempt == 0 and time.time() < deadline - 8:
                        time.sleep(1.3)
                        continue
                    model_busy = True
                    break
                if kind == "quota":
                    saw["quota"] += 1
                    with _STATE_LOCK:
                        _COOLDOWN[(_kid(key), model)] = time.time() + payload
                    break                       # next key, same model
                if kind == "auth":
                    saw["auth"] += 1
                    with _STATE_LOCK:
                        _COOLDOWN[(_kid(key), "*")] = time.time() + 6 * 3600
                    break
                if kind == "missing":
                    with _STATE_LOCK:
                        _MODEL_DEAD[model] = time.time() + 1800
                    model_busy = True           # skip rest of keys for this model
                    break
                saw["other"] += 1               # bad/empty reply -> next key
                break
            if model_busy:
                break
        if model_busy:
            with _STATE_LOCK:
                if _MODEL_DEAD.get(model, 0) <= time.time():
                    _MODEL_BUSY[model] = time.time() + 12   # brief pause, then retried

    # nothing answered
    now = time.time()
    waits = [v - now for v in _COOLDOWN.values() if v > now] + [v - now for v in _MODEL_BUSY.values() if v > now]
    retry_after = int(min(waits)) if waits else 30
    if saw["busy"] and not saw["quota"]:
        raise GeminiUnavailable("busy", last_detail or "model overloaded", retry_after)
    if saw["quota"] or (waits and not saw["other"]):
        raise GeminiUnavailable("quota", last_detail or "all keys cooling down", retry_after)
    if saw["auth"] and not saw["other"]:
        raise GeminiUnavailable("auth", last_detail, 0)
    raise GeminiUnavailable("other", last_detail or "unknown Gemini failure", retry_after)


def gemini_status_summary(api_keys=None):
    """Small dict for a sidebar/debug line: how many key+model pairs are cooling down."""
    now = time.time()
    cool = {k: int(v - now) for k, v in _COOLDOWN.items() if v > now}
    return {"cooling_pairs": len(cool), "busy_models": [m for m, v in _MODEL_BUSY.items() if v > now]}


# =====================================================================
# CONTEXT PACKING  -- every section is still sent (research stays complete),
# but giant raw tables are trimmed so one question does not cost 100k+ tokens.
# =====================================================================
def _build_context_block(context: dict, max_value_chars: int = 2500, max_total_chars: int = 32000) -> str:
    """Turns the current dashboard's live numbers into a plain-text block sent
    alongside every question, so answers come from the SAME live data shown on screen.
    Long values keep their most recent part (candle tables: the LAST rows)."""
    lines = ["=== LIVE DASHBOARD SNAPSHOT (this is the ONLY market data you may use) ==="]
    used = 0
    for label, value in context.items():
        if value is None or value == "":
            continue
        v = str(value)
        if len(v) > max_value_chars:
            if any(t in str(label) for t in ("Candle", "Primary Price", "Latest Price")):
                v = "...(older rows trimmed)... " + v[-max_value_chars:]
            else:
                v = v[:max_value_chars] + " ...(trimmed)"
        line = f"- {label}: {v}"
        if used + len(line) > max_total_chars:
            lines.append(f"- (remaining sections skipped to save quota: {label} ...)")
            break
        lines.append(line)
        used += len(line)
    lines.append("=== END SNAPSHOT ===")
    return "\n".join(lines)


def _is_explicit_trade_request(text: str) -> bool:
    """True only when the user explicitly asks the AI to produce a trade/setup.
    This keeps normal dashboard questions from triggering expensive deep research.
    """
    q = (text or "").lower().strip()
    phrases = [
        "ek trade do", "trade do", "trade plan do", "trade batao",
        "trade de", "best trade", "entry target sl", "entry batao",
        "buy ya sell", "buy or sell", "kahan buy", "kahan sell",
        "setup do", "setup batao", "trade setup", "signal do",
    ]
    return any(p in q for p in phrases)


def _clean_for_speech(text: str) -> str:
    """Strips markdown symbols (*, #, `, bullet dashes) before sending text
    to TTS, so the voice doesn't read out literal asterisks/hashes."""
    text = re.sub(r'[*_#`]', '', text)
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    return text


async def _generate_speech_bytes(text: str) -> bytes:
    """Natural, human-sounding Hindi male voice (Microsoft Edge Neural TTS,
    free, no API key). '+12%' rate makes it speak a bit faster/livelier
    than the default pace, closer to how a person actually talks."""
    communicate = edge_tts.Communicate(text=text, voice="hi-IN-MadhurNeural", rate="+12%")
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes


def _speak(text: str):
    """Renders an audio player for the given text using the natural Hindi
    voice. Safe to call from inside a normal (sync) Streamlit script."""
    try:
        clean_text = _clean_for_speech(text)
        if not clean_text.strip():
            return
        audio_bytes = asyncio.run(_generate_speech_bytes(clean_text))
        if audio_bytes:
            st.audio(io.BytesIO(audio_bytes), format="audio/mp3")
    except Exception:
        logger.exception("Broad exception caught; fallback path executed")
        st.caption("🔇 Awaaz generate nahi ho payi is baar -- text jawab upar hai.")


SYSTEM_PREAMBLE_CORE = """You are a trading-desk research assistant embedded inside a live Nifty 50 dashboard.
Rules you must always follow:
1. Answer ONLY using the LIVE DASHBOARD SNAPSHOT block provided with each question, plus the
   conversation so far. Do not invent price levels, percentages, or news that aren't in the snapshot.
2. You are NOT a financial advisor and must never phrase anything as a guarantee or certainty
   ("will definitely go up/down", "100% chance"). Use probabilistic, hedged language, the way a
   professional trading desk analyst would -- because the snapshot itself is a probabilistic model
   output, not a fact about the future.
3. If the user asks something the snapshot has no data for, say so plainly instead of guessing.
4. DO DEEP RESEARCH, not one-liners: when the question is about market direction, a signal, or
   "what should I know", actively cross-check EVERY relevant metric in the snapshot against each
   other -- confluence signal vs breadth vs VWAP vs PCR vs FII footprint vs Bank Nifty correlation
   vs SMC structure vs global sentiment vs Max Pain gravity. Explicitly call out where they agree
   and where they conflict, and explain what that conflict/agreement implies, before giving your
   overall read. For simple factual questions ("what's the RSI right now"), just answer directly
   and briefly -- match the depth of the answer to the depth of the question.
5. Respond in the same language mix (Hindi/English/Hinglish) the user writes in.
6. If the user uploads a screenshot, describe what you actually see in it and relate it to the
   live snapshot data -- don't assume it matches the snapshot exactly, since screenshots may be
   from a different moment in time.
7. ALWAYS check the "Market Status" and "Data Freshness" fields in the snapshot before answering
   any trade-related question. If Data Freshness says STALE/FROZEN, or Market Status says the
   market is closed, say so explicitly and make clear the rest of the numbers are from the last
   available session, not right now -- don't discuss them as if they're live in that case.
8. You have ALSO been given a separate SNIPER SETUP FRAMEWORK document below (loaded from
   sniper_setup_framework.py). Read it carefully -- it is a full trading knowledge base you must
   apply whenever the user asks for a trade signal, a setup, or "should I take a trade". For
   simple factual questions, you don't need to invoke the whole framework -- just answer directly.
9. NEVER volunteer a specific Entry price, Target, or Stop-Loss unless the user's CURRENT message
   explicitly asks for one. Give the Setup Bias / Key Levels / Confluence / Trigger analysis
   without it by default. Only when they explicitly ask (e.g. "entry target batao", "trade plan
   do") should you compute and state Entry/Target/SL -- and always compute it fresh from that
   message's live snapshot, never by repeating a number from earlier in the conversation.
10. When the user asks "buy ya sell entry hai kya", or asks how confident/accurate a call is, or
    asks for an "up % vs down %" chance: NEVER invent a made-up statistical probability (e.g. "73%
    chance of going up") -- a language model guessing a precise number like that is fabrication,
    not real statistics, and could get the user hurt financially. Instead:
    a) Report the actual "AI Confluence Score" from the snapshot as-is -- this is a real number
       computed by the app's code from how many technical signals align, not a guess.
    b) Break down which individual factors in the snapshot point bullish vs bearish (RSI, VWAP,
       EMA 20/50, MACD, Full/Heavyweight Breadth, Smart Money Structure, FII/DII Footprint, Bank
       Nifty correlation, PCR, Max Pain Gravity, Global Sentiment) as a simple count, e.g. "6 out
       of 9 tracked factors lean bullish, 3 lean bearish" -- only using factors present in the
       snapshot, never inventing ones that aren't there.
    c) Call this a "confluence lean strength", explicitly NOT a scientific probability of where
       price will go -- markets can and do move against strong confluence.
    d) Point to the "Live DB Win Rate" and "Walk-Forward ML Accuracy" fields in the snapshot as the
       closest thing to a real, historically-grounded accuracy figure, since those come from
       actual logged outcomes/backtesting rather than a single-moment guess.
11. EXPLICIT TRADE REQUEST MODE: If the CURRENT user message explicitly asks for a trade/setup
    (for example "ek trade do", "trade plan do", "buy ya sell", "entry target SL batao"), switch
    into FULL-DASHBOARD TRADE RESEARCH mode. Before producing the setup, synthesize EVERY
    AVAILABLE section of the supplied dashboard snapshot, not just the existing dashboard signal.
    Give priority to live/fresh data and explicitly flag anything STALE, SIMULATED, unavailable,
    or delayed. Re-check: spot/OHLC, all available timeframes, BOS/CHoCH/SMC/FVG/OB/sweeps,
    liquidity map and S/R ladder, CPR/pivots/PDH/PDL/opening range, VWAP/POC/volume, option-chain
    OI/OI-change/PCR/IV/Greeks/skew/max-pain when supplied, order-book imbalance, breadth/sectors,
    FII/DII footprint, Bank Nifty/Sensex correlation, VIX/volatility regime, global markets/news,
    ML raw + calibrated confidence and OOS metrics, backtest/Monte-Carlo/drift, position sizing and
    risk-engine blocks. Do NOT blindly copy the dashboard's precomputed Entry/SL/Target: independently
    reconcile the evidence and derive the setup from the CURRENT snapshot. If critical evidence
    conflicts or is missing, downgrade to WATCH/NO TRADE rather than forcing a trade.
12. In FULL-DASHBOARD TRADE RESEARCH mode, output: Market State/Regime -> Evidence that agrees ->
    Evidence that conflicts -> Trade direction (BUY/SELL/NO TRADE) -> exact Entry/trigger -> SL/
    invalidation -> T1/T2/T3 -> R:R -> position-size/risk note -> what must happen before entry ->
    what cancels the setup -> data freshness/source notes. Any probability/confidence must be an
    actual model/calibration value supplied by the snapshot; never invent a percentage.
13. You do NOT have live web search in this chat. Never claim you searched the web. For news / global
    context use ONLY the news, global-market and research sections present in the snapshot, and say
    plainly when a needed fact is missing. Never invent missing market prices.
"""

# Full preamble (rules + sniper framework) is only sent for explicit trade requests.
SYSTEM_PREAMBLE = SYSTEM_PREAMBLE_CORE + "\n\n" + SNIPER_SETUP_FRAMEWORK

_GREETINGS = {"hi", "hii", "hiii", "hello", "hey", "hlo", "helo", "namaste", "good morning",
              "good evening", "good afternoon", "kaise ho", "kese ho", "thanks", "thank you",
              "shukriya", "ok", "okay", "theek hai", "thik hai"}


def _is_small_talk(text: str) -> bool:
    q = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
    return q in _GREETINGS


def _local_snapshot_answer(context: dict, trade_mode: bool, note: str) -> str:
    """Zero-quota answer built straight from the dashboard snapshot. Used when every Gemini
    key/model is busy or exhausted, so the chat box NEVER just shows an error."""
    def g(k):
        v = context.get(k)
        return None if v in (None, "") else str(v)[:600]

    rows = []
    for label, key in [
        ("Market", "Market Status"), ("Data", "Data Freshness"), ("Spot", "Nifty Spot Price"),
        ("Confluence Signal", "Confluence Signal"), ("AI Confluence Score", "AI Confluence Score"),
        ("Smart Money", "Smart Money Structure (BOS/CHoCH)"), ("VWAP", "VWAP"), ("RSI", "RSI"),
        ("PCR", "Option Chain PCR"), ("Max Pain", "Max Pain Gravity"),
        ("FII/DII", "FII / DII Footprint"), ("Breadth (50)", "Full Nifty 50 Breadth"),
        ("Bank Nifty corr.", "Bank Nifty ⇄ Nifty 50 Correlation"),
        ("Early Warning S/R", "Early Warning (S/R Approach Predictor)"),
        ("Sniper Setup", "Sniper Setup (PDH/PDL/CPR + SMC + OI)"),
        ("Active trade", "Active DB Trade"), ("Live win rate", "Live DB Win Rate"),
    ]:
        v = g(key)
        if v:
            rows.append(f"- **{label}:** {v}")
    if trade_mode:
        v = g("Entry / SL / T1 / T2")
        if v:
            rows.append(f"- **Dashboard plan (calculated, AI-verified nahi):** {v}")
    status = (g("Market Status") or "").upper()
    fresh = (g("Data Freshness") or "").upper()
    warn = ""
    if "CLOSED" in status or "STALE" in fresh or "FROZEN" in fresh:
        warn = "\n\n⚠️ Market closed / data stale hai -- abhi fresh trade lena sahi nahi, ye figures last session ke hain."
    body = "\n".join(rows) if rows else "Dashboard snapshot abhi khaali hai -- pehle dashboard poora load hone do."
    return f"⚠️ {note}\n\n**Dashboard ka calculated snapshot (AI-written analysis nahi):**\n{body}{warn}"


def _friendly_error(exc: GeminiUnavailable) -> str:
    wait = f" (~{max(exc.retry_after, 5)} sec baad phir try karo)" if exc.retry_after else ""
    if exc.kind == "busy":
        return f"Gemini abhi overloaded hai (503) -- sab models try kar liye{wait}."
    if exc.kind == "quota":
        return f"Gemini ka quota abhi sab keys/models par khatam ya cooldown me hai{wait}."
    if exc.kind == "auth":
        return "Gemini API key invalid/blocked lag rahi hai -- secrets me key check karo."
    return f"Gemini se jawab nahi mila{wait}."


def get_trade_research(gemini_api_keys, dashboard_context: dict, cache_seconds: int = 180):
    """Optional independent AI second opinion (soft vote). Not called by the main app
    flow; kept for compatibility. Uses the shared router + a longer cache."""
    if isinstance(gemini_api_keys, str):
        gemini_api_keys = [gemini_api_keys] if gemini_api_keys else []
    if not gemini_api_keys or not dashboard_context:
        return {}
    try:
        key_material = "|".join([
            str(dashboard_context.get("Nifty Spot Price", "")),
            str(dashboard_context.get("Data Freshness", "")),
            str(dashboard_context.get("Confluence Signal", "")),
            str(dashboard_context.get("RAW S/R Ladder", ""))[:600],
        ])
        cache = st.session_state.get("trade_research_cache")
        now = time.time()
        if cache and cache.get("key") == key_material and now - float(cache.get("ts", 0)) < cache_seconds:
            return cache.get("result") or {}
        snapshot = _build_context_block(dashboard_context, 2500, 30000)
        prompt = ("You are an independent second-opinion research layer for a live NIFTY 50 trading engine.\n"
                  "Use ONLY the supplied snapshot. Do not invent prices or probabilities. Strong opposite OI wall, "
                  "stale data or clear invalidation => WAIT.\nReturn ONLY JSON: {\"direction\":\"BUY|SELL|WAIT\","
                  "\"verdict\":\"TRADEABLE|WATCH|NO_TRADE\",\"strength\":\"STRONG|MODERATE|WEAK\","
                  "\"rationale\":\"short Hinglish\",\"conflicts\":[\"...\"],"
                  "\"trigger_type\":\"BOUNCE|REJECTION|BREAKOUT|BREAKDOWN|MOMENTUM|WAIT\"}\n\n" + snapshot)
        text, _ = gemini_generate(gemini_api_keys, prompt, want_json=True, temperature=0.0)
        raw = (text or "").strip()
        st_, en_ = raw.find("{"), raw.rfind("}")
        if st_ < 0 or en_ <= st_:
            raise ValueError("AI research did not return JSON")
        result = json.loads(raw[st_:en_ + 1])
        d = str(result.get("direction", "WAIT")).upper()
        result["direction"] = d if d in ("BUY", "SELL", "WAIT") else "WAIT"
        result["verdict"] = str(result.get("verdict", "WATCH")).upper()
        result["strength"] = str(result.get("strength", "WEAK")).upper()
        result["source"] = "Gemini independent dashboard research"
        st.session_state.trade_research_cache = {"ts": now, "key": key_material, "result": result}
        return result
    except Exception:
        logger.exception("Automatic trade research failed; deterministic engine continues")
        return {}


def render_ai_chat(gemini_api_keys, dashboard_context: dict):
    """Chat box. Must be rendered OUTSIDE any auto-refreshing st.fragment (see app.py)."""
    st.subheader("💬 Ask the Dashboard AI")
    st.caption(
        "Isse dashboard ke live numbers ke baare me kuch bhi poochho -- deep analysis karke jawab dega. "
        "Ye sirf probability/analysis deta hai -- guarantee nahi, kyunki market ka koi bhi tool "
        "future ko pakka nahi bata sakta."
    )

    if isinstance(gemini_api_keys, str):
        gemini_api_keys = [gemini_api_keys] if gemini_api_keys else []

    if not gemini_api_keys:
        st.warning("Gemini API key configured nahi hai -- chat available nahi hai.")
        return

    if "dashboard_chat_history" not in st.session_state:
        st.session_state.dashboard_chat_history = []

    for msg in st.session_state.dashboard_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    uploaded_img = st.file_uploader(
        "Optional: chat/chart ka screenshot upload karo",
        type=["png", "jpg", "jpeg"],
        key=f"ai_chat_img_{len(st.session_state.dashboard_chat_history)}",
    )

    user_q = st.chat_input("Apna sawaal likho...")
    if not user_q:
        return

    st.session_state.dashboard_chat_history.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        trade_mode = _is_explicit_trade_request(user_q)
        answer = None
        used_ai = False

        # 1) Greetings cost ZERO quota.
        if _is_small_talk(user_q) and uploaded_img is None:
            answer = ("Hii! Main dashboard ka research assistant hoon. Market, levels, OI, trade setup -- "
                      "jo poochna hai poochho. Trade chahiye to likho \"ek trade do\".")
        else:
            # 2) Same question on same market data within 45s -> reuse (double-tap / re-ask).
            snap_key = "|".join(str(dashboard_context.get(k, "")) for k in
                                ("Nifty Spot Price", "Market Status", "Confluence Signal", "Data Freshness"))
            cache_id = hashlib.md5(f"{user_q.strip().lower()}|{trade_mode}|{snap_key}|{uploaded_img is not None}".encode()).hexdigest()
            cache = st.session_state.setdefault("_chat_answer_cache", {})
            hit = cache.get(cache_id)
            if hit and time.time() - hit["ts"] < 45 and uploaded_img is None:
                answer, used_ai = hit["answer"], True
            else:
                with st.spinner("Live data padh raha hoon, deep analysis kar raha hoon..."):
                    try:
                        history_text = "\n".join(
                            f"{m['role'].upper()}: {m['content'][:1200]}"
                            for m in st.session_state.dashboard_chat_history[-7:]
                        )
                        if trade_mode:
                            context_block = _build_context_block(dashboard_context, 4500, 60000)
                            preamble = SYSTEM_PREAMBLE
                            mode_block = ("=== EXPLICIT TRADE REQUEST: FULL-DASHBOARD RESEARCH REQUIRED ===\n"
                                          "Use ALL supplied dashboard sections, independently reconcile the evidence, "
                                          "and only then produce Entry/SL/Targets.\n")
                        else:
                            context_block = _build_context_block(dashboard_context, 2500, 32000)
                            preamble = SYSTEM_PREAMBLE_CORE
                            mode_block = "=== NORMAL CHAT MODE ===\n"
                        prompt_text = (f"{preamble}\n\n{mode_block}\n{context_block}\n\n"
                                       f"CONVERSATION SO FAR:\n{history_text}\n\nRespond to the latest USER message.")
                        image = None
                        if uploaded_img is not None:
                            image = {"mime_type": uploaded_img.type, "data": uploaded_img.getvalue()}
                        text, meta = gemini_generate(gemini_api_keys, prompt_text, image=image)
                        answer, used_ai = text, True
                        cache[cache_id] = {"ts": time.time(), "answer": answer}
                        if len(cache) > 15:
                            for old in list(cache.keys())[:-15]:
                                cache.pop(old, None)
                        st.caption(f"🔑 Key #{meta['key_index'] + 1}/{meta['keys']} | {meta['model']}")
                    except GeminiUnavailable as e:
                        logger.warning("Chat Gemini unavailable: %s", e)
                        answer = _local_snapshot_answer(dashboard_context or {}, trade_mode, _friendly_error(e))
                    except Exception as e:
                        logger.exception("Chat failed unexpectedly")
                        answer = _local_snapshot_answer(dashboard_context or {}, trade_mode,
                                                        f"AI se jawab nahi mil paya ({type(e).__name__}).")

        st.markdown(answer)
        st.session_state.dashboard_chat_history.append({"role": "assistant", "content": answer})

        # Voice only for real AI answers (skip for the fallback text), and capped so it stays fast.
        if HAS_TTS and used_ai and len(answer) > 40:
            _speak(answer[:1800])
