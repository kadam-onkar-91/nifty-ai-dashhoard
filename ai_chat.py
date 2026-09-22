from app_logging import get_logger
logger = get_logger(__name__)
import streamlit as st
import re
import io
import asyncio

try:
    from google import genai
    from google.genai import types as genai_types
    HAS_GENAI = True
except ImportError:
    genai = None
    genai_types = None
    HAS_GENAI = False

try:
    import edge_tts
    HAS_TTS = True
except ImportError:
    edge_tts = None
    HAS_TTS = False

from sniper_setup_framework import SNIPER_SETUP_FRAMEWORK


def _build_context_block(context: dict) -> str:
    """Turns the current dashboard's live numbers into a plain-text block
    that gets sent to Gemini alongside every question, so it always
    answers from the SAME live data the user is looking at on screen --
    not from its own general knowledge about markets."""
    lines = ["=== LIVE DASHBOARD SNAPSHOT (this is the ONLY market data you may use) ==="]
    for label, value in context.items():
        if value is None or value == "":
            continue
        lines.append(f"- {label}: {value}")
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


SYSTEM_PREAMBLE = """You are a trading-desk research assistant embedded inside a live Nifty 50 dashboard.
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
13. If external Google Search grounding is available for this request, use it only for current,
    relevant market-moving facts/news/events and clearly distinguish web-researched facts from
    dashboard-derived calculations. Never use external search to invent missing market prices.
"""

SYSTEM_PREAMBLE = SYSTEM_PREAMBLE + "\n\n" + SNIPER_SETUP_FRAMEWORK


def _is_quota_error(exc: Exception) -> bool:
    """Detects a rate-limit / quota-exhausted error from the Gemini API so we
    know it's worth rotating to the next key, instead of just failing on
    every kind of error (e.g. a bad prompt should not burn through all 5 keys)."""
    text = str(exc).lower()
    quota_signals = ["429", "quota", "rate limit", "resource_exhausted", "resource exhausted"]
    return any(sig in text for sig in quota_signals)


def _call_gemini_with_rotation(api_keys: list, model: str, contents: list,
                               enable_web_research: bool = False):
    """Rotate across the configured Gemini keys. For an explicit trade request,
    optionally enable Gemini's Google Search grounding when the installed SDK
    exposes it. If grounding is unsupported, retry the same key without it.
    """
    start_idx = st.session_state.get("gemini_working_key_idx", 0) % len(api_keys)
    order = list(range(start_idx, len(api_keys))) + list(range(0, start_idx))
    last_exc = None

    for idx in order:
        client = genai.Client(api_key=api_keys[idx])
        try:
            kwargs = {"model": model, "contents": contents}
            if enable_web_research and genai_types is not None:
                try:
                    kwargs["config"] = genai_types.GenerateContentConfig(
                        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
                    )
                except Exception:
                    logger.exception("Broad exception caught; fallback path executed")
                    # SDK/model may not expose Google Search grounding; use the
                    # same key normally rather than wasting another key/quota.
                    kwargs.pop("config", None)
            response = client.models.generate_content(**kwargs)
            st.session_state.gemini_working_key_idx = idx
            return response.text, idx
        except Exception as e:
            logger.exception("Broad exception caught; fallback path executed")
            last_exc = e
            if _is_quota_error(e):
                continue
            # If the optional grounding call itself is rejected by the model/API,
            # retry this SAME key once without grounding.
            if enable_web_research and "config" in locals() and "config" in kwargs:
                try:
                    response = client.models.generate_content(model=model, contents=contents)
                    st.session_state.gemini_working_key_idx = idx
                    return response.text, idx
                except Exception as retry_exc:
                    logger.exception("Broad exception caught; fallback path executed")
                    last_exc = retry_exc
                    if _is_quota_error(retry_exc):
                        continue
                    raise
            raise

    raise last_exc


def render_ai_chat(gemini_api_keys, dashboard_context: dict):
    """Renders a chat box at the point it's called. Every turn re-sends the
    CURRENT live dashboard_context (so answers always reflect this run's
    numbers, not stale ones from when the chat started), plus the running
    conversation, plus an optional uploaded screenshot.

    IMPORTANT: the caller must render this OUTSIDE any auto-refreshing
    st.fragment. If it lived inside one, the fragment's own refresh timer
    would tear down and rebuild this whole section every cycle -- cutting
    off audio mid-sentence and wiping any text you were still typing into
    the chat box before you hit enter.
    """
    st.subheader("💬 Ask the Dashboard AI")
    st.caption(
        "Isse dashboard ke live numbers ke baare me kuch bhi poochho -- deep analysis karke jawab dega. "
        "Ye sirf probability/analysis deta hai -- guarantee nahi, kyunki market ka koi bhi tool "
        "future ko pakka nahi bata sakta."
    )

    # Accept either a single key (string, old callers) or a list of keys.
    if isinstance(gemini_api_keys, str):
        gemini_api_keys = [gemini_api_keys] if gemini_api_keys else []

    if not gemini_api_keys or not HAS_GENAI:
        st.warning("Gemini API key configured nahi hai ya google-genai package missing hai -- chat available nahi hai.")
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
        with st.spinner("Live data padh raha hoon, deep analysis kar raha hoon..."):
            try:
                history_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}"
                    for m in st.session_state.dashboard_chat_history[-10:]
                )
                context_block = _build_context_block(dashboard_context)
                trade_mode = _is_explicit_trade_request(user_q)
                mode_block = (
                    "=== EXPLICIT TRADE REQUEST: FULL-DASHBOARD RESEARCH REQUIRED ===\n"
                    "Use ALL supplied dashboard sections, independently reconcile the evidence, and only then produce Entry/SL/Targets.\n"
                    if trade_mode else "=== NORMAL CHAT MODE ===\n"
                )
                prompt_text = f"{SYSTEM_PREAMBLE}\n\n{mode_block}\n{context_block}\n\nCONVERSATION SO FAR:\n{history_text}\n\nRespond to the latest USER message."

                contents = [prompt_text]
                if uploaded_img is not None:
                    contents.append({
                        "inline_data": {
                            "mime_type": uploaded_img.type,
                            "data": uploaded_img.getvalue(),
                        }
                    })

                answer, key_idx_used = _call_gemini_with_rotation(
                    api_keys=gemini_api_keys,
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                    enable_web_research=trade_mode,
                )
                if len(gemini_api_keys) > 1:
                    st.caption(f"🔑 Key #{key_idx_used + 1}/{len(gemini_api_keys)} se jawab mila.")
            except Exception as e:
                logger.exception("Broad exception caught; fallback path executed")
                if _is_quota_error(e):
                    answer = (
                        "⚠️ Saari 5 Gemini API keys ka quota abhi ke liye khatam ho gaya hai. "
                        "Thodi der (ya agle din) baad try karo, ya secrets me nayi key add karo."
                    )
                else:
                    answer = f"⚠️ AI se jawab nahi mil paya: {type(e).__name__}: {e}"

            st.markdown(answer)
            st.session_state.dashboard_chat_history.append({"role": "assistant", "content": answer})

            if HAS_TTS:
                _speak(answer)
