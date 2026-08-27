import streamlit as st
import re
import io
import asyncio

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

try:
    import edge_tts
    HAS_TTS = True
except ImportError:
    edge_tts = None
    HAS_TTS = False


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
   vs SMC structure vs global sentiment. Explicitly call out where they agree and where they
   conflict, and explain what that conflict/agreement implies, before giving your overall read.
   For simple factual questions ("what's the RSI right now"), just answer directly and briefly --
   match the depth of the answer to the depth of the question.
5. Respond in the same language mix (Hindi/English/Hinglish) the user writes in.
6. If the user uploads a screenshot, describe what you actually see in it and relate it to the
   live snapshot data -- don't assume it matches the snapshot exactly, since screenshots may be
   from a different moment in time.
"""


def _is_quota_error(exc: Exception) -> bool:
    """Detects a rate-limit / quota-exhausted error from the Gemini API so we
    know it's worth rotating to the next key, instead of just failing on
    every kind of error (e.g. a bad prompt should not burn through all 5 keys)."""
    text = str(exc).lower()
    quota_signals = ["429", "quota", "rate limit", "resource_exhausted", "resource exhausted"]
    return any(sig in text for sig in quota_signals)


def _call_gemini_with_rotation(api_keys: list, model: str, contents: list):
    """Tries each Gemini API key in turn, starting from the last key that
    worked (remembered in session_state), so we don't retry an already
    -exhausted key from scratch on every single message. Moves to the next
    key only when the current one is out of quota / rate-limited; any other
    error is raised immediately since a different key won't fix it.

    Returns (response_text, key_index_used). Raises the last exception if
    every key is exhausted.
    """
    start_idx = st.session_state.get("gemini_working_key_idx", 0) % len(api_keys)
    order = list(range(start_idx, len(api_keys))) + list(range(0, start_idx))

    last_exc = None
    for idx in order:
        try:
            client = genai.Client(api_key=api_keys[idx])
            response = client.models.generate_content(model=model, contents=contents)
            st.session_state.gemini_working_key_idx = idx  # remember which key worked
            return response.text, idx
        except Exception as e:
            last_exc = e
            if _is_quota_error(e):
                continue  # this key's limit is over -- try the next one
            raise  # a non-quota error (bad request, network, etc.) -- retrying with another key won't help

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
                prompt_text = f"{SYSTEM_PREAMBLE}\n\n{context_block}\n\nCONVERSATION SO FAR:\n{history_text}\n\nRespond to the latest USER message."

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
                )
                if len(gemini_api_keys) > 1:
                    st.caption(f"🔑 Key #{key_idx_used + 1}/{len(gemini_api_keys)} se jawab mila.")
            except Exception as e:
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
