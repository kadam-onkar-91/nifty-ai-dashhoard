import streamlit as st
import re
import io
import asyncio
import time

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

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


# Each Gemini model on the free tier has its OWN separate daily quota, so
# trying them in order (cheapest/fastest first) multiplies the effective
# free capacity instead of stopping at the first model's 500-1000/day cap.
GEMINI_MODEL_FALLBACK_CHAIN = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

# Groq is a completely separate company/free tier from Google -- used only
# as a last resort if EVERY Gemini model's daily quota is exhausted.
GROQ_MODEL = "llama-3.3-70b-versatile"


def _call_groq(prompt_text: str, groq_api_key: str) -> str:
    """Last-resort fallback once every Gemini model's free quota is used up
    for the day. Groq has its own independent free tier, so this only kicks
    in when Gemini is completely exhausted -- not used for images, since
    this path is text-only."""
    if not groq_api_key or not HAS_REQUESTS:
        raise RuntimeError("Groq not configured")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {groq_api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


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


def render_ai_chat(gemini_api_key: str, dashboard_context: dict, groq_api_key: str = None):
    """Renders a chat box at the point it's called. Every turn re-sends the
    CURRENT live dashboard_context (so answers always reflect this run's
    numbers, not stale ones from when the chat started), plus the running
    conversation, plus an optional uploaded screenshot.

    groq_api_key is OPTIONAL -- if not provided, the app still works, it
    just won't have the last-resort Groq fallback once all Gemini models'
    daily quotas are exhausted.

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

    if not gemini_api_key or not HAS_GENAI:
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
            answer = None
            used_fallback_note = None
            try:
                client = genai.Client(api_key=gemini_api_key)

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

                last_err = None
                quota_exhausted_models = []

                # 1) Walk the Gemini model fallback chain -- each model has
                #    its own independent free-tier daily quota, so a 429 on
                #    one model just moves to the next, not to a hard stop.
                for model_name in GEMINI_MODEL_FALLBACK_CHAIN:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                        )
                        answer = response.text
                        if model_name != GEMINI_MODEL_FALLBACK_CHAIN[0]:
                            used_fallback_note = f"_(via {model_name} -- pehla model busy tha)_"
                        break
                    except Exception as e:
                        last_err = e
                        err_str = str(e)
                        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                            quota_exhausted_models.append(model_name)
                            continue  # try next model in the chain
                        # Non-quota error (network blip) -- one short retry
                        # on this same model before moving on.
                        time.sleep(1.5)
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=contents,
                            )
                            answer = response.text
                            break
                        except Exception as e2:
                            last_err = e2
                            continue

                # 2) If every Gemini model is quota-exhausted, fall back to
                #    Groq (separate provider, separate quota) -- text only.
                if answer is None and groq_api_key:
                    try:
                        answer = _call_groq(prompt_text, groq_api_key)
                        used_fallback_note = "_(Gemini aaj busy tha, isliye Groq se jawab diya)_"
                    except Exception as e:
                        last_err = e

                if answer is None:
                    if quota_exhausted_models:
                        answer = (
                            "⚠️ Aaj ke free requests Gemini ke saare models pe khatam ho gaye hain. "
                            "Kal quota reset hone ke baad dobara try karo, ya billing enable karo "
                            "zyada requests ke liye."
                        )
                    else:
                        answer = (
                            f"⚠️ AI se jawab nahi mil paya (kuch technical dikkat hui). "
                            f"Thodi der baad phir try karo. [{type(last_err).__name__ if last_err else 'Unknown'}]"
                        )
            except Exception as e:
                answer = f"⚠️ AI se jawab nahi mil paya: {type(e).__name__}: {e}"

            st.markdown(answer)
            if used_fallback_note:
                st.caption(used_fallback_note)
            st.session_state.dashboard_chat_history.append({"role": "assistant", "content": answer})

            if HAS_TTS:
                _speak(answer)
