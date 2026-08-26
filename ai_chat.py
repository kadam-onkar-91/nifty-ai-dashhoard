import streamlit as st
import re
import io

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

try:
    from gtts import gTTS
    HAS_TTS = True
except ImportError:
    gTTS = None
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


def render_ai_chat(gemini_api_key: str, dashboard_context: dict):
    """Renders a chat box at the point it's called. Every turn re-sends the
    CURRENT live dashboard_context (so answers always reflect this run's
    numbers, not stale ones from when the chat started), plus the running
    conversation, plus an optional uploaded screenshot."""
    st.subheader("💬 Ask the Dashboard AI")
    st.caption(
        "Isse dashboard ke live numbers ke baare me kuch bhi poochho. "
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
        with st.spinner("Live data padh raha hoon..."):
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

                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=contents,
                )
                answer = response.text
            except Exception as e:
                answer = f"⚠️ AI se jawab nahi mil paya: {type(e).__name__}: {e}"

            st.markdown(answer)
            st.session_state.dashboard_chat_history.append({"role": "assistant", "content": answer})

            # ---------------------------------------------------
            # Text-to-speech: read the answer aloud in Hindi voice
            # ---------------------------------------------------
            if HAS_TTS:
                try:
                    speech_text = _clean_for_speech(answer)
                    if speech_text.strip():
                        tts = gTTS(text=speech_text, lang="hi", slow=False)
                        audio_buffer = io.BytesIO()
                        tts.write_to_fp(audio_buffer)
                        audio_buffer.seek(0)
                        st.audio(audio_buffer, format="audio/mp3")
                except Exception:
                    st.caption("🔇 Awaaz generate nahi ho payi is baar -- text jawab upar hai.")
