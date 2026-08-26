import streamlit as st
import re
import io
import asyncio

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    genai = None
    types = None
    HAS_GENAI = False

try:
    import edge_tts
    HAS_TTS = True
except ImportError:
    edge_tts = None
    HAS_TTS = False


def _get_safe_secret(key_name: str, default=None):
    """Safely fetches secrets without throwing exceptions if secrets.toml is missing."""
    try:
        return st.secrets.get(key_name, default)
    except Exception:
        return default


def _build_context_block(context: dict) -> str:
    """Turns the current dashboard's live numbers into a plain-text block."""
    if not isinstance(context, dict) or not context:
        return "=== LIVE DASHBOARD SNAPSHOT: No live data available ==="

    lines = ["=== LIVE DASHBOARD SNAPSHOT (this is the ONLY market data you may use) ==="]
    for label, value in context.items():
        if value is None or value == "":
            continue
        lines.append(f"- {label}: {value}")
    lines.append("=== END SNAPSHOT ===")
    return "\n".join(lines)


def _clean_for_speech(text: str) -> str:
    """Strips markdown formatting, code blocks, and URLs before speech synthesis."""
    text = re.sub(r'```.*?
    
