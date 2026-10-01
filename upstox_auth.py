from app_logging import get_logger
logger = get_logger(__name__)
import json
import os
import tempfile
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st

# =====================================================================
# UPSTOX LOGIN  (rewritten)
# ---------------------------------------------------------------------
# Why login "nahi hota" tha:
#   1) The token lived ONLY in st.session_state. Upstox login opens a NEW browser
#      tab (link_button) -> that tab is a brand-new session. A page refresh, a
#      websocket reconnect or a second tab = new session = token gone.
#   2) The ?code=... in the URL is SINGLE-USE. After a reconnect the app tried to
#      exchange the same dead code again, failed, and the error was swallowed
#      (`except: pass`), so the sidebar just kept showing "Login with Upstox".
#   3) Without a token every module silently fell back to Yahoo
#      (YAHOO_FALLBACK_NON_TRADABLE) -> slow dashboard AND zero trades.
# Fix: the token is kept in a process-wide store (shared by every session/tab,
# survives refresh/reconnect) until Upstox's own expiry (~03:30 IST next day),
# each auth code is exchanged exactly once, the code is removed from the URL, and
# the real Upstox error text is shown if anything fails.
# =====================================================================
_IST = timezone(timedelta(hours=5, minutes=30))
_LOCK = threading.Lock()
_STORE = {"token": None, "expires": 0.0, "checked": 0.0}
_CODE_RESULTS = {}          # auth code -> ("ok", token) | ("err", message)
_TOKEN_FILE = os.path.join(tempfile.gettempdir(), ".nifty_dash_upstox_token.json")
_PROFILE_URL = "https://api.upstox.com/v2/user/profile"
_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def _next_expiry_ts() -> float:
    """Upstox access tokens die at 03:30 IST. Keep a 5 min safety margin."""
    now = datetime.now(_IST)
    exp = now.replace(hour=3, minute=30, second=0, microsecond=0)
    if exp <= now:
        exp += timedelta(days=1)
    return (exp - timedelta(minutes=5)).timestamp()


def _save_token(token: str):
    with _LOCK:
        _STORE.update(token=token, expires=_next_expiry_ts(), checked=time.time())
    try:
        with open(_TOKEN_FILE, "w") as f:
            json.dump({"token": token, "expires": _STORE["expires"]}, f)
        os.chmod(_TOKEN_FILE, 0o600)
    except Exception:
        logger.warning("Could not persist Upstox token to temp file (memory store still works).")


def _load_token_from_disk():
    if _STORE["token"]:
        return
    try:
        with open(_TOKEN_FILE) as f:
            d = json.load(f)
        if d.get("token") and float(d.get("expires", 0)) > time.time():
            with _LOCK:
                _STORE.update(token=d["token"], expires=float(d["expires"]), checked=0.0)
    except Exception:
        pass


def _clear_token():
    with _LOCK:
        _STORE.update(token=None, expires=0.0, checked=0.0)
    try:
        os.remove(_TOKEN_FILE)
    except Exception:
        pass


def _stored_token():
    """Valid shared token or None. Re-validates against Upstox at most every 10 minutes."""
    _load_token_from_disk()
    tok = _STORE["token"]
    if not tok:
        return None
    if time.time() >= _STORE["expires"]:
        _clear_token()
        return None
    if time.time() - _STORE["checked"] > 600:
        try:
            r = requests.get(_PROFILE_URL, headers={"Accept": "application/json",
                                                    "Authorization": f"Bearer {tok}"}, timeout=4)
            if r.status_code == 401:
                _clear_token()
                return None
        except requests.exceptions.RequestException:
            pass            # network hiccup is not proof the token is bad
        _STORE["checked"] = time.time()
    return tok


def _exchange_code(code: str, api_key: str, api_secret: str, redirect_uri: str):
    """Exchange an auth code exactly once (cached result for repeats / parallel sessions)."""
    with _LOCK:
        done = _CODE_RESULTS.get(code)
    if done:
        return done
    payload = {"code": code, "client_id": api_key, "client_secret": api_secret,
               "redirect_uri": redirect_uri, "grant_type": "authorization_code"}
    headers = {"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    try:
        res = requests.post(_TOKEN_URL, data=payload, headers=headers, timeout=12)
        try:
            body = res.json()
        except Exception:
            body = {}
        if body.get("access_token"):
            result = ("ok", body["access_token"])
        else:
            errs = body.get("errors") or []
            msg = (errs[0].get("message") if errs and isinstance(errs[0], dict) else None) or res.text[:200]
            result = ("err", f"HTTP {res.status_code}: {msg}")
    except requests.exceptions.RequestException as exc:
        # network problem: do NOT cache, the code may still be unused
        return ("net", f"Upstox se connect nahi ho paya ({type(exc).__name__}). Dobara try karo.")
    with _LOCK:
        _CODE_RESULTS[code] = result
        if len(_CODE_RESULTS) > 20:
            for k in list(_CODE_RESULTS)[:-20]:
                _CODE_RESULTS.pop(k, None)
    return result


def _secrets():
    try:
        return (st.secrets["UPSTOX_API_KEY"], st.secrets["UPSTOX_API_SECRET"], st.secrets["REDIRECT_URI"])
    except Exception:
        return None


def get_upstox_access_token():
    st.sidebar.subheader("🔐 Upstox Live Authentication")
    if "access_token" not in st.session_state:
        st.session_state.access_token = None

    creds = _secrets()
    if creds is None:
        st.sidebar.error("Secrets me UPSTOX_API_KEY / UPSTOX_API_SECRET / REDIRECT_URI me se kuch missing hai.")
        return None
    API_KEY, API_SECRET, REDIRECT_URI = creds

    # 1) shared token (survives refresh / reconnect / second tab)
    shared = _stored_token()
    if shared:
        st.session_state.access_token = shared
    elif st.session_state.access_token and st.session_state.access_token != _STORE.get("token"):
        # shared store was cleared (expired / rejected by Upstox) -> drop the stale session copy
        st.session_state.access_token = None

    # 2) fresh login redirect: ?code=...
    auth_code = st.query_params.get("code")
    if auth_code and not st.session_state.access_token:
        kind, value = _exchange_code(str(auth_code), API_KEY, API_SECRET, REDIRECT_URI)
        if kind == "ok":
            _save_token(value)
            st.session_state.access_token = value
            st.session_state.pop("upstox_login_error", None)
        else:
            st.session_state["upstox_login_error"] = value
        if kind in ("ok", "err"):
            try:
                st.query_params.clear()      # a used code must never be re-sent
            except Exception:
                pass
    elif auth_code and st.session_state.access_token:
        try:
            st.query_params.clear()
        except Exception:
            pass

    if st.session_state.access_token:
        st.sidebar.success("🟢 Live Feed Active (Upstox connected)")
        if st.sidebar.button("Disconnect", key="upstox_disconnect_btn"):
            _clear_token()
            st.session_state.access_token = None
            st.rerun()
        return st.session_state.access_token

    err = st.session_state.get("upstox_login_error")
    if err:
        st.sidebar.error(f"Login fail: {err}\n\nCheck karo: Upstox portal me Redirect URI bilkul wahi ho jo secrets me "
                         f"REDIRECT_URI hai (`{REDIRECT_URI}`), aur API key/secret sahi ho.")

    encoded_redirect = urllib.parse.quote(REDIRECT_URI, safe="")
    login_url = (f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code"
                 f"&client_id={API_KEY}&redirect_uri={encoded_redirect}")
    st.sidebar.link_button("🚀 Login with Upstox", login_url)
    st.sidebar.caption("Login naye tab me khulta hai. PIN daalne ke baad ye dashboard apne aap connect ho jayega "
                       "(agle refresh me, max 30 sec). Purana tab band kar sakte ho.")
    st.sidebar.button("🔄 Login ho gaya? Status check karo", key="upstox_check_btn")
    return None
