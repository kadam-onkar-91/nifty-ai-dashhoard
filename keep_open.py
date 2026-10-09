"""
keep_open.py -- the "robot tab".

Streamlit only runs the dashboard (research, shadow trades, learning, trades) while a browser tab is connected to it, and it puts
the app to sleep after a while with no visitor.  This script is run by GitHub Actions (free) during market hours: it opens the app
in a headless Chrome, wakes the app if it is asleep, and keeps the tab open until the market closes -- exactly as if you had the
website open on your phone.

Settings (GitHub -> Settings -> Secrets and variables -> Actions -> Variables):
    APP_URL     the full link of your Streamlit app (required)
Optional environment values (usually leave them alone):
    END_IST     "HH:MM" IST at which to stop.  Default: 14:55 for the morning run, 15:35 for the afternoon run.
    SKIP_DATES  comma separated YYYY-MM-DD market holidays to skip, e.g. "2026-11-08,2026-12-25"
"""
from __future__ import annotations

import os
import re
import sys
import time
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
MORNING_END = (14, 55)     # the morning job stops here; the afternoon job (queued behind it) takes over
AFTERNOON_END = (15, 35)
SPLIT_AT = (14, 50)        # a run that starts before this is the "morning" run
RELOAD_EVERY_MIN = 30      # a fresh page every so often, in case the websocket died silently
CHECK_EVERY_S = 60


def ist_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(IST)


def pick_end(now: datetime, override: str | None = None) -> datetime:
    """Absolute IST end time for this run."""
    if override:
        h, m = [int(x) for x in override.strip().split(":")[:2]]
    else:
        h, m = MORNING_END if (now.hour, now.minute) < SPLIT_AT else AFTERNOON_END
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


def is_skip_day(now: datetime, skip_dates: str | None) -> bool:
    if now.weekday() >= 5:
        return True
    days = {d.strip() for d in (skip_dates or "").split(",") if d.strip()}
    return now.date().isoformat() in days


def log(msg: str) -> None:
    print(f"[{ist_now().strftime('%H:%M:%S')} IST] {msg}", flush=True)


HEARTBEAT_RE = re.compile(r"IST Time:\s*(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2})")
MAX_LAG_S = 180            # the dashboard prints its own IST clock every cycle (~30 s); older than this = it stopped processing


def parse_heartbeat(body: str, now: datetime):
    """Newest 'IST Time: dd-Mon-YYYY HH:MM:SS' on the page -> lag in seconds behind real IST time, or None if not found."""
    try:
        found = HEARTBEAT_RE.findall(body or "")
        if not found:
            return None
        t = datetime.strptime(found[0], "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
        return (now - t).total_seconds()
    except Exception:
        return None


def main() -> int:
    url = (os.environ.get("APP_URL") or "").strip()
    if not url.startswith("http"):
        log("APP_URL is missing. Add it under Settings -> Secrets and variables -> Actions -> Variables.")
        return 1
    host = urlparse(url).netloc.lower()
    if "share.streamlit.io" in host or not host:
        log("APP_URL is the Streamlit DASHBOARD link (share.streamlit.io), not your app's own link. Open your app, copy the address "
            "that ends with .streamlit.app and put THAT in the APP_URL variable.")
        return 1
    log(f"App host: {host}")
    now = ist_now()
    if is_skip_day(now, os.environ.get("SKIP_DATES")):
        log("Weekend or listed holiday -> nothing to do.")
        return 0
    end = pick_end(now, os.environ.get("END_IST"))
    if now >= end:
        log(f"Already past {end.strftime('%H:%M')} IST -> nothing to do.")
        return 0

    from playwright.sync_api import sync_playwright   # imported late so the helpers above can be tested without it

    wake_btn = re.compile(r"get this app back up", re.I)
    bad_text = ("Connection error", "Disconnected", "Oh no.", "This app has gone to sleep")
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        ctx = browser.new_context(viewport={"width": 1366, "height": 900})
        page = ctx.new_page()

        def app_scope():
            """The frame that really holds the Streamlit app.  On Streamlit Cloud the app lives INSIDE an iframe, so looking only at the
            outer page finds nothing (that was why the first version kept reloading)."""
            try:
                for f in page.frames:
                    try:
                        if f.locator('[data-testid="stApp"]').count() > 0:
                            return f
                    except Exception:
                        continue
            except Exception:
                pass
            return None

        def diag(tag: str) -> None:
            """What the robot sees, for debugging (also saved as keeper_last.png)."""
            try:
                frames = [f.url[:90] for f in page.frames]
                txt = page.inner_text("body", timeout=5000)[:240].replace("\n", " | ")
                log(f"diag[{tag}]: title={page.title()!r} frames={frames} text={txt!r}")
                page.screenshot(path="keeper_last.png")
            except Exception as exc:
                log(f"diag failed: {exc}")

        def wake_if_asleep() -> bool:
            try:
                btn = page.get_by_role("button", name=wake_btn)
                if btn.count():
                    log("App was asleep -> clicking 'get this app back up'.")
                    btn.first.click()
                    page.wait_for_timeout(20000)
                    return True
            except Exception as exc:
                log(f"wake check problem (ignored): {exc}")
            return False

        def open_app() -> None:
            page.goto(url, wait_until="domcontentloaded", timeout=180000)
            page.wait_for_timeout(8000)
            wake_if_asleep()
            deadline = time.time() + 150
            while time.time() < deadline:
                if app_scope() is not None:
                    return
                wake_if_asleep()
                page.wait_for_timeout(3000)
            log("Streamlit app not visible yet (app may still be starting).")
            diag("open")

        def healthy():
            """-> (ok, note).  ok=False when the page is broken OR the dashboard's own IST clock stopped moving (engine not cycling)."""
            try:
                scope = app_scope()
                if scope is None:
                    return False, "no Streamlit page"
                body = scope.locator("body").inner_text(timeout=15000)
                if any(t in body for t in bad_text):
                    return False, "error/disconnected text on page"
                lag = parse_heartbeat(body, ist_now())
                if lag is None:
                    return True, "heartbeat not visible yet"
                if lag > MAX_LAG_S:
                    return False, f"dashboard clock is {lag:.0f}s behind -> not refreshing"
                note = f"engine heartbeat OK (clock lag {max(lag, 0):.0f}s)"
                if "DATA STALE/FROZEN" in body:
                    note += " | WARNING: dashboard says DATA STALE/FROZEN (market data source)"
                if "Login with Upstox" in body and "Login ho gaya" not in body:
                    note += " | NOTE: Upstox login may be needed today"
                return True, note
            except Exception as exc:
                return False, f"check failed: {exc}"

        log(f"Opening the app; will stay until {end.strftime('%H:%M')} IST.")
        try:
            open_app()
        except Exception as exc:
            log(f"First open failed: {exc}")
        last_reload = time.time()
        fails = 0
        while ist_now() < end:
            time.sleep(CHECK_EVERY_S)
            try:
                if wake_if_asleep():
                    last_reload = time.time()
                    continue
                ok, note = healthy()
                if not ok:
                    fails += 1
                    if fails == 1:
                        log(f"Page not healthy (#1): {note} -> checking again in a minute.")
                        diag("unhealthy")
                        continue
                    log(f"Page not healthy (#{fails}): {note} -> reloading.")
                    open_app(); last_reload = time.time()
                    continue
                fails = 0
                if time.time() - last_reload > RELOAD_EVERY_MIN * 60:
                    log("Periodic refresh of the tab.")
                    open_app(); last_reload = time.time()
                else:
                    log(f"tab alive | {note}")
            except Exception as exc:
                log(f"loop problem (will retry): {exc}")
        try:
            page.screenshot(path="keeper_last.png", full_page=False)
        except Exception:
            pass
        browser.close()
    log("Market time over -> closing the tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
