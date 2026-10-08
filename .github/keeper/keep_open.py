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


def main() -> int:
    url = (os.environ.get("APP_URL") or "").strip()
    if not url.startswith("http"):
        log("APP_URL is missing. Add it under Settings -> Secrets and variables -> Actions -> Variables.")
        return 1
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
            try:
                page.wait_for_selector('[data-testid="stApp"]', timeout=120000)
            except Exception:
                log("Streamlit container not visible yet (app may still be starting).")

        def healthy() -> bool:
            try:
                if page.locator('[data-testid="stApp"]').count() == 0:
                    return False
                body = page.inner_text("body", timeout=15000)
                return not any(t in body for t in bad_text)
            except Exception:
                return False

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
                if not healthy():
                    fails += 1
                    log(f"Page not healthy (#{fails}) -> reloading.")
                    open_app(); last_reload = time.time()
                    continue
                fails = 0
                if time.time() - last_reload > RELOAD_EVERY_MIN * 60:
                    log("Periodic refresh of the tab.")
                    open_app(); last_reload = time.time()
                else:
                    log("tab alive")
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
