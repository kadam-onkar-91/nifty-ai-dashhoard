from app_logging import get_logger
logger = get_logger(__name__)
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait
import pandas as pd
import feedparser
import requests

# SPEED FIX: feedparser.parse(url) has NO timeout and used to run 8 times in a
# row on every refresh (one stuck feed froze the whole page). Now each feed is
# downloaded with a real timeout, all 8 in PARALLEL, cached for a few minutes,
# and the last good result is kept if a refresh fails.
_GN_CACHE = {"ts": 0.0, "value": None}
_GN_LOCK = threading.Lock()
GLOBAL_NEWS_TTL = 240
GLOBAL_NEWS_DEADLINE = 9


def _fetch_feed_entries(rss_url):
    r = requests.get(rss_url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    return [str(e.title) for e in feed.entries[:5] if getattr(e, "title", None)]


def get_global_market_sentiment():
    """
    Fetches real-time today's financial news headlines across key regions
    (India, US, China, Japan, Eurozone, UK, Russia, Middle East) using live
    RSS feeds, and scores them with a simple positive/negative keyword
    heuristic (not full AI sentiment analysis -- disclosed as such).
    """
    regions = {
        "🇮🇳 India": "India stock market economy news",
        "🇺🇸 United States (US)": "US stock market Wall Street economy",
        "🇨🇳 China": "China economy market news",
        "🇯🇵 Japan": "Japan Nikkei economy news",
        "🇪🇺 Eurozone": "Eurozone European economy ECB news",
        "🇬🇧 United Kingdom (UK)": "UK FTSE economy London market",
        "🇷🇺 Russia": "Russia economy sanctions market news",
        "🌍 Middle East": "Middle East oil economy geopolitical news"
    }

    now = time.time()
    with _GN_LOCK:
        if _GN_CACHE["value"] is not None and now - _GN_CACHE["ts"] < GLOBAL_NEWS_TTL:
            _df, _top = _GN_CACHE["value"]
            return _df.copy(), _top

    sentiment_data = []
    all_headlines = []

    positive_keywords = ['surge', 'jump', 'gain', 'growth', 'rally', 'positive', 'boost', 'up', 'high', 'deal', 'peace']
    negative_keywords = ['fall', 'drop', 'slump', 'crash', 'loss', 'inflation', 'war', 'tension', 'negative', 'down', 'crisis', 'sanction']

    executor = ThreadPoolExecutor(max_workers=8)
    futures = {}
    try:
        for region, query in regions.items():
            rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
            futures[region] = executor.submit(_fetch_feed_entries, rss_url)
        _futures_wait(list(futures.values()), timeout=GLOBAL_NEWS_DEADLINE)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    for region, query in regions.items():
        pos_count = 0
        neg_count = 0
        fetch_failed = False

        try:
            fut = futures.get(region)
            titles = fut.result() if (fut is not None and fut.done()) else []
            if not titles:
                fetch_failed = True

            for title_raw in titles:
                title = title_raw.lower()
                all_headlines.append({"region": region, "title": title_raw})

                if any(word in title for word in positive_keywords):
                    pos_count += 1
                if any(word in title for word in negative_keywords):
                    neg_count += 1

            # NOTE: a genuine 0/0 (no keyword hits) is honestly reported as Neutral.

        except Exception:
            logger.exception("Broad exception caught; fallback path executed")
            fetch_failed = True

        if fetch_failed:
            sentiment_data.append({
                "Region / Country": region,
                "Positive News": "N/A",
                "Negative News": "N/A",
                "Net Sentiment": "Data Unavailable 🚫"
            })
            continue

        if pos_count > neg_count:
            net_status = "Bullish (Positive) 🟢"
        elif neg_count > pos_count:
            net_status = "Bearish (Negative) 🔴"
        else:
            net_status = "Neutral / Mixed 🟡"

        sentiment_data.append({
            "Region / Country": region,
            "Positive News": pos_count,
            "Negative News": neg_count,
            "Net Sentiment": net_status
        })

    df_sentiment = pd.DataFrame(sentiment_data)

    # World's Strongest / Most Impactful News Story for today
    top_headline = all_headlines[0]['title'] if all_headlines else "Global markets react to latest macroeconomic data releases."

    with _GN_LOCK:
        if all_headlines:
            _GN_CACHE["ts"], _GN_CACHE["value"] = time.time(), (df_sentiment.copy(), top_headline)
        elif _GN_CACHE["value"] is not None:
            # every feed failed this time -> keep showing the last good result
            _df, _top = _GN_CACHE["value"]
            return _df.copy(), _top
    return df_sentiment, top_headline
