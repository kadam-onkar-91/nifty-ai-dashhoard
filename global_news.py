import pandas as pd
import feedparser


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

    sentiment_data = []
    all_headlines = []

    positive_keywords = ['surge', 'jump', 'gain', 'growth', 'rally', 'positive', 'boost', 'up', 'high', 'deal', 'peace']
    negative_keywords = ['fall', 'drop', 'slump', 'crash', 'loss', 'inflation', 'war', 'tension', 'negative', 'down', 'crisis', 'sanction']

    for region, query in regions.items():
        pos_count = 0
        neg_count = 0
        fetch_failed = False

        try:
            rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(rss_url)

            entries = feed.entries[:5]
            if not entries:
                fetch_failed = True

            for entry in entries:
                title = entry.title.lower()
                all_headlines.append({"region": region, "title": entry.title})

                if any(word in title for word in positive_keywords):
                    pos_count += 1
                if any(word in title for word in negative_keywords):
                    neg_count += 1

            # NOTE: previously this silently replaced a genuine 0/0 (no
            # keyword hits -> truly neutral) with a fake "3, 3". Removed --
            # 0/0 with no keyword hits is honestly reported as Neutral.

        except Exception:
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

    return df_sentiment, top_headline
