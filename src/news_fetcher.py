import requests
import argparse
import json
import sys
import os
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

# Add AlphaEar skill scripts to path
# Configurable via environment variable, defaults to relative path from project root
ALPHA_EAR_PATH = Path(
    os.getenv(
        "ALPHA_EAR_SCRIPTS_PATH",
        str(Path(__file__).parent.parent / ".agents" / "skills" / "alphaear-news" / "scripts"),
    )
)
if ALPHA_EAR_PATH.exists():
    sys.path.append(str(ALPHA_EAR_PATH))
    try:
        from news_tools import NewsNowTools

        HAS_ALPHA_EAR = True
    except ImportError:
        HAS_ALPHA_EAR = False
else:
    HAS_ALPHA_EAR = False


TOPICS_MAP = {
    "CRUDP.PA": ["oil", "crude oil", "WTI", "OPEC", "energy"],
    "CL=F": ["crude oil", "WTI", "oil futures", "OPEC"],
    "SXRV.DE": ["NASDAQ", "QQQ", "tech stocks", "Fed", "earnings"],
    "^NDX": ["NASDAQ 100", "NDX", "tech stocks", "Fed"],
    "SXRV.FRK": ["NASDAQ", "QQQ", "tech stocks", "Fed"],
}


def fetch_alpha_vantage_news(ticker: str, api_key: str):
    """
    Fetches news and sentiment from Alpha Vantage for a given ticker.
    Tries the original ticker first, then falls back to broader topics.
    """
    queries = TOPICS_MAP.get(ticker, [ticker])[:2]

    all_headlines = []
    total_sentiment = 0
    sentiment_count = 0

    for query in queries:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&keywords={query}&apikey={api_key}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Bug A fix (July 2026): the old `break` on rate-limit silently zeroed
            # the score AND discarded headlines collected from earlier successful
            # queries. Now log explicitly and `continue` to preserve what we have.
            if "Information" in data:
                logger.warning(
                    f"Alpha Vantage rate-limit/notice for query '{query}': "
                    f"{str(data.get('Information', ''))[:120]}. Skipping this query."
                )
                continue

            for item in data.get("feed", []):
                title = item.get("title", "")
                if title and title not in all_headlines:
                    all_headlines.append(title)
                # Bug C fix (July 2026): the old loop summed ticker_sentiment_score
                # across ALL tickers mentioned in each article (e.g. SPY, MSFT in an
                # oil story), diluting the score toward 0. Now filter to rows whose
                # ticker field matches the target ticker.
                for sentiment in item.get("ticker_sentiment", []):
                    try:
                        sent_ticker = str(sentiment.get("ticker", "")).upper()
                        if sent_ticker and sent_ticker != ticker.upper():
                            continue
                        total_sentiment += float(sentiment.get("ticker_sentiment_score", 0))
                        sentiment_count += 1
                    except (ValueError, TypeError):
                        continue

            time.sleep(12)  # Respect Alpha Vantage free tier rate limit (5/min)

        except Exception as e:
            logger.warning(f"Alpha Vantage news fetch failed for query '{query}': {e}")
            continue

    overall_sentiment = total_sentiment / sentiment_count if sentiment_count > 0 else 0
    return all_headlines, overall_sentiment


def fetch_alpha_ear_trends():
    """
    Fetches hot trends from AlphaEar sources.
    """
    if not HAS_ALPHA_EAR:
        return [], 0

    try:
        tools = NewsNowTools()
        # Bug B fix (July 2026): the old call used `get_unified_trends()` which
        # returns a Markdown STRING, then iterated it with `.get("title", "")` —
        # iterating a string yields characters, `.get` raised AttributeError, and
        # the bare `except` swallowed it silently (so AlphaEar never contributed).
        # Now call `fetch_hot_news()` per source, which returns List[Dict] with a
        # "title" key (confirmed in news_tools.py).
        headlines = []
        for source_id in ("cls", "wallstreetcn"):
            try:
                items = tools.fetch_hot_news(source_id, count=10)
                for it in items:
                    t = it.get("title", "")
                    if t:
                        headlines.append(t)
            except Exception as e:
                logger.warning(f"AlphaEar fetch_hot_news('{source_id}') failed: {e}")
                continue
        # No per-article sentiment score from this source; neutral default when
        # headlines exist so it contributes context without biasing the score.
        return headlines, 0.0
    except Exception as e:
        logger.warning(f"AlphaEar trends failed: {e}")
        return [], 0


def fetch_google_news_rss(query: str, max_items: int = 10) -> tuple:
    """
    Fallback news source using Google News RSS (no API key required).
    Returns (headlines_list, sentiment_score).
    """
    url = f"https://news.google.com/rss/search?q={query}+commodity&hl=en-US&gl=US&ceid=US:en"
    headlines = []
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items = root.findall(".//item")[:max_items]
        for item in items:
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                headlines.append(title_el.text.strip())
        logger.info(f"Google News RSS: fetched {len(headlines)} headlines for query '{query}'")
    except Exception as e:
        logger.warning(f"Google News RSS fetch failed for query '{query}': {e}")
    return headlines, 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch news headlines and sentiment.")
    parser.add_argument("ticker", type=str, help="The stock ticker to fetch news for.")
    parser.add_argument("api_key", type=str, help="Your Alpha Vantage API key.")
    args = parser.parse_args()

    # Fetch from both sources
    av_headlines, av_sentiment = fetch_alpha_vantage_news(args.ticker, args.api_key)
    ae_headlines, ae_sentiment = fetch_alpha_ear_trends()

    # Fallback to Google News RSS if Alpha Vantage returned nothing
    gn_headlines = []
    gn_sentiment = 0.0
    if not av_headlines:
        gn_topics = TOPICS_MAP.get(args.ticker, [args.ticker])
        gn_query = "+".join(gn_topics[:3])
        gn_headlines, gn_sentiment = fetch_google_news_rss(gn_query)
        if gn_headlines:
            logger.info(f"Used Google News RSS fallback: {len(gn_headlines)} headlines")

    # Merge results
    all_headlines = av_headlines + ae_headlines + gn_headlines
    # Weighted average. The old `elif gn_headlines` / `else` branches were
    # identical (both = ae_sentiment) so gn_sentiment was never used — dead code.
    # Google News RSS provides no per-ticker sentiment (returns 0.0), so when AV
    # is empty we fall back to the neutral AlphaEar default.
    if av_headlines:
        final_sentiment = (av_sentiment * 0.7) + (ae_sentiment * 0.3)
    else:
        final_sentiment = ae_sentiment

    output = {
        "headlines": all_headlines[:20],  # Limit to top 20 for LLM context
        "sentiment": final_sentiment,
    }

    print(json.dumps(output))
