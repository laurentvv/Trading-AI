import requests
import argparse
import json
import sys
import os
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Add AlphaEar skill scripts to path
# Configurable via environment variable, defaults to relative path from project root
ALPHA_EAR_PATH = Path(
    os.getenv(
        "ALPHA_EAR_SCRIPTS_PATH",
        str(
            Path(__file__).parent.parent
            / ".agents"
            / "skills"
            / "alphaear-news"
            / "scripts"
        ),
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


def fetch_alpha_vantage_news(ticker: str, api_key: str):
    """
    Fetches news and sentiment from Alpha Vantage for a given ticker.
    Tries the original ticker first, then falls back to broader topics.
    """
    topics_map = {
        "CRUDP.PA": ["oil", "crude oil", "WTI", "OPEC", "energy"],
        "CL=F": ["crude oil", "WTI", "oil futures", "OPEC"],
        "SXRV.DE": ["NASDAQ", "QQQ", "tech stocks", "Fed", "earnings"],
        "^NDX": ["NASDAQ 100", "NDX", "tech stocks", "Fed"],
        "SXRV.FRK": ["NASDAQ", "QQQ", "tech stocks", "Fed"],
    }

    queries = topics_map.get(ticker, [ticker])[
        :2
    ]  # Max 2 queries to respect rate limits

    all_headlines = []
    total_sentiment = 0
    sentiment_count = 0

    for query in queries:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&keywords={query}&apikey={api_key}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "Information" in data and "limit" in data.get("Information", ""):
                logger.warning("Limite Alpha Vantage atteinte, arret des requetes.")
                break

            for item in data.get("feed", []):
                title = item.get("title", "")
                if title and title not in all_headlines:
                    all_headlines.append(title)
                for sentiment in item.get("ticker_sentiment", []):
                    try:
                        total_sentiment += float(
                            sentiment.get("ticker_sentiment_score", 0)
                        )
                        sentiment_count += 1
                    except (ValueError, TypeError):
                        continue

            time.sleep(12)  # Respect Alpha Vantage free tier rate limit (5/min)

        except Exception:
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
        # Aggregating news from multiple financial sources
        trends = tools.get_unified_trends(sources=["cls", "weibo", "wallstreetcn"])

        # Simple sentiment heuristic for AlphaEar (can be enhanced with another LLM call)
        # For now, we return the headlines and a neutral/positive default if trends exist
        headlines = [t.get("title", "") for t in trends if t.get("title")]
        return headlines, 0.1 if headlines else 0
    except Exception:
        return [], 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch news headlines and sentiment.")
    parser.add_argument("ticker", type=str, help="The stock ticker to fetch news for.")
    parser.add_argument("api_key", type=str, help="Your Alpha Vantage API key.")
    args = parser.parse_args()

    # Fetch from both sources
    av_headlines, av_sentiment = fetch_alpha_vantage_news(args.ticker, args.api_key)
    ae_headlines, ae_sentiment = fetch_alpha_ear_trends()

    # Merge results
    all_headlines = av_headlines + ae_headlines
    # Weighted average: Alpha Vantage (specific to ticker) vs AlphaEar (general trends)
    final_sentiment = (av_sentiment * 0.7) + (ae_sentiment * 0.3)

    output = {
        "headlines": all_headlines[:20],  # Limit to top 20 for LLM context
        "sentiment": final_sentiment,
    }

    print(json.dumps(output))
