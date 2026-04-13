import requests
import argparse
import json
import sys
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Add AlphaEar skill scripts to path
# Configurable via environment variable, defaults to relative path from project root
ALPHA_EAR_PATH = Path(os.getenv(
    "ALPHA_EAR_SCRIPTS_PATH",
    str(Path(__file__).parent.parent / ".agents" / "skills" / "alphaear-news" / "scripts")
))
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
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        headlines = [item['title'] for item in data.get('feed', [])]

        sentiment_score = 0
        sentiment_count = 0
        for item in data.get('feed', []):
            for sentiment in item.get('ticker_sentiment', []):
                if sentiment['ticker'] == ticker:
                    try:
                        sentiment_score += float(sentiment['ticker_sentiment_score'])
                        sentiment_count += 1
                    except (ValueError, TypeError):
                        continue

        overall_sentiment = sentiment_score / sentiment_count if sentiment_count > 0 else 0
        return headlines, overall_sentiment

    except Exception:
        return [], 0

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
        headlines = [t.get('title', '') for t in trends if t.get('title')]
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
        "headlines": all_headlines[:20], # Limit to top 20 for LLM context
        "sentiment": final_sentiment
    }

    print(json.dumps(output))