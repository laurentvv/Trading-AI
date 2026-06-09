import feedparser
import requests

COMMON_FEEDS = [
    ("Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss"),
    ("EIA Today in Energy", "https://www.eia.gov/rss/todayinenergy.xml"),
    ("EIA Weekly Petroleum", "https://www.eia.gov/petroleum/weekly/includes/week_in_petroleum_rss.xml"),
]


def fetch_rss_entries(url, max_entries=5):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        return feed.entries[:max_entries]
    except Exception:
        return []
