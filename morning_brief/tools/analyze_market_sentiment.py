import re
from collections import Counter

from smolagents import Tool

from morning_brief.tools.rss_helpers import COMMON_FEEDS


def _make_word_pattern(word):
    return re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)


class AnalyzeMarketSentimentTool(Tool):
    name = "analyze_market_sentiment"
    description = (
        "Macro market sentiment from RSS feeds and Alpha Vantage NEWS_SENTIMENT. "
        "Returns a compact summary with sentiment score, macro signals, and top themes. "
        "Full data saved to output/tools/."
    )
    inputs = {}
    output_type = "string"

    _POSITIVE_WORDS = [
        "rally", "surge", "gain", "rise", "jump", "climb", "soar", "boom",
        "growth", "recovery", "beat", "exceed", "strong", "bullish", "upgrade",
        "positive", "optimistic", "expansion", "profit", "record high",
        "breakthrough", "deal", "agreement", "stimulus",
    ]
    _NEGATIVE_WORDS = [
        "crash", "plunge", "drop", "fall", "slump", "decline", "recession",
        "crisis", "bearish", "downgrade", "loss", "cut", "slashing", "weak",
        "fear", "risk", "warning", "threat", "sanction", "tariff", "deficit",
        "inflation surge", "rate hike", "tightening", "default", "collapse",
        "pandemic", "war", "conflict", "attack",
    ]
    _MACRO_KEYWORDS = {
        "fed": ["fed", "federal reserve", "fomc", "interest rate", "rate hike", "rate cut"],
        "cpi": ["cpi", "inflation", "consumer price", "pce", "core inflation"],
        "employment": ["employment", "unemployment", "jobless", "payroll", "nonfarm"],
        "m2": ["m2", "money supply", "m1", "liquidity"],
    }
    _RSS_FEEDS = COMMON_FEEDS + [
        ("Google News Macro", "https://news.google.com/rss/search?q=macroeconomics+M2+Fed+CPI&hl=en-US&gl=US&ceid=US:en"),
    ]

    def _fetch_rss_headlines(self):
        from morning_brief.tools.rss_helpers import fetch_rss_entries

        headlines = []
        all_text = []
        for _source, url in self._RSS_FEEDS:
            for entry in fetch_rss_entries(url):
                title = entry.get("title", "").strip()
                if title:
                    headlines.append(title)
                    all_text.append(title.lower())
        return headlines, all_text

    def _fetch_alpha_vantage(self, headlines, all_text):
        import os
        import time
        import requests

        for query in ["crude oil", "Fed rate"]:
            try:
                api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
                if not api_key:
                    break
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&keywords={query}&apikey={api_key}"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if "Information" in data and "limit" in data.get("Information", ""):
                    break
                for item in data.get("feed", [])[:3]:
                    title = item.get("title", "")
                    if title:
                        headlines.append(title)
                        all_text.append(title.lower())
                time.sleep(12)
            except Exception:
                continue

    def _compute_sentiment(self, all_text):
        pos_patterns = [_make_word_pattern(w) for w in self._POSITIVE_WORDS]
        neg_patterns = [_make_word_pattern(w) for w in self._NEGATIVE_WORDS]
        pos = sum(1 for t in all_text for p in pos_patterns if p.search(t))
        neg = sum(1 for t in all_text for p in neg_patterns if p.search(t))
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0.0

    def _compute_macro_signals(self, all_text):
        signals = {}
        for key, keywords in self._MACRO_KEYWORDS.items():
            patterns = [_make_word_pattern(kw) for kw in keywords]
            mentions = sum(1 for t in all_text for p in patterns if p.search(t))
            signals[key] = "ACTIVE" if mentions > 0 else "QUIET"
        return signals

    def _extract_themes(self, all_text):
        words = []
        for t in all_text:
            tokens = re.findall(r"\b[a-z]{4,}\b", t)
            words.extend(tokens)
        stop = {"with", "from", "this", "that", "which", "their", "about", "would", "could",
                "should", "other", "than", "after", "before", "between", "through", "during"}
        filtered = [w for w in words if w not in stop]
        return [w for w, _ in Counter(filtered).most_common(5)]

    def forward(self) -> str:
        from morning_brief.tools import save_tool_result

        headlines, all_text = self._fetch_rss_headlines()
        self._fetch_alpha_vantage(headlines, all_text)
        score = self._compute_sentiment(all_text)
        macro_signals = self._compute_macro_signals(all_text)
        themes = self._extract_themes(all_text)

        full = {
            "headline_count": len(headlines),
            "headlines": headlines[:15],
            "sentiment_score": round(score, 2),
            "macro_signals": macro_signals,
            "key_themes": themes,
        }
        save_tool_result("sentiment", full)

        active = [k for k, v in macro_signals.items() if v == "ACTIVE"]
        return (
            f"Sentiment={score:+.2f} | {len(headlines)} headlines | "
            f"Macro: {', '.join(active) if active else 'quiet'} | Themes: {', '.join(themes[:3])}"
        )
