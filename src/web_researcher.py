import asyncio
from ddgs import DDGS
import hashlib
import json
import logging
import math
import re
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
from llm_client import TEXT_LLM_MODEL, _query_ollama, SCHEMA_SEARCH_QUERY

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
except ImportError:
    AsyncWebCrawler = None


logger = logging.getLogger(__name__)

# --- Search query cache (24h) ---
SEARCH_QUERY_CACHE_DIR = Path("data_cache") / "search_queries"
SEARCH_QUERY_CACHE_TTL_HOURS = 24

# Strict ticker validation: allow letters, digits, ., -, ^, = (e.g. CL=F, ^NDX, BRK.B)
# Reject path separators, dots followed by path segments, etc.
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-\^=]{1,16}$")

# Patterns that indicate an LLM-generated query is invalid (also applied to cached values).
_INVALID_QUERY_PATTERNS = ["execution failed", "error", "unexpected", "llm", "failed"]


def _validate_ticker(ticker: str) -> str:
    """Validates and returns a safe ticker. Raises ValueError on invalid input."""
    if not isinstance(ticker, str) or not _TICKER_RE.match(ticker):
        raise ValueError(f"Invalid ticker (path traversal risk): {ticker!r}")
    return ticker


def get_fallback_search_query(ticker: str) -> str:
    """Canonical fallback query — single source of truth, used by all callers."""
    return f"Macroeconomic forecast and market analysis for {ticker}"


def _data_signature(latest_data: pd.DataFrame | None) -> str:
    """Short signature of recent price action to invalidate cache on regime change.

    Returns an empty string when no data is provided (cache key degrades to ticker+date).
    Buckets close into ~2% log bands and RSI into 10-unit bands — small noise won't bust
    the cache, but a 10% gap or volatility regime change will.
    """
    if latest_data is None or latest_data.empty:
        return ""
    try:
        last = latest_data.iloc[-1]
        close = float(last["Close"])
        rsi = float(last.get("RSI", 50))
        # Log2 bucketing: each bucket is ~1.4% wide (2^(1/50) ≈ 1.014).
        # close=100 -> bucket=332, close=102 -> bucket=333, close=120 -> bucket=343
        close_bucket = int(math.log2(max(close, 1e-6)) * 50)
        rsi_bucket = int(rsi // 10)
        sig = f"c{close_bucket}_r{rsi_bucket}"
        # Tiny hash to keep filename short
        return hashlib.md5(sig.encode("utf-8")).hexdigest()[:8]
    except Exception:
        return ""


def _search_query_cache_path(ticker: str, data_sig: str = "") -> Path:
    """Cache filename — includes data signature so regime changes invalidate."""
    _validate_ticker(ticker)
    safe_ticker = re.sub(r"[^A-Za-z0-9.\-]", "_", ticker)
    today = datetime.now().strftime("%Y-%m-%d")
    if data_sig:
        return SEARCH_QUERY_CACHE_DIR / f"{safe_ticker}_{today}_{data_sig}.json"
    return SEARCH_QUERY_CACHE_DIR / f"{safe_ticker}_{today}.json"


def _is_query_valid(query) -> bool:
    """Re-applies the invalid_patterns filter on any candidate query (cached or fresh)."""
    if not query or not isinstance(query, str):
        return False
    q_lower = query.lower()
    return not any(p in q_lower for p in _INVALID_QUERY_PATTERNS)


def _load_cached_search_query(ticker: str, data_sig: str = "") -> str | None:
    """Returns the cached search query if it exists, is fresh, and passes sanitization."""
    cache_file = _search_query_cache_path(ticker, data_sig)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached_at = datetime.fromisoformat(data["cached_at"])
        age = datetime.now() - cached_at
        if age < timedelta(hours=SEARCH_QUERY_CACHE_TTL_HOURS):
            query = data.get("query")
            if _is_query_valid(query):
                logger.info(
                    f"Using cached search query for {ticker} (age={age.total_seconds()/3600:.1f}h): '{query}'"
                )
                return query
            else:
                logger.warning(
                    f"Cached search query for {ticker} failed sanitization, treating as cache miss."
                )
        else:
            logger.info(f"Search query cache for {ticker} expired (age={age.total_seconds()/3600:.1f}h).")
    except Exception as e:
        logger.warning(f"Failed to read search query cache for {ticker}: {e}")
    return None


def _save_cached_search_query(ticker: str, query: str, data_sig: str = "") -> None:
    """Persists a *successful* LLM-generated search query to disk.

    IMPORTANT: must never be called with the fallback query — see review Finding #4.
    """
    cache_file = _search_query_cache_path(ticker, data_sig)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"cached_at": datetime.now().isoformat(), "query": query}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save search query cache for {ticker}: {e}")


class _GrokipediaFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage().lower()
        if "grokipedia" in msg:
            return False
        return True


logging.getLogger("crawl4ai").addFilter(_GrokipediaFilter())

# Load env variables to get the brave api key
load_dotenv()


def generate_search_query(ticker: str, latest_data: pd.DataFrame = None, use_cache: bool = True) -> str:
    """
    Uses the LLM to generate the most relevant web search query for a given ticker.
    Integrates current date and recent price action for temporal relevance.

    Cache strategy (24h TTL, keyed by ticker + date + price-bucket signature):
    - Successful LLM-generated queries are cached.
    - Fallback queries are NEVER cached — see review Finding #4.
    - Cache keys include a price-action signature so a regime change invalidates.
    - Set ``use_cache=False`` to force regeneration (e.g. for tests).
    """
    _validate_ticker(ticker)
    data_sig = _data_signature(latest_data)

    if use_cache:
        cached = _load_cached_search_query(ticker, data_sig)
        if cached:
            return cached

    current_date = datetime.now().strftime("%B %Y")
    logger.info(f"Generating dynamic web search query for {ticker} ({current_date})...")

    price_context = ""
    if latest_data is not None and not latest_data.empty:
        last_price = latest_data["Close"].iloc[-1]
        prev_price = latest_data["Close"].iloc[-5] if len(latest_data) > 5 else latest_data["Close"].iloc[0]
        change = ((last_price / prev_price) - 1) * 100
        trend = "upward" if change > 0.5 else "downward" if change < -0.5 else "sideways"
        price_context = f"The current price is {last_price:.2f} with a 5-day {trend} trend ({change:+.2f}%)."

    # Specific instruction for Hyperliquid on Oil and NASDAQ
    extra_context = ""
    if any(x in ticker.upper() for x in ["CL=F", "OIL", "WTI", "CRUDP"]):
        extra_context = "Specifically focus on OPEC+ supply decisions, global inventory levels, and 'flx:OIL' sentiment on Hyperliquid."
    elif any(x in ticker.upper() for x in ["^NDX", "NASDAQ", "QQQ", "SXRV"]):
        extra_context = "Specifically focus on Federal Reserve interest rate expectations, major tech earnings, and NDX speculative positioning."

    prompt = f"""
    You are an expert macroeconomic research assistant. Today is {current_date}.
    Target Asset: {ticker}
    Current Context: {price_context}
    {extra_context}

    Your goal is to find the most impactful news or reports FROM THE LAST 30 DAYS that explain the current market regime.
    Generate the single most effective Google/DuckDuckGo search query (maximum 10 words).

    Output ONLY a valid JSON object:
    {{
      "query": "<your optimized search query>"
    }}
    """

    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt.strip(),
        "stream": False,
        "format": SCHEMA_SEARCH_QUERY,
        "options": {"temperature": 0.4, "num_predict": 512},
        "system": "<|think|> You are a professional financial researcher. Be precise and focus on current market catalysts. Output ONLY the requested JSON object — never add a 'thought' key.",
    }

    try:
        response = _query_ollama(payload, expected_keys=["query"])
        query = response.get("query")
        if _is_query_valid(query):
            logger.info(f"Generated search query: '{query}'")
            if use_cache:
                _save_cached_search_query(ticker, query, data_sig)
            return query
        logger.warning(f"Requete de recherche invalide ignoree: '{query}'")
    except Exception as e:
        logger.error(f"Failed to generate search query: {e}")

    # Fallback — NEVER cached (would poison 24h after a single LLM failure).
    fallback_query = get_fallback_search_query(ticker)
    logger.warning(f"Using fallback search query (not cached): '{fallback_query}'")
    return fallback_query


def _sync_search_ddg(query: str, count: int) -> List[Dict[str, str]]:

    results_list = []
    try:
        with DDGS() as ddgs:
            # text(query, max_results=count) returns a generator of dicts
            results = list(ddgs.text(query, max_results=count))
            for item in results:
                if item.get("href"):
                    results_list.append(
                        {
                            "url": item.get("href"),
                            "title": item.get("title", ""),
                            "body": item.get("body", ""),  # DDG already provides a small snippet
                        }
                    )
    except Exception as e:
        logger.error(f"Error during DuckDuckGo sync search: {e}")
    return results_list


async def search_ddg(query: str, count: int = 3) -> List[Dict[str, str]]:
    """Effectue une recherche via DuckDuckGo Search et retourne une liste de dicts (url, title, body)."""
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _sync_search_ddg, query, count)
        return results
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search: {e}")
        return []


async def fetch_and_clean(search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Tente de crawler les URLs, sinon utilise le snippet 'body' de DDG."""
    pages_content = []

    if AsyncWebCrawler is None:
        logger.warning("Crawl4AI is not installed. Using DDG snippets only.")
        for res in search_results:
            pages_content.append(
                {
                    "url": res["url"],
                    "content": f"Title: {res['title']}\nSnippet: {res['body']}",
                }
            )
        return pages_content


    try:
        browser_config = BrowserConfig(verbose=True)
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.45, threshold_type="dynamic", min_word_threshold=50)
            )
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for res in search_results:
                url = res["url"]
                logger.info(f"Extraction de : {url}...")
                try:
                    # Set a timeout for crawling
                    result = await asyncio.wait_for(crawler.arun(url=url, config=run_config), timeout=30.0)
                    if result.success:
                        content = result.markdown.fit_markdown if result.markdown else ""
                        if not content:
                            content = result.markdown.raw_markdown if result.markdown else "No content extracted"

                        pages_content.append({"url": url, "content": content})
                    else:
                        logger.warning(f"Failed to crawl {url}. Using snippet.")
                        pages_content.append({"url": url, "content": f"Title: {res['title']} - Snippet: {res['body']}"})
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}. Using snippet.")
                    pages_content.append({"url": url, "content": f"Title: {res['title']} - Snippet: {res['body']}"})
    except Exception as e:
        logger.error(f"Failed to initialize crawler: {e}. Falling back to snippets.")
        for res in search_results:
            pages_content.append({"url": res["url"], "content": f"Title: {res['title']} - Snippet: {res['body']}"})
    return pages_content


async def get_web_research_context_async(query: str) -> str:
    """
    Exécute la recherche et le crawling asynchrone et formate le résultat en Markdown.
    """
    logger.info(f"Starting web research for query: '{query}'")
    search_results = await search_ddg(query)

    if not search_results:
        logger.info("No results found during web research.")
        return ""

    data = await fetch_and_clean(search_results)

    # Limit to 1200 chars per page to avoid context overflow
    return "\n---\n".join(
        [
            f"Source: {page['url']}\nContent Excerpt:\n{c[:1200] if (c := page.get('content')) else 'No content available.'}...\n"
            for page in data
        ]
    )


def get_web_context_sync(query: str) -> str:
    """
    Wrapper synchrone pour faciliter l'intégration dans le code existant.
    """
    try:
        return asyncio.run(get_web_research_context_async(query))
    except Exception as e:
        logger.error(f"Sync wrapper failed for web research: {e}")
        return ""


if __name__ == "__main__":
    # Test script if run directly
    logging.basicConfig(level=logging.INFO)
    test_query = "Macroeconomic forecast 2024"
    context = get_web_context_sync(test_query)
    print("\n--- TEST CONTEXT RESULT ---")
    print(context[:1000] + "..." if context else "No context fetched.")
