import asyncio
from ddgs import DDGS
import logging
from dotenv import load_dotenv
from typing import List, Dict
from llm_client import TEXT_LLM_MODEL, _query_ollama

try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    # Handle if not strictly available or loading failure
    AsyncWebCrawler = None

logger = logging.getLogger(__name__)

# Load env variables to get the brave api key
load_dotenv()


def generate_search_query(ticker: str) -> str:
    """
    Uses the LLM to generate the most relevant web search query for a given ticker.
    """
    logger.info(f"Generating dynamic web search query for {ticker}...")

    # Specific instruction for Hyperliquid on Oil and NASDAQ
    extra_context = ""
    if any(x in ticker.upper() for x in ["CL=F", "OIL", "WTI", "CRUDP"]):
        extra_context = "Specifically check for 'flx:OIL' or 'OIL-USDH' price action and sentiment on Hyperliquid (DEX) as a leading indicator."
    elif any(x in ticker.upper() for x in ["^NDX", "NASDAQ", "QQQ", "SXRV"]):
        extra_context = "Specifically check for 'NDX' or tech index price action and sentiment on Hyperliquid (DEX) as a leading indicator for the US tech opening."

    prompt = f"""
    You are an expert macroeconomic analyst. I am preparing to analyze the financial asset '{ticker}'.
    To gather the most relevant long-term context, I need to search the web for recent macroeconomic forecasts,
    fundamental events, or policy decisions that directly impact this specific asset.
    {extra_context}

    Generate the single most effective web search query (in English, maximum 10 words) to find this information.
    For example, if the asset is a tech index, you might search for Federal Reserve rates or semiconductor supply chain.
    If it's oil, you might search for OPEC+ decisions and global demand.

    Provide your response ONLY as a valid JSON object:
    {{
      "query": "<your optimized search query>"
    }}
    """

    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt.strip(),
        "stream": False,
        "format": "json",
        "system": "You are a web research assistant. Output only a valid JSON object."
    }

    try:
        response = _query_ollama(payload, expected_keys=["query"])
        query = response.get("query")
        # Handle cases where _query_ollama fallback was triggered with default keys (query would hold a string like error message if fallback triggered without expected_keys, but now we pass expected_keys)
        if query and not str(query).startswith("LLM") and not str(query).startswith("Unexpected"):
            logger.info(f"Generated search query: '{query}'")
            return query
    except Exception as e:
        logger.error(f"Failed to generate search query: {e}")

    # Fallback
    fallback_query = f"Macroeconomic forecast and market analysis for {ticker}"
    logger.warning(f"Using fallback search query: '{fallback_query}'")
    return fallback_query

def _sync_search_ddg(query: str, count: int) -> List[Dict[str, str]]:

    results_list = []
    try:
        with DDGS() as ddgs:
            # text(query, max_results=count) returns a generator of dicts
            results = list(ddgs.text(query, max_results=count))
            for item in results:
                if item.get("href"):
                    results_list.append({
                        "url": item.get("href"),
                        "title": item.get("title", ""),
                        "body": item.get("body", "") # DDG already provides a small snippet
                    })
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
            pages_content.append({
                "url": res['url'],
                "content": f"Title: {res['title']}\nSnippet: {res['body']}"
            })
        return pages_content

    try:
        async with AsyncWebCrawler() as crawler:
            for res in search_results:
                url = res['url']
                logger.info(f"Extraction de : {url}...")
                try:
                    # Set a timeout for crawling
                    result = await asyncio.wait_for(crawler.arun(url=url), timeout=30.0)
                    if result.success:
                        content = getattr(result, "markdown_links_removed", getattr(result, "markdown_fit", ""))
                        if not content:
                            content = result.markdown
                        
                        pages_content.append({
                            "url": url,
                            "content": content
                        })
                    else:
                        logger.warning(f"Failed to crawl {url}. Using snippet.")
                        pages_content.append({
                            "url": url,
                            "content": f"Title: {res['title']}\nSnippet: {res['body']}"
                        })
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}. Using snippet.")
                    pages_content.append({
                        "url": url,
                        "content": f"Title: {res['title']}\nSnippet: {res['body']}"
                    })
    except Exception as e:
        logger.error(f"Failed to initialize crawler: {e}. Falling back to snippets.")
        for res in search_results:
            pages_content.append({
                "url": res['url'],
                "content": f"Title: {res['title']}\nSnippet: {res['body']}"
            })

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

    context_parts = []
    for page in data:
        # Limit to 1200 chars per page to avoid context overflow
        content = page.get('content', '')
        snippet = content[:1200] if content else "No content available."
        context_parts.append(f"Source: {page['url']}\nContent Excerpt:\n{snippet}...\n")

    return "\n---\n".join(context_parts)

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