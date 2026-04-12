import asyncio
import httpx
import logging
import os
from dotenv import load_dotenv
from typing import List, Dict

try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    # Handle if not strictly available or loading failure
    AsyncWebCrawler = None

logger = logging.getLogger(__name__)

# Load env variables to get the brave api key
load_dotenv()

async def search_brave(query: str, count: int = 3) -> List[str]:
    """Effectue une recherche via l'API Brave et retourne une liste d'URLs."""
    brave_api_key = os.getenv("BRAVE_API_KEY")
    if not brave_api_key:
        logger.warning("BRAVE_API_KEY not found in environment variables. Web research will be skipped.")
        return []

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key
    }
    url = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": count}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=10.0)
            response.raise_for_status()
            results = response.json()
            # Extraire les URLs des résultats web
            return [item["url"] for item in results.get("web", {}).get("results", [])]
    except Exception as e:
        logger.error(f"Error during Brave search: {e}")
        return []

async def fetch_and_clean(urls: List[str]) -> List[Dict[str, str]]:
    """Utilise Crawl4AI pour récupérer le contenu de chaque page en Markdown."""
    if AsyncWebCrawler is None:
        logger.error("Crawl4AI is not installed or available.")
        return []

    pages_content = []

    try:
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                logger.info(f"Extraction de : {url}...")
                try:
                    result = await crawler.arun(url=url)
                    if result.success:
                        # On ne garde que le Markdown "fit" (nettoyé du bruit)
                        # Depending on Crawl4AI version, the attribute might be 'markdown_fit' or 'markdown_links_removed'
                        content = getattr(result, "markdown_links_removed", getattr(result, "markdown_fit", ""))
                        if not content:
                            content = result.markdown # Fallback

                        pages_content.append({
                            "url": url,
                            "content": content
                        })
                    else:
                        logger.warning(f"Failed to crawl {url}: {getattr(result, 'error_message', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize crawler: {e}")

    return pages_content

async def get_web_research_context_async(query: str) -> str:
    """
    Exécute la recherche et le crawling asynchrone et formate le résultat en Markdown.
    """
    logger.info(f"Starting web research for query: '{query}'")
    urls = await search_brave(query)

    if not urls:
        logger.info("No URLs found during web research.")
        return ""

    data = await fetch_and_clean(urls)

    if not data:
        logger.info("Failed to extract content from URLs.")
        return ""

    context_parts = []
    for page in data:
        # Limit to 1500 chars per page to avoid exploding the context window
        snippet = page['content'][:1500]
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
