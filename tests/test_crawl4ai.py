import asyncio
from crawl4ai import AsyncWebCrawler


async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://www.google.com")
        if result.success:
            print("Crawl4AI Success!")
            print(f"Content length: {len(result.markdown)}")
        else:
            print(f"Crawl4AI Failed: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
