import asyncio
import cognee
from cognee.tasks.web_scraper.config import SoupCrawlerConfig


async def test_web_scraping_using_bs4():
    # 0. Prune only data (not full system prune)
    await cognee.prune.prune_data()

    # 1. Setup test URL and extraction rules
    url = "https://quotes.toscrape.com/"
    rules = {
        "quotes": {"selector": ".quote span.text", "all": True},
        "authors": {"selector": ".quote small", "all": True},
    }

    soup_config = SoupCrawlerConfig(
        concurrency=5,
        crawl_delay=0.5,
        timeout=15.0,
        max_retries=2,
        retry_delay_factor=0.5,
        extraction_rules=rules,
        use_playwright=False,
        structured=True,
    )

    # 2. Add / ingest the page
    await cognee.add(
        data=url,
        soup_crawler_config=soup_config,
        incremental_loading=False,
    )

    # 3. Cognify
    await cognee.cognify()

    # 4. Search for a known quote
    results = await cognee.search(
        "Who said 'The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking'?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
    )
    assert "Albert Einstein" in results[0], (
        "Test failed! Albert Einstein not found in scraped data."
    )
    print("Test passed! Found Albert Einstein in scraped data.")
    print(results)
    print("Web scraping test using bs4 completed.")


async def test_web_scraping_using_bs4_and_incremental_loading():
    # 0. Prune only data (not full system prune)
    await cognee.prune.prune_data()

    # 1. Setup test URL and extraction rules
    url = "https://books.toscrape.com/"
    rules = {"titles": "article.product_pod h3 a", "prices": "article.product_pod p.price_color"}

    soup_config = SoupCrawlerConfig(
        concurrency=1,
        crawl_delay=0.1,
        timeout=10.0,
        max_retries=1,
        retry_delay_factor=0.5,
        extraction_rules=rules,
        use_playwright=False,
        structured=True,
    )

    # 2. Add / ingest the page
    await cognee.add(
        data=url,
        soup_crawler_config=soup_config,
        incremental_loading=True,
    )

    # 3. Cognify
    await cognee.cognify()

    # 4. Search for a known book
    results = await cognee.search(
        "What is the price of 'A Light in the Attic' book?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
    )
    assert "51.77" in results[0], "Test failed! 'A Light in the Attic' not found in scraped data."
    print("Test passed! Found 'A Light in the Attic' in scraped data.")
    print(results)
    print("Web scraping test using bs4 with incremental loading completed.")


async def test_web_scraping_using_tavily():
    # 0. Prune only data (not full system prune)
    await cognee.prune.prune_data()

    # 1. Setup test URL and extraction rules
    url = "https://quotes.toscrape.com/"

    # 2. Add / ingest the page
    await cognee.add(
        data=url,
        incremental_loading=False,
    )

    # 3. Cognify
    await cognee.cognify()

    # 4. Search for a known quote
    results = await cognee.search(
        "Who said 'The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking'?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
    )
    assert "Albert Einstein" in results[0], (
        "Test failed! Albert Einstein not found in scraped data."
    )
    print("Test passed! Found Albert Einstein in scraped data.")
    print(results)
    print("Web scraping test using tavily completed.")


async def test_web_scraping_using_tavily_and_incremental_loading():
    # 0. Prune only data (not full system prune)
    await cognee.prune.prune_data()

    # 1. Setup test URL and extraction rules
    url = "https://quotes.toscrape.com/"

    # 2. Add / ingest the page
    await cognee.add(
        data=url,
        incremental_loading=True,
    )

    # 3. Cognify
    await cognee.cognify()

    # 4. Search for a known quote
    results = await cognee.search(
        "Who said 'The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking'?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
    )
    assert "Albert Einstein" in results[0], (
        "Test failed! Albert Einstein not found in scraped data."
    )
    print("Test passed! Found Albert Einstein in scraped data.")
    print(results)
    print("Web scraping test using tavily with incremental loading completed.")


async def main():
    print("starting web scraping test using bs4 with incremental loading...")
    await test_web_scraping_using_bs4_and_incremental_loading()
    print("starting web scraping test using bs4 without incremental loading...")
    await test_web_scraping_using_bs4()
    print("starting web scraping test using tavily with incremental loading...")
    await test_web_scraping_using_tavily_and_incremental_loading()
    print("starting web scraping test using tavily without incremental loading...")
    await test_web_scraping_using_tavily()


if __name__ == "__main__":
    asyncio.run(main())
