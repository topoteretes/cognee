import os
from cognee.modules.ingestion import save_data_to_file
from cognee.tasks.ingestion.data_fetchers.data_fetcher_interface import DataFetcherInterface
from cognee.tasks.web_scraper.config import TavilyConfig, SoupCrawlerConfig
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class WebUrlFetcher(DataFetcherInterface):
    def __init__(self): ...

    def fetcher_name(self):
        return "web_url_fetcher"

    async def fetch(self, data_item_path: str):
        from cognee.context_global_variables import tavily_config, soup_crawler_config
        from cognee.tasks.web_scraper import fetch_page_content

        if os.getenv("TAVILY_API_KEY"):
            _tavily_config = TavilyConfig()
            _soup_config = None
            preferred_tool = "tavily"
        else:
            _tavily_config = None
            _soup_config = SoupCrawlerConfig()
            preferred_tool = "beautifulsoup"

        tavily_config.set(_tavily_config)
        soup_crawler_config.set(_soup_config)

        logger.info(f"Starting web URL crawling for: {data_item_path}")
        logger.info(f"Using scraping tool: {preferred_tool}")

        data = await fetch_page_content(
            data_item_path,
            preferred_tool=preferred_tool,
        )

        logger.info(f"Successfully fetched content from URL {data_item_path}")

        # fetch_page_content returns a dict like {url: content}
        # Extract the content string before saving
        if isinstance(data, dict):
            # Concatenate all URL contents (usually just one URL)
            content = ""
            for url, text in data.items():
                content += f"{url}:\n{text}\n\n"
            logger.info(
                f"Extracted content from {len(data)} URL(s), total size: {len(content)} characters"
            )
        else:
            content = data

        return await save_data_to_file(content)
