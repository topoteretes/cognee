from cognee.modules.ingestion import save_data_to_file
from cognee.tasks.ingestion.data_fetchers.data_fetcher_interface import DataFetcherInterface
from typing import Any
from cognee.tasks.web_scraper.config import TavilyConfig, SoupCrawlerConfig
from cognee.modules.ingestion.exceptions.exceptions import IngestionError
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class WebUrlFetcher(DataFetcherInterface):
    def __init__(self): ...

    def fetcher_name(self):
        return "web_url_fetcher"

    async def fetch(self, data_item_path: str, fetchers_config: dict[str, Any]):
        from cognee.context_global_variables import tavily_config, soup_crawler_config
        from cognee.tasks.web_scraper import fetch_page_content

        web_url_fetcher_config = fetchers_config.get(self.fetcher_name())
        if not isinstance(web_url_fetcher_config, dict):
            raise IngestionError(f"{self.fetcher_name()} configuration must be a valid dictionary")

        tavily_dict = web_url_fetcher_config.get("tavily_config")
        _tavily_config = TavilyConfig(**tavily_dict) if tavily_dict else None

        soup_dict = web_url_fetcher_config.get("soup_config")
        _soup_config = SoupCrawlerConfig(**soup_dict) if soup_dict else None

        # Set global configs for downstream access
        tavily_config.set(_tavily_config)
        soup_crawler_config.set(_soup_config)

        preferred_tool = "beautifulsoup" if _soup_config else "tavily"
        if preferred_tool == "tavily" and _tavily_config is None:
            raise IngestionError(
                message="TavilyConfig must be set on the ingestion context when fetching HTTP URLs without a SoupCrawlerConfig."
            )
        if preferred_tool == "beautifulsoup" and _soup_config is None:
            raise IngestionError(
                message="SoupCrawlerConfig must be set on the ingestion context when using the BeautifulSoup scraper."
            )

        logger.info(f"Starting web URL crawling for: {data_item_path}")
        logger.info(f"Using scraping tool: {preferred_tool}")

        data = await fetch_page_content(
            data_item_path,
            preferred_tool=preferred_tool,
            soup_crawler_config=_soup_config,
            tavily_config=_tavily_config,
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
