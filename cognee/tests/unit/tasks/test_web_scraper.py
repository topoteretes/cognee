import os
from unittest.mock import patch
from cognee.tasks.web_scraper.config import DefaultCrawlerConfig, TavilyConfig


def test_web_scraper_config_env_vars():
    """Verify that web scraper configuration is read dynamically from environment variables."""
    test_env = {
        "WEB_SCRAPER_TIMEOUT": "25.5",
        "WEB_SCRAPER_CRAWL_DELAY": "2.5",
        "WEB_SCRAPER_MAX_DELAY": "30.0",
        "WEB_SCRAPER_CONCURRENCY": "12",
        "TAVILY_TIMEOUT": "20",
    }

    with patch.dict(os.environ, test_env):
        crawler_config = DefaultCrawlerConfig()
        tavily_config = TavilyConfig()

        assert crawler_config.timeout == 25.5
        assert crawler_config.crawl_delay == 2.5
        assert crawler_config.max_crawl_delay == 30.0
        assert crawler_config.concurrency == 12
        assert tavily_config.timeout == 20
