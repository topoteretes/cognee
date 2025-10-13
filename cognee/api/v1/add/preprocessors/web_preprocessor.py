"""
Web preprocessor for handling URL inputs in the cognee add function.

This preprocessor handles web URLs by setting up appropriate crawling configurations
and modifying the processing context for web content.
"""

import os
from urllib.parse import urlparse
from typing import Union, BinaryIO

from .base import Preprocessor, PreprocessorContext, PreprocessorResult

try:
    from cognee.tasks.web_scraper.config import TavilyConfig, SoupCrawlerConfig
    from cognee.context_global_variables import (
        tavily_config as tavily,
        soup_crawler_config as soup_crawler,
    )

    WEB_SCRAPER_AVAILABLE = True
except ImportError:
    WEB_SCRAPER_AVAILABLE = False


class WebPreprocessor(Preprocessor):
    """Preprocessor for handling web URL inputs."""

    @property
    def name(self) -> str:
        return "web_preprocessor"

    def _is_http_url(self, item: Union[str, BinaryIO]) -> bool:
        """Check if an item is an HTTP/HTTPS URL."""
        http_schemes = {"http", "https"}
        return isinstance(item, str) and urlparse(item).scheme in http_schemes

    def can_handle(self, context: PreprocessorContext) -> bool:
        """Check if this preprocessor can handle the given context."""
        if not WEB_SCRAPER_AVAILABLE:
            return False

        if self._is_http_url(context.data):
            return True

        if isinstance(context.data, list):
            return any(self._is_http_url(item) for item in context.data)

        return False

    async def process(self, context: PreprocessorContext) -> PreprocessorResult:
        """Process web URLs by setting up crawling configurations."""
        try:
            extraction_rules = context.extra_params.get("extraction_rules")
            tavily_config_param = context.extra_params.get("tavily_config")
            soup_crawler_config_param = context.extra_params.get("soup_crawler_config")

            if not soup_crawler_config_param and extraction_rules:
                soup_crawler_config_param = SoupCrawlerConfig(extraction_rules=extraction_rules)

            if not tavily_config_param and os.getenv("TAVILY_API_KEY"):
                tavily_config_param = TavilyConfig(api_key=os.getenv("TAVILY_API_KEY"))

            if soup_crawler_config_param:
                soup_crawler.set(soup_crawler_config_param)

                tavily.set(tavily_config_param)

            modified_context = context.model_copy()

            if self._is_http_url(context.data):
                modified_context.node_set = (
                    ["web_content"] if not context.node_set else context.node_set + ["web_content"]
                )
            elif isinstance(context.data, list) and any(
                self._is_http_url(item) for item in context.data
            ):
                modified_context.node_set = (
                    ["web_content"] if not context.node_set else context.node_set + ["web_content"]
                )

            return PreprocessorResult(modified_context=modified_context)

        except Exception as e:
            return PreprocessorResult(error=f"Failed to configure web scraping: {str(e)}")


def register_web_preprocessor() -> None:
    """Register the web preprocessor with the global registry."""
    from .registry import get_preprocessor_registry

    registry = get_preprocessor_registry()

    if WEB_SCRAPER_AVAILABLE:
        try:
            registry.register(WebPreprocessor())
        except ValueError:
            pass
