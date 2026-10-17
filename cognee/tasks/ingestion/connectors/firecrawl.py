import os
from importlib import import_module
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = get_logger("firecrawl")
FIRECRAWL_API_BASE_URL = "https://api.firecrawl.dev/v1"


class _FirecrawlClient:
    def __init__(self, http_client: Any, token: str):
        self._http = http_client
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def scrape(self, url: str) -> dict:
        # Endpoint to initiate scraping
        api_url = f"{FIRECRAWL_API_BASE_URL}/scrape"

        # Data payload for the request
        payload = {
            "url": url,
            "formats": ["markdown"],
        }

        # Send the POST request to initiate scraping
        response = self._http.post(
            api_url,
            json=payload,
            headers=self._headers,
        )

        # Check for successful response
        response.raise_for_status()

        # Return the JSON response
        return response.json()

def _iter_pages(client: _FirecrawlClient, urls: list[str], dataset_name: str):
        """Yield one document row per scraped URL."""
        for url in urls:
            try:
                # Step 1: Scrape the URL using the Firecrawl client
                result = client.scrape(url)

                # Step 2: Extract the data from the result
                data = result.get("data", {})
                raw_markdown = data.get("markdown", "")
                title = data.get("metadata", {}).get("title", url)

                # Step 3: Yield the document row with the required fields
                yield {
                    "id": url,
                    "title": title,
                    "content": raw_markdown,
                    "_deleted": False,
                    NODE_SET_COLUMN: [f"firecrawl_{dataset_name}"],
                    DOCUMENT_SOURCE_ATTR: "firecrawl",
                }
            except Exception as e:
                logger.warning("Firecrawl: URL scrape failed for %s: %s", url, str(e))

def firecrawl_source(
            token: str | None = None,
            urls: list[str] | None = None, 
            dataset_name: str = "default_dataset", 
            resource_name: str = "firecrawl_pages",
            http_client: Any = None,
        ):

        dlt = import_module("dlt")
        import httpx
        resolved_token = token or os.getenv("FIRECRAWL_API_KEY")
        if not resolved_token:
            raise ValueError("Firecrawl API token is required.")

        resolved_urls = list(urls or os.getenv("FIRECRAWL_URLS", "").split(","))

        @dlt.source(
            name=resource_name, 
            primary_key="id", 
            write_disposition="merge", 
            columns={"_deleted": {"data_type": "bool", "hard_delete": True}}
        )

        def firecrawl_pages():
            owned_client = None if http_client is not None else httpx.Client(timeout=30.0)
            try:
                client = _FirecrawlClient(http_client or owned_client, resolved_token)
                yield from _iter_pages(client, resolved_urls, dataset_name)
            finally:
                if owned_client is not None:
                    owned_client.close()

        resource = firecrawl_pages()
        setattr(resource, DOCUMENT_SOURCE_ATTR, "firecrawl")
        setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
        return resource