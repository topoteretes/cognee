"""Readwise connector for cognee, a ``dlt`` source that turns highlights into memory.

Sync a set of explicit Readwise resources into cognee incrementally.

    import cognee
    from cognee.tasks.ingestion.connectors import readwise_source

    await cognee.remember(
        readwise_source(),
        dataset_name="my_readwise_data",
        primary_key="id",
        write_disposition="merge",
    )
"""

import os
import logging
from typing import Iterator

import httpx

from cognee.tasks.ingestion import dlt_utils

logger = logging.getLogger(__name__)

READWISE_API_URL = "https://readwise.io/api/v2"

def _get_headers(token: str) -> dict:
    return {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }

def readwise_source(
    token: str | None = None,
    resource_name: str = "readwise_highlights"
):
    """Create a dlt source yielding Readwise highlights.

    Args:
        token: Readwise Access Token. Falls back to READWISE_ACCESS_TOKEN.
        resource_name: The dlt resource / table name to yield rows into.
    """
    try:
        import dlt
    except ImportError as e:
        raise ImportError(
            "The Readwise connector requires dlt: pip install dlt"
        ) from e

    token = token or os.environ.get("READWISE_ACCESS_TOKEN")
    if not token:
        raise ValueError("Readwise access token must be provided or set in READWISE_ACCESS_TOKEN environment variable")

    @dlt.resource(name=resource_name, write_disposition="merge", primary_key="id")
    def readwise_highlights(last_updated_time: dlt.sources.incremental = dlt.sources.incremental("updated")) -> Iterator[dict]:  # noqa: B008
        headers = _get_headers(token)
        url = f"{READWISE_API_URL}/highlights/"
        params = {}
        
        while url:
            resp = httpx.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            
            data = resp.json()
            highlights = data.get("results", [])
            
            for hl in highlights:
                hl_id = hl.get("id")
                text = hl.get("text")
                note = hl.get("note", "")
                url_hl = hl.get("url")
                updated = hl.get("updated")
                
                content = f"Highlight: {text}\nNote: {note}"
                        
                yield {
                    "id": str(hl_id),
                    "title": f"Readwise Highlight {hl_id}",
                    "content": content,
                    "updated": updated,
                    "url": url_hl or f"https://readwise.io/open/{hl_id}"
                }
            
            url = data.get("next")
            params = {}  # query params are embedded in the next URL if they exist
