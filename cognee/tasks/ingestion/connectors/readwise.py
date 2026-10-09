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
            url = None  # prevent infinite loop for now
