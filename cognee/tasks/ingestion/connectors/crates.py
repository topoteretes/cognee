"""Crates.io Rust package registry data-source connector for Cognee.

Ingests Rust crate specifications, documentation links, repository links,
and version histories directly from crates.io into Cognee memory.
"""

import logging
from collections.abc import Iterable, Iterator
from typing import Any
import httpx

logger = logging.getLogger(__name__)

CRATES_IO_API_BASE = "https://crates.io/api/v1/crates"

def fetch_crate_details(crate_name: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Fetches raw crate metadata from the Crates.io REST API."""
    url = f"{CRATES_IO_API_BASE}/{crate_name}"
    headers = {"User-Agent": "Cognee-Crates-Connector/1.0 (https://cognee.ai)"}
    
    if client is not None:
        response = client.get(url, headers=headers, timeout=20.0)
    else:
        with httpx.Client() as c:
            response = c.get(url, headers=headers, timeout=20.0)
            
    response.raise_for_status()
    data = response.json()
    return data.get("crate", {})

def crates_source(crates: list[str]) -> Iterator[dict[str, Any]]:
    """Yields normalized document records from Crates.io for ingestion into Cognee memory.
    
    Args:
        crates: List of Rust crate names (e.g. ['tokio', 'serde', 'anyhow', 'axum'])
    """
    with httpx.Client() as client:
        for crate_name in crates:
            try:
                crate_data = fetch_crate_details(crate_name, client=client)
                name = crate_data.get("name", crate_name)
                desc = crate_data.get("description", "")
                docs = crate_data.get("documentation", f"https://docs.rs/{crate_name}")
                repo = crate_data.get("repository", "")
                max_ver = crate_data.get("max_version", "")
                downloads = crate_data.get("downloads", 0)
                
                doc_text = f"Crate: {name}\nLatest Version: {max_ver}\nDescription: {desc}\nDocs: {docs}\nRepository: {repo}\nTotal Downloads: {downloads}"
                
                yield {
                    "id": f"crates:{name}",
                    "name": name,
                    "version": max_ver,
                    "description": desc,
                    "documentation": docs,
                    "repository": repo,
                    "downloads": downloads,
                    "text": doc_text,
                }
            except Exception as e:
                logger.warning("Failed to fetch Crate %s: %s", crate_name, e)
