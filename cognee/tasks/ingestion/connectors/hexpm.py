"""Hex.pm (Elixir / Erlang) package registry data-source connector for Cognee.

Ingests Elixir/Erlang hex package specifications, hexdocs links, releases,
and metadata directly from hex.pm into Cognee memory.
"""

import logging
from collections.abc import Iterable, Iterator
from typing import Any
import httpx

logger = logging.getLogger(__name__)

HEX_API_BASE = "https://hex.pm/api/packages"

def fetch_hex_package(package_name: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Fetches raw package details from the Hex.pm public JSON API."""
    url = f"{HEX_API_BASE}/{package_name}"
    headers = {"User-Agent": "Cognee-Hex-Connector/1.0 (https://cognee.ai)"}
    
    if client is not None:
        response = client.get(url, headers=headers, timeout=20.0)
    else:
        with httpx.Client() as c:
            response = c.get(url, headers=headers, timeout=20.0)
            
    response.raise_for_status()
    return response.json()

def hexpm_source(packages: list[str]) -> Iterator[dict[str, Any]]:
    """Yields normalized document records from Hex.pm for ingestion into Cognee memory.
    
    Args:
        packages: List of Hex package names (e.g. ['phoenix', 'ecto', 'jason', 'plug'])
    """
    with httpx.Client() as client:
        for pkg_name in packages:
            try:
                pkg_data = fetch_hex_package(pkg_name, client=client)
                name = pkg_data.get("name", pkg_name)
                meta = pkg_data.get("meta", {})
                desc = meta.get("description", "")
                licenses = meta.get("licenses", [])
                links = meta.get("links", {})
                docs_url = pkg_data.get("docs_html_url", f"https://hexdocs.pm/{pkg_name}")
                releases = [r.get("version") for r in pkg_data.get("releases", [])]
                latest_version = releases[0] if releases else ""
                
                doc_text = f"Package: {name}\nLatest Version: {latest_version}\nDescription: {desc}\nDocs: {docs_url}\nLicenses: {', '.join(licenses)}\nRecent Releases: {', '.join(releases[:5])}"
                
                yield {
                    "id": f"hexpm:{name}",
                    "name": name,
                    "version": latest_version,
                    "description": desc,
                    "documentation": docs_url,
                    "licenses": licenses,
                    "links": links,
                    "text": doc_text,
                }
            except Exception as e:
                logger.warning("Failed to fetch Hex package %s: %s", pkg_name, e)
