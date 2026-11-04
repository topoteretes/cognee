"""Packagist / Composer data-source connector for Cognee.

Ingests PHP/Composer package specifications, release versions, dependencies,
and README documentation directly from Packagist.org into Cognee memory.
"""

import logging
from collections.abc import Iterable, Iterator
from typing import Any
import httpx

logger = logging.getLogger(__name__)

PACKAGIST_API_BASE = "https://packagist.org/packages"

def fetch_packagist_package(package_name: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Fetches raw package details from the official Packagist JSON API."""
    url = f"{PACKAGIST_API_BASE}/{package_name}.json"
    headers = {"User-Agent": "Cognee-Packagist-Connector/1.0"}
    
    if client is not None:
        response = client.get(url, headers=headers, timeout=20.0)
    else:
        with httpx.Client() as c:
            response = c.get(url, headers=headers, timeout=20.0)
            
    response.raise_for_status()
    data = response.json()
    return data.get("package", {})

def packagist_source(packages: list[str]) -> Iterator[dict[str, Any]]:
    """Yields normalized document records from Packagist for ingestion into Cognee memory.
    
    Args:
        packages: List of composer package names in vendor/package format (e.g. ['laravel/framework', 'guzzlehttp/guzzle'])
    """
    with httpx.Client() as client:
        for pkg_name in packages:
            try:
                pkg_data = fetch_packagist_package(pkg_name, client=client)
                name = pkg_data.get("name", pkg_name)
                desc = pkg_data.get("description", "")
                repository = pkg_data.get("repository", "")
                maintainers = [m.get("name") for m in pkg_data.get("maintainers", [])]
                versions = list(pkg_data.get("versions", {}).keys())
                
                doc_text = f"Package: {name}\nDescription: {desc}\nRepository: {repository}\nMaintainers: {', '.join(maintainers)}\nLatest Versions: {', '.join(versions[:5])}"
                
                yield {
                    "id": f"packagist:{name}",
                    "name": name,
                    "description": desc,
                    "repository": repository,
                    "maintainers": maintainers,
                    "versions": versions[:10],
                    "text": doc_text,
                }
            except Exception as e:
                logger.warning("Failed to fetch Packagist package %s: %s", pkg_name, e)
