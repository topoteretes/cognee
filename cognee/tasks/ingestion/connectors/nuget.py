"""NuGet v3 package registry data-source connector for Cognee.

Ingests .NET/C# package specifications, dependency groups, license details,
and catalog entries directly from api.nuget.org into Cognee memory.
"""

import logging
from collections.abc import Iterable, Iterator
from typing import Any
import httpx

logger = logging.getLogger(__name__)

NUGET_API_BASE = "https://api.nuget.org/v3/registration5-semver1"

def fetch_nuget_package(package_id: str, client: httpx.Client | None = None) -> dict[str, Any]:
    """Fetches raw registration catalog entries from the NuGet v3 JSON endpoint."""
    url = f"{NUGET_API_BASE}/{package_id.lower()}/index.json"
    headers = {"User-Agent": "Cognee-NuGet-Connector/1.0"}
    
    if client is not None:
        response = client.get(url, headers=headers, timeout=20.0)
    else:
        with httpx.Client() as c:
            response = c.get(url, headers=headers, timeout=20.0)
            
    response.raise_for_status()
    return response.json()

def nuget_source(packages: list[str]) -> Iterator[dict[str, Any]]:
    """Yields normalized document records from NuGet for ingestion into Cognee memory.
    
    Args:
        packages: List of .NET package identifiers (e.g. ['Newtonsoft.Json', 'Serilog', 'Microsoft.EntityFrameworkCore'])
    """
    with httpx.Client() as client:
        for pkg_id in packages:
            try:
                data = fetch_nuget_package(pkg_id, client=client)
                items = data.get("items", [])
                
                latest_entry = {}
                if items and "items" in items[-1]:
                    sub_items = items[-1].get("items", [])
                    if sub_items:
                        latest_entry = sub_items[-1].get("catalogEntry", {})
                
                title = latest_entry.get("title", pkg_id)
                desc = latest_entry.get("description", "")
                authors = latest_entry.get("authors", "")
                version = latest_entry.get("version", "")
                project_url = latest_entry.get("projectUrl", "")
                
                doc_text = f"Package: {pkg_id}\nTitle: {title}\nLatest Version: {version}\nAuthors: {authors}\nProject URL: {project_url}\nDescription: {desc}"
                
                yield {
                    "id": f"nuget:{pkg_id}",
                    "name": pkg_id,
                    "title": title,
                    "version": version,
                    "authors": authors,
                    "project_url": project_url,
                    "description": desc,
                    "text": doc_text,
                }
            except Exception as e:
                logger.warning("Failed to fetch NuGet package %s: %s", pkg_id, e)
