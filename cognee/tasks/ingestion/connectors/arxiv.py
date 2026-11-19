"""
arXiv data-source connector for Cognee.

Fetches research papers from the public arXiv Atom API (no auth required)
and yields them as dlt rows for ingestion into Cognee's knowledge graph.

Usage::

    import cognee
    from cognee.tasks.ingestion.connectors.arxiv import arxiv_source

    await cognee.remember(
        arxiv_source(
            query="transformer attention mechanism",
            max_results=10,
            resource_name="arxiv_papers",
        ),
        dataset_name="my_arxiv_papers",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
"""

import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

logger = get_logger("arxiv_connector")

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds between retries


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _get_text(element, path: str, ns: dict, default: str = "") -> str:
    """Safely extract stripped text from an XML element."""
    node = element.find(path, ns)
    return node.text.strip() if node is not None and node.text else default


def _parse_arxiv_xml(xml_text: str) -> list[dict]:
    """Parse an arXiv Atom XML response into a list of paper dicts."""
    root = ET.fromstring(xml_text)
    papers = []

    for entry in root.findall("atom:entry", ARXIV_NS):
        authors = [
            a.find("atom:name", ARXIV_NS).text
            for a in entry.findall("atom:author", ARXIV_NS)
            if a.find("atom:name", ARXIV_NS) is not None
        ]

        # Primary category first, then the rest — deduplicated, order preserved
        primary = [
            c.get("term", "")
            for c in entry.findall("arxiv:primary_category", ARXIV_NS)
        ]
        all_cats = primary + [
            c.get("term", "")
            for c in entry.findall("atom:category", ARXIV_NS)
        ]
        categories = list(dict.fromkeys(all_cats))

        pdf_url = next(
            (
                lnk.get("href", "")
                for lnk in entry.findall("atom:link", ARXIV_NS)
                if lnk.get("title") == "pdf"
            ),
            "",
        )

        papers.append(
            {
                "id": _get_text(entry, "atom:id", ARXIV_NS),
                "title": _get_text(entry, "atom:title", ARXIV_NS).replace("\n", " "),
                "abstract": _get_text(entry, "atom:summary", ARXIV_NS),
                "published": _get_text(entry, "atom:published", ARXIV_NS)[:10],
                "authors": authors,
                "categories": categories,
                "pdf_url": pdf_url,
                "doi": _get_text(entry, "arxiv:doi", ARXIV_NS),
            }
        )

    return papers


# ---------------------------------------------------------------------------
# HTTP fetch (synchronous — dlt resources are generators, not async)
# ---------------------------------------------------------------------------

def _fetch_papers(
    http_client: Any,
    query: str,
    paper_ids: list[str] | None,
    category: str | None,
    max_results: int,
) -> list[dict]:
    """Call the arXiv API and return parsed paper dicts, with retry on errors."""
    if paper_ids:
        params: dict = {"id_list": ",".join(paper_ids), "max_results": max_results}
    else:
        parts = []
        if query:
            parts.append(f"all:{query}")
        if category:
            parts.append(f"cat:{category}")
        params = {
            "search_query": " AND ".join(parts) if parts else "all:machine learning",
            "max_results": max_results,
            "sortBy": "relevance",
        }

    for attempt in range(_MAX_RETRIES):
        try:
            response = http_client.get(ARXIV_API_BASE, params=params, timeout=30.0)
            response.raise_for_status()
            return _parse_arxiv_xml(response.text)
        except httpx.HTTPStatusError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = _RETRY_DELAY * (attempt + 1)
            logger.warning(
                "arXiv: HTTP %s, retrying in %.1fs (%d/%d).",
                exc.response.status_code,
                delay,
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(delay)
        except httpx.TransportError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = _RETRY_DELAY * (attempt + 1)
            logger.warning(
                "arXiv: transport error (%s), retrying in %.1fs (%d/%d).",
                exc,
                delay,
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(delay)

    return []


# ---------------------------------------------------------------------------
# Document rendering
# ---------------------------------------------------------------------------

def _paper_to_document(paper: dict) -> str:
    """Convert a paper dict to a Cognee-ready markdown document."""
    authors_str = ", ".join(paper["authors"]) if paper["authors"] else "Unknown"
    categories_str = ", ".join(paper["categories"]) if paper["categories"] else "Unknown"
    doi_line = f"**DOI:** {paper['doi']}\n" if paper.get("doi") else ""

    return (
        f"# {paper['title']}\n\n"
        f"**Source:** arxiv\n"
        f"**Paper ID:** {paper['id']}\n"
        f"**Authors:** {authors_str}\n"
        f"**Published:** {paper['published']}\n"
        f"**Categories:** {categories_str}\n"
        f"**PDF:** {paper['pdf_url']}\n"
        f"{doi_line}"
        f"\n## Abstract\n\n"
        f"{paper['abstract']}\n"
    )


# ---------------------------------------------------------------------------
# Public dlt resource factory
# ---------------------------------------------------------------------------

def arxiv_source(
    query: str = "",
    paper_ids: list[str] | None = None,
    category: str | None = None,
    max_results: int = 10,
    resource_name: str = "arxiv_papers",
    http_client: Any = None,
):
    """Return a dlt resource yielding one row per arXiv paper.

    Hand the result to ``cognee.remember(...)`` with
    ``write_disposition="merge"`` and ``primary_key="id"``.

    Args:
        query:         Free-text search (e.g. "diffusion models image generation").
        paper_ids:     Specific arXiv IDs (e.g. ["1706.03762", "2301.07041"]).
                       When provided, *query* and *category* are ignored.
        category:      arXiv category filter (e.g. "cs.AI", "cs.LG", "math.CO").
        max_results:   Number of papers to fetch (default 10, arXiv caps at 100).
        resource_name: Stable dlt resource name (and staging table name).
        http_client:   Pre-built ``httpx.Client`` — mainly a test injection point.

    Example::

        await cognee.remember(
            arxiv_source(query="attention is all you need", max_results=5),
            dataset_name="arxiv_papers",
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
    """
    import dlt

    node_set_tag = category or query or "general"

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def _arxiv_papers():
        owned_client = None if http_client is not None else httpx.Client(timeout=30.0)
        try:
            client = http_client or owned_client
            papers = _fetch_papers(client, query, paper_ids, category, max_results)
            logger.info("arXiv: fetched %d paper(s).", len(papers))
            for paper in papers:
                yield {
                    "id": paper["id"],
                    "title": paper["title"],
                    "url": paper["pdf_url"],
                    "content": _paper_to_document(paper),
                    "_deleted": False,
                    NODE_SET_COLUMN: [f"arxiv:{node_set_tag}"],
                }
        finally:
            if owned_client is not None:
                owned_client.close()

    resource = _arxiv_papers()
    # Mark this as a document source so resolve_dlt_sources picks it up.
    setattr(resource, DOCUMENT_SOURCE_ATTR, "arxiv")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    return resource
