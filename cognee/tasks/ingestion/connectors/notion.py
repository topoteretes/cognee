"""DLT source for Notion pages (full-snapshot sync + forget-on-delete).

Fetches Notion pages and renders their block children to markdown, then yields
them as a dlt resource for cognee's ingestion pipeline.

Unlike the relational dlt path (SQL/CSV), Notion pages are ingested as *normal
documents*: the source declares ``cognee_document_source = "notion"``, so
``resolve_dlt_sources`` tags each row ``external_metadata["source"] = "notion"``
(not ``"dlt"``). ``is_dlt_sourced`` therefore returns False and each page flows
through the standard cognify entity-extraction pipeline — the right treatment
for prose — instead of the deterministic dlt-row schema-context path.

The source is a full snapshot: ``write_disposition="replace"`` rewrites staging
with exactly the pages currently visible to the integration each run. Deletions
propagate for free — an archived, trashed, or unshared page simply drops out of
Notion's listings, so it is absent from the snapshot and cognee's existing
``orphan_cleanup`` removes it from the graph and vector stores. Unchanged pages
keep a stable content-hash ``data_id``, so they are not re-ingested or
re-cognified. (Notion has no delete feed, and search/database queries omit
trashed pages rather than returning them flagged, so a merge + ``hard_delete``
approach cannot see deletions — hence the Slack-style full-snapshot model.)
"""

import os
import time
from typing import Any, Optional

from cognee.shared.logging_utils import get_logger

from ..dlt_utils import DOCUMENT_SOURCE_ATTR

logger = get_logger("notion_connector")

# dlt resource / staging-table name for Notion pages.
NOTION_TABLE_NAME = "notion_pages"
NOTION_SOURCE_NAME = "notion"
# Pin the Notion API version so upstream changes can't silently alter parsing.
_NOTION_VERSION = "2022-06-28"

# Retry budget for rate-limited / transient Notion API responses.
_MAX_RETRIES = 5

_EXTRA_HINT = (
    'The Notion connector requires the "notion" extra: pip install "cognee[notion]" '
    "(provides dlt and notion-client)."
)

_HEADING_PREFIX = {"heading_1": "# ", "heading_2": "## ", "heading_3": "### "}


def notion_source(
    token: Optional[str] = None,
    page_ids: Optional[list[str]] = None,
    database_ids: Optional[list[str]] = None,
    client: Any = None,
):
    """Create a dlt source that yields Notion pages as markdown documents.

    Args:
        token: Notion integration token. Falls back to ``NOTION_API_KEY``.
        page_ids: Restrict ingestion to these page ids. When omitted (and no
            ``database_ids``), all pages the integration can see are searched.
        database_ids: Restrict ingestion to pages in these databases.
        client: Pre-built ``notion_client.Client`` (mainly a test-injection
            point); when omitted one is built from the token above.

    Returns:
        A dlt source suitable for ``cognee.add(...)`` / ``cognee.remember(...)``.
    """
    try:
        import dlt
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc

    if client is None:
        try:
            from notion_client import Client
        except ImportError as exc:
            raise ImportError(_EXTRA_HINT) from exc

        resolved_token = token or os.environ.get("NOTION_API_KEY")
        if not resolved_token:
            raise ValueError(
                "Notion integration token required: pass token= or set NOTION_API_KEY."
            )
        client = Client(auth=resolved_token, notion_version=_NOTION_VERSION)

    @dlt.resource(name=NOTION_TABLE_NAME, primary_key="id", write_disposition="replace")
    def notion_pages():
        # Full-snapshot sync: each run replaces staging with exactly the pages
        # currently visible to the integration. Archived/trashed pages are
        # dropped from Notion's listings (and skipped below on the page_ids
        # path), so they fall out of staging and cognee's orphan_cleanup then
        # forgets them from the graph + vector stores. Unchanged pages keep a
        # stable content-hash data_id, so they are not re-ingested/re-cognified.
        count = 0
        for page in _iter_pages(client, page_ids, database_ids):
            if page.get("archived") or page.get("in_trash"):
                continue
            try:
                row = _page_to_row(client, page)
            except Exception as exc:
                # Isolate a single bad page (permission/render error) instead of
                # forfeiting the whole sync.
                logger.warning("Notion: skipping page %s: %s", page.get("id"), exc)
                continue
            count += 1
            yield row
        logger.info("Notion: synced %d page(s).", count)

    @dlt.source(name=NOTION_SOURCE_NAME)
    def _notion():
        return notion_pages

    source = _notion()
    # Opt into the document ingestion path (page → text document → cognify).
    # resolve_dlt_sources reads this marker; it never imports this connector.
    setattr(source, DOCUMENT_SOURCE_ATTR, NOTION_SOURCE_NAME)
    return source


# ---------------------------------------------------------------------------
# Notion API helpers (module-private)
# ---------------------------------------------------------------------------


def _request(method, **kwargs):
    """Call a Notion API method, retrying on rate-limit / transient errors.

    notion-client does not retry or honor ``Retry-After`` itself, and Notion
    enforces ~3 requests/second, so a page with many nested blocks would
    otherwise 429 and abort the sync.
    """
    from notion_client.errors import APIResponseError

    for attempt in range(_MAX_RETRIES):
        try:
            return method(**kwargs)
        except APIResponseError as exc:
            status = getattr(exc, "status", None)
            retryable = status in (429, 502, 503, 504) or getattr(exc, "code", "") == "rate_limited"
            if not retryable or attempt == _MAX_RETRIES - 1:
                raise
            delay = _retry_after(getattr(exc, "headers", None), attempt)
            logger.warning(
                "Notion: %s — retrying in %.1fs (%d/%d).",
                status or exc,
                delay,
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(delay)


def _retry_after(headers, attempt: int) -> float:
    """Seconds to wait before retrying: the Retry-After header, else backoff."""
    header = (headers or {}).get("retry-after") or (headers or {}).get("Retry-After")
    try:
        return float(header)
    except (TypeError, ValueError):
        return float(2**attempt)


def _iter_pages(client, page_ids, database_ids):
    """Yield raw Notion page objects for the configured scope."""
    if page_ids:
        for page_id in page_ids:
            try:
                yield _request(client.pages.retrieve, page_id=page_id)
            except Exception as exc:
                # A single missing/inaccessible page must not abort the sync.
                logger.warning("Notion: skipping page %s: %s", page_id, exc)
        return

    if database_ids:
        for database_id in database_ids:
            yield from _paginate(client.databases.query, database_id=database_id)
        return

    # No explicit scope: search every page the integration can see.
    yield from _paginate(client.search, filter={"property": "object", "value": "page"})


def _paginate(method, **kwargs):
    """Yield results across Notion's cursor-based pagination."""
    cursor = None
    while True:
        response = (
            _request(method, start_cursor=cursor, **kwargs)
            if cursor
            else _request(method, **kwargs)
        )
        for item in response.get("results", []):
            yield item
        cursor = response.get("next_cursor")
        # Stop on the last page, or if Notion signals "more" without a cursor
        # (contract violation) so we can't loop forever.
        if not response.get("has_more") or not cursor:
            return


def _page_to_row(client, page: dict) -> dict:
    """Flatten a Notion page + its block children into a document row.

    Only ``title``/``content`` (+ ``id``/``url`` for identity and provenance)
    are kept, so a metadata-only edit that bumps ``last_edited_time`` without
    changing the text does not churn the content-hash data_id.
    """
    return {
        "id": page.get("id"),
        "url": page.get("url"),
        "title": _page_title(page),
        "content": _render_blocks(client, page.get("id")),
    }


def _page_title(page: dict) -> str:
    """Extract the page title from its title property."""
    properties = page.get("properties") or {}
    for prop in properties.values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return _rich_text(prop.get("title"))
    return ""


def _render_blocks(client, block_id: Optional[str], depth: int = 0) -> str:
    """Render a block's children to markdown, recursing into nested blocks."""
    # Guard against pathological nesting / cycles.
    if not block_id or depth > 10:
        return ""

    lines: list[str] = []
    for block in _paginate(client.blocks.children.list, block_id=block_id):
        rendered = _render_block(block)
        if rendered:
            lines.append(rendered)
        if block.get("has_children"):
            nested = _render_blocks(client, block.get("id"), depth + 1)
            if nested:
                lines.append(nested)

    return "\n".join(lines)


def _render_block(block: dict) -> str:
    """Render a single Notion block to a markdown line."""
    block_type = block.get("type")
    if not block_type:
        return ""

    payload = block.get(block_type) or {}
    text = _rich_text(payload.get("rich_text"))

    if block_type in _HEADING_PREFIX:
        return f"{_HEADING_PREFIX[block_type]}{text}" if text else ""
    if block_type == "bulleted_list_item":
        return f"- {text}" if text else ""
    if block_type == "numbered_list_item":
        return f"1. {text}" if text else ""
    if block_type == "to_do":
        checked = "x" if payload.get("checked") else " "
        return f"- [{checked}] {text}" if text else ""
    if block_type == "code":
        language = payload.get("language") or ""
        return f"```{language}\n{text}\n```" if text else ""

    # Paragraph, quote, callout, toggle, and any other rich_text block render as
    # their plain text.
    return text


def _rich_text(rich_text: Any) -> str:
    """Concatenate the plain_text of a Notion rich_text array."""
    if not isinstance(rich_text, list):
        return ""
    return "".join(part.get("plain_text", "") for part in rich_text if isinstance(part, dict))
