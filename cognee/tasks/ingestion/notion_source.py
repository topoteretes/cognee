"""DLT source for Notion pages (incremental re-sync + forget-on-delete).

Fetches Notion pages and renders their block children to markdown, then yields
them as a dlt resource for cognee's ingestion pipeline.

Unlike the relational dlt path (SQL/CSV), Notion pages are ingested as *normal
documents*: rows are tagged ``external_metadata["source"] = "notion"`` (not
``"dlt"``), so ``is_dlt_sourced`` returns False and each page flows through the
standard cognify entity-extraction pipeline instead of the deterministic
dlt-row path.

Incremental re-sync uses ``last_edited_time`` as the dlt cursor and the Notion
page id as the primary key, so re-runs only process changed pages. Archived or
trashed pages are flagged with dlt's ``hard_delete`` column hint: they drop out
of the staging table on the next sync, which makes them orphans that cognee's
existing ``orphan_cleanup`` removes from the graph and vector stores.
"""

import os
from typing import Any, Optional
from uuid import UUID

from cognee.modules.data.methods.get_unique_data_id import get_unique_data_id
from cognee.modules.users.models import User

from .data_item import DataItem
from .dlt_row_data import DltRowData

# dlt resource / staging-table name for Notion pages. resolve_dlt_sources keys
# off the source name to route Notion rows through the document path.
NOTION_TABLE_NAME = "notion_pages"
NOTION_SOURCE_NAME = "notion"
_DEFAULT_NOTION_VERSION = "2022-06-28"

# Notion rows are read back in full each sync so orphan cleanup can spot pages
# that dropped out of staging (archived/deleted). Use a high cap so the default
# per-table limit doesn't truncate the read and orphan live pages.
NOTION_MAX_ROWS = 100_000

# Notion block types whose rich_text renders as a plain markdown paragraph.
_TEXT_BLOCKS = {"paragraph", "quote", "callout", "toggle"}
_HEADING_PREFIX = {"heading_1": "# ", "heading_2": "## ", "heading_3": "### "}


def notion_source(
    token: Optional[str] = None,
    page_ids: Optional[list[str]] = None,
    database_ids: Optional[list[str]] = None,
    notion_version: Optional[str] = None,
):
    """Create a dlt source that yields Notion pages as markdown documents.

    Args:
        token: Notion integration token. Falls back to ``NOTION_API_KEY``.
        page_ids: Restrict ingestion to these page ids. When omitted (and no
            ``database_ids``), all pages the integration can see are searched.
        database_ids: Restrict ingestion to pages in these databases.
        notion_version: Notion API version header. Defaults to ``2022-06-28``.

    Returns:
        A dlt source suitable for ``cognee.add(...)`` / ``resolve_dlt_sources``.
    """
    import dlt
    from notion_client import Client

    resolved_token = token or os.environ.get("NOTION_API_KEY")
    if not resolved_token:
        raise ValueError("Notion integration token required: pass token= or set NOTION_API_KEY.")

    client = Client(
        auth=resolved_token,
        notion_version=notion_version or _DEFAULT_NOTION_VERSION,
    )

    @dlt.resource(
        name=NOTION_TABLE_NAME,
        primary_key="id",
        write_disposition="merge",
        # Archiving/trashing a page bumps last_edited_time, so it comes through
        # the incremental window; the hard_delete hint then removes it from
        # staging, making it an orphan that cognee's orphan_cleanup deletes.
        columns={"archived": {"hard_delete": True}},
    )
    def notion_pages(
        last_edited=dlt.sources.incremental("last_edited_time"),
    ):
        for page in _iter_pages(client, page_ids, database_ids):
            yield _page_to_row(client, page)

    @dlt.source(name=NOTION_SOURCE_NAME)
    def _notion():
        return notion_pages

    return _notion()


async def expand_notion_rows(
    rows: list[DltRowData],
    user: User,
) -> tuple[list[DataItem], set[UUID]]:
    """Turn Notion dlt rows into document DataItems for the standard pipeline.

    Returns ``(data_items, fresh_data_ids)``. ``fresh_data_ids`` feeds the
    deferred orphan cleanup so pages that vanished from the source (archived /
    deleted, dropped from staging via ``hard_delete``) are removed downstream.
    """
    data_items: list[DataItem] = []
    fresh_data_ids: set[UUID] = set()

    for row in rows:
        row_identifier = f"dlt:{row.table_name}:{row.primary_key_value}:{row.content_hash}"
        data_id = await get_unique_data_id(row_identifier, user)
        fresh_data_ids.add(data_id)
        data_items.append(_build_notion_data_item(row, data_id))

    return data_items, fresh_data_ids


def _build_notion_data_item(row: DltRowData, data_id: UUID) -> DataItem:
    """Build a document DataItem from a Notion page row (source ``"notion"``)."""
    row_data = row.row_data
    title = _clean(row_data.get("title"))
    content = _clean(row_data.get("content"))
    text = f"# {title}\n\n{content}".strip() if title else content

    external_metadata = {
        "source": "notion",
        "notion_page_id": row_data.get("id"),
        "notion_url": row_data.get("url"),
        "title": title,
        "last_edited_time": row_data.get("last_edited_time"),
        "content_hash": row.content_hash,
    }

    return DataItem(
        data=text,
        label=title or str(row_data.get("id")),
        external_metadata=external_metadata,
        data_id=data_id,
    )


# ---------------------------------------------------------------------------
# Notion API helpers (module-private)
# ---------------------------------------------------------------------------


def _iter_pages(client, page_ids, database_ids):
    """Yield raw Notion page objects for the configured scope."""
    if page_ids:
        for page_id in page_ids:
            yield client.pages.retrieve(page_id=page_id)
        return

    if database_ids:
        for database_id in database_ids:
            yield from _paginate(client.databases.query, database_id=database_id)
        return

    # No explicit scope: search every page the integration can see.
    yield from _paginate(
        client.search,
        filter={"property": "object", "value": "page"},
        sort={"timestamp": "last_edited_time", "direction": "ascending"},
    )


def _paginate(method, **kwargs):
    """Yield results across Notion's cursor-based pagination."""
    cursor = None
    while True:
        response = method(start_cursor=cursor, **kwargs) if cursor else method(**kwargs)
        for item in response.get("results", []):
            yield item
        if not response.get("has_more"):
            return
        cursor = response.get("next_cursor")


def _page_to_row(client, page: dict) -> dict:
    """Flatten a Notion page + its block children into a dlt row."""
    return {
        "id": page.get("id"),
        "last_edited_time": page.get("last_edited_time"),
        # in_trash is the newer flag; archived is the legacy one. Either means
        # the page should be forgotten, so the hard_delete hint fires on both.
        "archived": bool(page.get("archived") or page.get("in_trash")),
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
    if block_type in _TEXT_BLOCKS:
        return text

    return text


def _rich_text(rich_text: Any) -> str:
    """Concatenate the plain_text of a Notion rich_text array."""
    if not isinstance(rich_text, list):
        return ""
    return "".join(part.get("plain_text", "") for part in rich_text if isinstance(part, dict))


def _clean(value: Any) -> str:
    """Coerce a possibly-None cell value to a stripped string."""
    return str(value).strip() if value is not None else ""
