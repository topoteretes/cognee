"""Notion connector for cognee, a ``dlt`` source that turns Notion pages and
database rows into memory.

Sync a set of explicit Notion pages and/or data sources (database "tables")
into cognee, incrementally and with forget-on-deletion::

    import cognee
    from cognee.tasks.ingestion.connectors import notion_source

    await cognee.remember(
        notion_source(
            root_page_ids=["<page id from the page URL>"],
            resource_name="notion_<connection id>",
        ),
        dataset_name="my_notion_space",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

Design
------
* **Scope**: explicit ``root_page_ids`` and/or ``root_data_source_ids``. This
  connector does not search the workspace; the caller (or a pod/UI on top of
  it) picks the roots. A ``root_data_source_ids`` entry may also be a
  *database* id, it is resolved into that database's data sources.
* **Transport**: plain ``httpx`` (a core cognee dependency), not
  ``notion-client``, so this connector adds no new dependency. ``http_client``
  is an injection point for tests.
* **API version**: pinned to ``2025-09-03``, the version that introduced data
  sources as the object a database's rows actually belong to. Verified
  against the official ``notion-sdk-js`` source (``Client.defaultNotionVersion``
  and ``src/api-endpoints/*.ts``), not the docs site, see the module docstring
  note on ``NOTION_API_VERSION`` below.
* **One row per page**: a database row is a page too (Notion's API treats it
  that way); its properties are rendered as text lines on top of its block
  content. ``cognee_node_set`` places the page under its workspace and
  selected root, by id; no titles or paths.
* **Incremental**: the tree is walked on every run (Notion has no delete
  feed, so presence has to be rediscovered), including container blocks such
  as columns, toggles and synced blocks that can hold sub-pages, but a page's
  block content is only rendered when its ``last_edited_time`` or its root
  changed since the last run. A synced block that duplicates another page's
  original is not searched: its sub-pages live under the original, so they
  are only synced when the original's page is under a selected root. A tree nested past the
  depth caps aborts the run, and keeps aborting every run until it is edited.
  Pages that disappeared (unshared, trashed, moved out of the selected roots)
  are emitted as ``_deleted`` tombstones. A transient API error aborts the
  run before any tombstone is emitted, so a partial walk can never look like
  a mass deletion. Only a 403/404 on fetching a page or database reached
  through the tree (not one of the explicit roots) means it is gone and is
  tombstoned; a failure listing a readable page's blocks or querying a data
  source aborts the run instead.
* **Moving a page** between two selected roots, or changing the roots so
  another one reaches it, changes its ``cognee_node_set`` (which is hashed
  into its row identity), so it is re-ingested once under its new root.
* **Structure**: each row also carries ``cognee_structure``, where the page
  sits in the tree it was reached through (see ``_ancestors_of``). It is not
  hashed, so moving a page inside its root keeps its identity and does not
  extract again; it is part of what is remembered per page, so a move
  re-emits the row. The structure pass turns it into ``child_of`` edges.

Limitations
-----------
* Teamspaces are not exposed by Notion's public REST API, so pages are
  grouped by selected root, never by teamspace.
* Attachments (files/images/PDF/video blocks) are rendered as a name/caption
  reference line only, the binary is not downloaded in this version.
* Relation and people property values are rendered as raw ids/names in the
  page's content text; resolving them into real edges is a later ticket.
* No client-side throttle: Notion's rate limit (about 3 requests/second)
  surfaces as 429s, which are retried with ``Retry-After`` or exponential
  backoff, bounded by ``_MAX_RETRIES``.
* Every run lists each page's blocks (and its container blocks) to find
  sub-pages and deletions, so a run costs one or more requests per page even
  when nothing changed: a 20,000-page tree is tens of thousands of requests,
  hours at Notion's rate limit.
* Each source needs its own ``resource_name`` within a dataset (see
  ``notion_source``); a run refuses to continue when the stored state belongs
  to another workspace.
* A data source with more than 10,000 rows aborts every run: Notion cuts a
  query off there, and syncing the first 10,000 would forget the rest.
* The walk runs inside ``ingest_dlt_source``'s process-wide staging lock, so
  while a large tree syncs every other dlt ingestion in the process waits.
* A page is only re-rendered when it changes itself. Text it shows from other
  objects (sub-page and database titles, synced copies, mentions) stays as it
  was until the page is edited.
"""

import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN, STRUCTURE_COLUMN

logger = get_logger("notion_connector")

NOTION_API_BASE_URL = "https://api.notion.com/v1"

# 2025-09-03 is the current Notion API version and the first where a database's
# rows belong to a "data source" rather than the database directly. Confirmed
# against notion-sdk-js (Client.defaultNotionVersion = "2025-09-03" and the
# endpoint paths below) rather than the docs site: developers.notion.com is a
# JS-rendered Mintlify site this environment could not scrape.
NOTION_API_VERSION = "2025-09-03"

# Retry budget for rate-limited / transient Notion API responses.
_MAX_RETRIES = 5
# Caps how many page/database/data-source levels are walked below a root (a
# page under a database uses three). Hitting it aborts the run, like the
# discovery cap, because stopping would tombstone everything below it.
_MAX_WALK_DEPTH = 100
# Caps how deep the children of any block are rendered into a page's text;
# deeper content is dropped from the text only. Table rows are not affected.
_MAX_RENDER_DEPTH = 10
# Caps how deep container blocks are searched for sub-pages. Hitting it aborts
# the run: stopping silently would tombstone every page below the cap.
_MAX_DISCOVERY_DEPTH = 50
_PAGE_SIZE = 100

_EXTRA_HINT = "Set token= or the NOTION_API_KEY environment variable."

_HEADING_PREFIX = {"heading_1": "# ", "heading_2": "## ", "heading_3": "### "}


class NotionAPIError(Exception):
    """A Notion API response that isn't worth retrying (or ran out of retries)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# HTTP client (thin wrapper: auth header, version header, retry/backoff)
# ---------------------------------------------------------------------------
class _NotionClient:
    def __init__(self, http_client: Any, token: str):
        self._http = http_client
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        }

    def request(
        self, method: str, path: str, *, params: dict | None = None, json_body: dict | None = None
    ) -> dict:
        import httpx

        url = f"{NOTION_API_BASE_URL}/{path}"
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._http.request(
                    method, url, params=params, json=json_body, headers=self._headers
                )
            except httpx.TransportError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = float(2**attempt)
                logger.warning(
                    "Notion: transport error on %s (%s), retrying in %.1fs.", path, exc, delay
                )
                time.sleep(delay)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == _MAX_RETRIES - 1:
                    raise NotionAPIError(
                        response.status_code,
                        f"Notion API error {response.status_code} on {path}: {response.text}",
                    )
                delay = _retry_after(response.headers, attempt)
                logger.warning(
                    "Notion: %s on %s, retrying in %.1fs (%d/%d).",
                    response.status_code,
                    path,
                    delay,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                raise NotionAPIError(
                    response.status_code,
                    f"Notion API error {response.status_code} on {path}: {response.text}",
                )

            return response.json()

        raise NotionAPIError(
            0, f"Notion API request to {path} failed after {_MAX_RETRIES} retries."
        )

    def get_self(self) -> dict:
        return self.request("GET", "users/me")

    def get_page(self, page_id: str) -> dict:
        return self.request("GET", f"pages/{page_id}")

    def get_database(self, database_id: str) -> dict:
        return self.request("GET", f"databases/{database_id}")

    def get_data_source(self, data_source_id: str) -> dict:
        return self.request("GET", f"data_sources/{data_source_id}")


def _retry_after(headers, attempt: int) -> float:
    """Seconds to wait before retrying: the Retry-After header, else backoff."""
    header = headers.get("retry-after") if headers is not None else None
    try:
        return float(header)
    except (TypeError, ValueError):
        return float(2**attempt)


def _is_gone(exc: Exception) -> bool:
    """True when a page/database/data source is permanently inaccessible."""
    return isinstance(exc, NotionAPIError) and exc.status_code in (403, 404)


def _list_block_children(client: _NotionClient, block_id: str):
    """Yield a block's direct children, across Notion's cursor pagination."""
    cursor = None
    while True:
        params = {"page_size": _PAGE_SIZE}
        if cursor:
            params["start_cursor"] = cursor
        response = client.request("GET", f"blocks/{block_id}/children", params=params)
        yield from response.get("results", [])
        cursor = response.get("next_cursor")
        if not response.get("has_more"):
            return
        if not cursor:
            raise NotionAPIError(0, "Notion reported more results without a next_cursor.")


def _query_data_source(client: _NotionClient, data_source_id: str):
    """Yield every result of a data source query, across Notion's cursor pagination.

    A wiki's data source returns its child data sources next to its pages, so
    results are pages or data sources.
    """
    cursor = None
    while True:
        body = {"page_size": _PAGE_SIZE}
        if cursor:
            body["start_cursor"] = cursor
        response = client.request("POST", f"data_sources/{data_source_id}/query", json_body=body)
        yield from response.get("results", [])
        # Notion stops a query at 10,000 rows with has_more false and this
        # status; the rows past the cut would otherwise be tombstoned.
        if (response.get("request_status") or {}).get("type") == "incomplete":
            raise NotionAPIError(
                0,
                f"Notion returned an incomplete result for data source {data_source_id} "
                "(over 10,000 rows); aborting instead of forgetting the rest.",
            )
        cursor = response.get("next_cursor")
        if not response.get("has_more"):
            return
        if not cursor:
            raise NotionAPIError(0, "Notion reported more results without a next_cursor.")


# ---------------------------------------------------------------------------
# Tree walk (pure given a client + state, unit-testable without dlt)
# ---------------------------------------------------------------------------
def _canonical_id(value: str) -> str:
    """Notion accepts ids with or without dashes; key everything on the dashed form."""
    try:
        return str(UUID(str(value)))
    except ValueError:
        return str(value)


@dataclass(frozen=True)
class _WalkItem:
    kind: str  # "page" | "database" | "data_source"
    id: str
    root_id: str
    depth: int = 0
    is_root: bool = False
    # Pages discovered via a data-source query already come back as full page
    # objects; carrying them forward here avoids a redundant GET per row.
    prefetched: dict | None = None
    # Where this item sits in the tree it was reached through, nearest parent
    # first (see ``_ancestors_of``). Empty for a root.
    ancestors: tuple[dict, ...] = ()
    # The title of a database or data source, as the object that listed it names it.
    name: str = ""


def _ancestors_of(parent: _WalkItem, name: str | None = None) -> tuple[dict, ...]:
    """The ``ancestors`` of an item reached through ``parent``.

    A page is a row of this source and the chain stops there: whatever sits
    above it is that page's own business, so moving a page never changes the
    structure of its descendants and never re-renders them. A database or data
    source has no row, so its chain carries on up to the page it hangs from.
    ``name`` overrides ``parent.name`` for a database whose title was only read
    once it was fetched.
    """
    if parent.kind == "page":
        return ({"kind": "page", "id": parent.id, "document": True},)
    title = parent.name if name is None else name
    entry = {"kind": parent.kind, "id": parent.id, **({"name": title} if title else {})}
    return (entry, *parent.ancestors)


def _data_source_name(data_source: dict) -> str:
    """The title of a data source, from its own object or from a database's listing of it."""
    return data_source.get("name") or _rich_text(data_source.get("title"))


def _resolve_roots(
    client: _NotionClient, root_page_ids: list[str], root_data_source_ids: list[str]
) -> list[_WalkItem]:
    """Turn the caller's roots into walk items, resolving a database id to its
    data sources.

    A failure here (bad id, no access) always raises: these are the roots the
    caller explicitly configured, so a silent skip would quietly ingest
    nothing instead of surfacing the misconfiguration.
    """
    roots = []
    for page_id in root_page_ids:
        page_id = _canonical_id(page_id)
        roots.append(_WalkItem("page", page_id, page_id, is_root=True))
    for root_id in root_data_source_ids:
        root_id = _canonical_id(root_id)
        try:
            root_data_source = client.get_data_source(root_id)
        except NotionAPIError as exc:
            if exc.status_code != 404:
                raise
            try:
                database = client.get_database(root_id)
            except NotionAPIError:
                raise NotionAPIError(
                    exc.status_code,
                    f"Notion root {root_id!r} is not a page, data source, or database "
                    "this integration can access.",
                ) from exc
            database_root = _WalkItem(
                "database", root_id, root_id, name=_rich_text(database.get("title"))
            )
            for data_source in database.get("data_sources") or []:
                roots.append(
                    _WalkItem(
                        "data_source",
                        _canonical_id(data_source["id"]),
                        root_id,
                        is_root=True,
                        ancestors=_ancestors_of(database_root),
                        name=_data_source_name(data_source),
                    )
                )
            continue
        # A data source named as a root still knows the database it belongs to.
        parent = root_data_source.get("parent") or {}
        database_root = (
            _WalkItem("database", _canonical_id(parent["database_id"]), root_id)
            if parent.get("type") == "database_id" and parent.get("database_id")
            else None
        )
        roots.append(
            _WalkItem(
                "data_source",
                root_id,
                root_id,
                is_root=True,
                ancestors=_ancestors_of(database_root) if database_root else (),
                name=_data_source_name(root_data_source),
            )
        )
    return roots


def _iter_rows(
    client: _NotionClient,
    root_page_ids: list[str],
    root_data_source_ids: list[str],
    workspace_id: str,
    previous_pages: dict[str, dict],
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
):
    """Yield one document row per changed/new page, then deletion tombstones.

    ``previous_pages`` is ``{page_id: {"last_edited_time", "root_id", "ancestors"}}``
    from the prior run. Any transient error (network, 429/5xx exhausted retries)
    propagates un-caught, which aborts the whole dlt extraction before
    ``state`` is touched, a partial walk never produces tombstones. A 403/404
    on a page or database reached *through* the tree (not an explicit root)
    means it is gone and is skipped so it drops out of ``present_pages`` and
    gets tombstoned below.
    """
    stats.clear()
    stats.update(
        pages_scanned=0, pages_changed=0, pages_unchanged=0, deleted=0, containers_scanned=0
    )

    queue: deque[_WalkItem] = deque(_resolve_roots(client, root_page_ids, root_data_source_ids))
    visited: set[str] = set()
    present_pages: dict[str, dict] = {}

    while queue:
        item = queue.popleft()
        if check_active is not None:
            check_active()
        if item.id in visited:
            continue
        visited.add(item.id)
        if item.depth > _MAX_WALK_DEPTH:
            raise NotionAPIError(
                0,
                f"Notion {item.kind} {item.id} is nested deeper than {_MAX_WALK_DEPTH} levels "
                "below its root; aborting instead of forgetting the pages below it.",
            )

        if item.kind == "database":
            stats["containers_scanned"] += 1
            try:
                database = client.get_database(item.id)
            except NotionAPIError as exc:
                if item.is_root or not _is_gone(exc):
                    raise
                logger.warning("Notion: database %s is gone, skipping: %s", item.id, exc)
                continue
            ancestors = _ancestors_of(item, _rich_text(database.get("title")))
            for data_source in database.get("data_sources") or []:
                queue.append(
                    _WalkItem(
                        "data_source",
                        _canonical_id(data_source["id"]),
                        item.root_id,
                        item.depth + 1,
                        ancestors=ancestors,
                        name=_data_source_name(data_source),
                    )
                )
            continue

        if item.kind == "data_source":
            stats["containers_scanned"] += 1
            # A data source is only reached from a root or from a database that was
            # just read, so a failing query is never proof its rows are gone.
            ancestors = _ancestors_of(item)
            for result in _query_data_source(client, item.id):
                is_data_source = result.get("object") == "data_source"
                queue.append(
                    _WalkItem(
                        "data_source" if is_data_source else "page",
                        _canonical_id(result["id"]),
                        item.root_id,
                        item.depth + 1,
                        prefetched=None if is_data_source else result,
                        ancestors=ancestors,
                        name=_data_source_name(result) if is_data_source else "",
                    )
                )
            continue

        # item.kind == "page"
        try:
            page = item.prefetched if item.prefetched is not None else client.get_page(item.id)
        except NotionAPIError as exc:
            if item.is_root or not _is_gone(exc):
                raise
            logger.warning("Notion: page %s is gone, forgetting it: %s", item.id, exc)
            continue

        if page.get("archived") or page.get("in_trash"):
            continue

        # The root is part of what was synced: a page that another selected root
        # reaches now keeps its last_edited_time but needs a new cognee_node_set.
        # So is where it sits: a page moved under another parent keeps its
        # last_edited_time but needs its new ancestors.
        seen = {
            "last_edited_time": page.get("last_edited_time"),
            "root_id": item.root_id,
            "ancestors": list(item.ancestors),
        }
        present_pages[item.id] = seen
        stats["pages_scanned"] += 1
        is_database_row = (page.get("parent") or {}).get("type") == "data_source_id"

        # A readable page whose children cannot be listed is not proof its subtree
        # is gone; abort rather than tombstone live descendants.
        blocks = list(_list_block_children(client, item.id))
        # Sub-pages can sit inside columns, toggles or synced blocks, not only at
        # the top level, so every run expands container blocks to find them.
        children_cache: dict[str, list[dict]] = {}
        discovered = _collect_nested_children(client, blocks, children_cache)

        if previous_pages.get(item.id) != seen:
            stats["pages_changed"] += 1
            yield {
                "id": item.id,
                "url": page.get("url"),
                "title": _page_title(page),
                "content": _render_page_content(
                    client, page, blocks, is_database_row, children_cache
                ),
                "_deleted": False,
                NODE_SET_COLUMN: [f"notion:{workspace_id}:{item.root_id}"],
                STRUCTURE_COLUMN: {"ancestors": list(item.ancestors)},
            }
        else:
            stats["pages_unchanged"] += 1

        ancestors = _ancestors_of(item)
        for block in discovered:
            block_type = block.get("type")
            if block_type == "child_page":
                queue.append(
                    _WalkItem(
                        "page",
                        _canonical_id(block["id"]),
                        item.root_id,
                        item.depth + 1,
                        ancestors=ancestors,
                    )
                )
            elif block_type == "child_database":
                queue.append(
                    _WalkItem(
                        "database",
                        _canonical_id(block["id"]),
                        item.root_id,
                        item.depth + 1,
                        ancestors=ancestors,
                    )
                )

    deleted_ids = sorted(set(previous_pages) - set(present_pages))
    for page_id in deleted_ids:
        stats["deleted"] += 1
        yield {"id": page_id, "_deleted": True}

    state["pages"] = present_pages
    logger.info(
        "Notion: synced %d page(s) (%d changed, %d unchanged), %d deletion(s).",
        stats["pages_scanned"],
        stats["pages_changed"],
        stats["pages_unchanged"],
        stats["deleted"],
    )


def _resolve_workspace_id(client: _NotionClient) -> str:
    response = client.get_self()
    bot = response.get("bot")
    workspace_id = bot.get("workspace_id") if isinstance(bot, dict) else None
    if not workspace_id:
        raise RuntimeError(
            "Could not resolve a Notion workspace id from users/me. Pass workspace_id "
            "explicitly to notion_source(...)."
        )
    return workspace_id


# ---------------------------------------------------------------------------
# Rendering (page title, properties, block content)
# ---------------------------------------------------------------------------
def _rich_text(rich_text: Any) -> str:
    """Concatenate the plain_text of a Notion rich_text array (mentions included)."""
    if not isinstance(rich_text, list):
        return ""
    return "".join(part.get("plain_text", "") for part in rich_text if isinstance(part, dict))


def _page_title(page: dict) -> str:
    for prop in (page.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return _rich_text(prop.get("title"))
    return ""


def _render_property_value(property_type: str, prop: dict) -> str:
    if property_type == "rich_text":
        return _rich_text(prop.get("rich_text"))
    if property_type == "select":
        value = prop.get("select")
        return value.get("name", "") if value else ""
    if property_type == "multi_select":
        return ", ".join(option.get("name", "") for option in prop.get("multi_select") or [])
    if property_type == "status":
        value = prop.get("status")
        return value.get("name", "") if value else ""
    if property_type == "date":
        value = prop.get("date")
        if not value:
            return ""
        start, end = value.get("start"), value.get("end")
        return f"{start} to {end}" if end else str(start or "")
    if property_type == "people":
        return ", ".join(
            person.get("name") or person.get("id", "") for person in prop.get("people") or []
        )
    if property_type == "number":
        number = prop.get("number")
        return "" if number is None else str(number)
    if property_type == "checkbox":
        return "Yes" if prop.get("checkbox") else "No"
    if property_type == "url":
        return prop.get("url") or ""
    if property_type == "email":
        return prop.get("email") or ""
    if property_type == "relation":
        return ", ".join(relation.get("id", "") for relation in prop.get("relation") or [])
    return ""


def _render_properties(properties: dict) -> list[str]:
    lines = []
    for name, prop in properties.items():
        if not isinstance(prop, dict) or prop.get("type") == "title":
            continue
        value = _render_property_value(prop.get("type"), prop)
        if value:
            lines.append(f"{name}: {value}")
    return lines


def _table_cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").splitlines())


def _render_table(client: _NotionClient, block: dict) -> str:
    rows = list(_list_block_children(client, block["id"]))
    lines = []
    for index, row in enumerate(rows):
        cells = (row.get("table_row") or {}).get("cells") or []
        cell_texts = [_table_cell(_rich_text(cell)) for cell in cells]
        lines.append("| " + " | ".join(cell_texts) + " |")
        if index == 0:
            lines.append("| " + " | ".join("---" for _ in cell_texts) + " |")
    return "\n".join(lines)


def _render_simple_block(block_type: str, payload: dict, text: str) -> str:
    if block_type in _HEADING_PREFIX:
        return f"{_HEADING_PREFIX[block_type]}{text}" if text else ""
    if block_type == "bulleted_list_item":
        return f"- {text}" if text else ""
    if block_type == "numbered_list_item":
        return f"1. {text}" if text else ""
    if block_type == "to_do":
        checked = "x" if payload.get("checked") else " "
        return f"- [{checked}] {text}"
    if block_type in ("quote", "callout"):
        return f"> {text}" if text else ""
    if block_type == "code":
        language = payload.get("language") or ""
        return f"```{language}\n{text}\n```"
    if block_type == "divider":
        return "---"
    if block_type == "equation":
        expression = payload.get("expression") or ""
        return f"$$ {expression} $$" if expression else ""
    if block_type in ("bookmark", "embed", "link_preview"):
        url = payload.get("url")
        if not url:
            return ""
        caption = _rich_text(payload.get("caption"))
        return f"[{caption or url}]({url})"
    if block_type == "link_to_page":
        target_type = payload.get("type")
        target_id = payload.get(target_type) if target_type else None
        return f"[Linked {target_type}: {target_id}]" if target_id else ""
    if block_type in ("file", "pdf", "image", "video"):
        caption = _rich_text(payload.get("caption"))
        name = payload.get("name") or caption or block_type
        return f"[{block_type}: {name}]"
    # paragraph and anything else render as plain rich text.
    return text


_PAGE_BLOCK_TYPES = frozenset({"child_page", "child_database"})


def _is_synced_copy(block: dict) -> bool:
    return block.get("type") == "synced_block" and bool(
        (block.get("synced_block") or {}).get("synced_from")
    )


def _collect_nested_children(
    client: _NotionClient, blocks: list[dict], cache: dict[str, list[dict]], depth: int = 0
) -> list[dict]:
    """Return every child_page/child_database block at any depth under ``blocks``.

    Children of container blocks (columns, toggles, synced blocks, callouts) are
    listed once and kept in ``cache`` so rendering does not fetch them again.
    Tables only hold rows, and a child page's own content belongs to that page.
    """
    found = []
    for block in blocks:
        block_type = block.get("type")
        if block_type in _PAGE_BLOCK_TYPES:
            found.append(block)
            continue
        if not block.get("has_children") or block_type == "table" or _is_synced_copy(block):
            continue
        if depth >= _MAX_DISCOVERY_DEPTH:
            raise NotionAPIError(
                0,
                f"Notion block {block['id']} is nested deeper than {_MAX_DISCOVERY_DEPTH} "
                "levels; aborting instead of forgetting the pages below it.",
            )
        children = list(_list_block_children(client, block["id"]))
        cache[block["id"]] = children
        found.extend(_collect_nested_children(client, children, cache, depth + 1))
    return found


def _render_block(
    client: _NotionClient, block: dict, depth: int, cache: dict[str, list[dict]] | None = None
) -> str:
    block_type = block.get("type")
    if not block_type:
        return ""
    if block_type == "child_page":
        return f"[Page: {(block.get('child_page') or {}).get('title', '')}]"
    if block_type == "child_database":
        return f"[Database: {(block.get('child_database') or {}).get('title', '')}]"
    if block_type == "table":
        return _render_table(client, block)

    payload = block.get(block_type) or {}
    text = _rich_text(payload.get("rich_text"))
    rendered = _render_simple_block(block_type, payload, text)

    if block.get("has_children") and depth < _MAX_RENDER_DEPTH:
        if cache is not None and block["id"] in cache:
            nested = _render_blocks_list(client, cache[block["id"]], depth + 1, cache)
        elif _is_synced_copy(block):
            # The original may sit on a page this integration cannot read; the
            # copy (or any block inside it) then renders empty instead of
            # failing the whole page.
            try:
                children = list(_list_block_children(client, block["id"]))
                nested = _render_blocks_list(client, children, depth + 1, cache)
            except NotionAPIError as exc:
                if not _is_gone(exc):
                    raise
                nested = ""
        else:
            children = list(_list_block_children(client, block["id"]))
            nested = _render_blocks_list(client, children, depth + 1, cache)
        if nested:
            rendered = f"{rendered}\n{nested}" if rendered else nested

    return rendered


def _render_blocks_list(
    client: _NotionClient,
    blocks: list[dict],
    depth: int,
    cache: dict[str, list[dict]] | None = None,
) -> str:
    if depth > _MAX_RENDER_DEPTH:
        return ""
    lines = [_render_block(client, block, depth, cache) for block in blocks]
    return "\n".join(line for line in lines if line)


def _render_page_content(
    client: _NotionClient,
    page: dict,
    blocks: list[dict],
    is_database_row: bool,
    cache: dict[str, list[dict]] | None = None,
) -> str:
    parts = []
    if is_database_row:
        property_lines = _render_properties(page.get("properties") or {})
        if property_lines:
            parts.append("\n".join(property_lines))
    body = _render_blocks_list(client, blocks, 0, cache)
    if body:
        parts.append(body)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def notion_source(
    token: str | None = None,
    root_page_ids: list[str] | None = None,
    root_data_source_ids: list[str] | None = None,
    workspace_id: str | None = None,
    resource_name: str = "notion_pages",
    check_active: Callable[[], None] | None = None,
    http_client: Any = None,
):
    """Return a ``dlt`` resource yielding one row per in-scope Notion page.

    Hand the result to ``cognee.remember(...)`` with
    ``write_disposition="merge"`` and ``primary_key="id"``.

    Args:
        token: Notion integration token. Falls back to ``NOTION_API_KEY``.
        root_page_ids: Page ids to walk (plus everything nested under them).
        root_data_source_ids: Data source ids to sync, plus everything nested
            under their rows. A database id also works here, it is resolved
            into that database's data sources.
        workspace_id: Stable workspace id used in ``cognee_node_set``. When
            omitted it is resolved once from the integration's bot user
            (``users/me``); the sync fails rather than emitting untagged rows
            if that cannot be determined.
        resource_name: Stable dlt resource name, also the staging table and the
            key of the incremental state. Every Notion source syncing into one
            dataset must have its own name: two sources sharing a name share
            state, so each run tombstones and forgets the other's pages. A
            source whose name changes starts over, and documents it synced
            under the old name are not cleaned up.
        check_active: Optional host authorization checkpoint during extraction.
        http_client: Pre-built ``httpx`` client (mainly a test-injection
            point); when omitted one is built for this sync.
    """
    import dlt
    import httpx

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Notion sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )

    resolved_token = token or os.getenv("NOTION_API_KEY")
    if not resolved_token:
        raise ValueError(f"Notion integration token required: {_EXTRA_HINT}")

    resolved_root_page_ids = list(root_page_ids or [])
    resolved_root_data_source_ids = list(root_data_source_ids or [])
    if not resolved_root_page_ids and not resolved_root_data_source_ids:
        raise ValueError(
            "notion_source requires at least one of root_page_ids or root_data_source_ids."
        )

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        # cognee_node_set and cognee_structure need no json hint here:
        # ingest_dlt_source applies DOCUMENT_COLUMN_HINTS to every document
        # source, merged with this one.
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def notion_pages():
        owned_client = None if http_client is not None else httpx.Client(timeout=30.0)
        try:
            client = _NotionClient(http_client or owned_client, resolved_token)
            resolved_workspace_id = _canonical_id(workspace_id or _resolve_workspace_id(client))
            state = dlt.current.resource_state()
            previous_pages = {
                _canonical_id(page_id): seen for page_id, seen in state.get("pages", {}).items()
            }
            previous_workspace_id = state.get("workspace_id")
            if previous_pages and previous_workspace_id not in (None, resolved_workspace_id):
                # Same resource_name, different workspace: walking on would
                # tombstone every page of the other workspace.
                raise ValueError(
                    f"Notion resource {resource_name!r} already holds pages from another "
                    "workspace in this dataset. Give each Notion source its own "
                    "resource_name."
                )
            state["workspace_id"] = resolved_workspace_id
            yield from dlt_utils.guarded_rows(
                _iter_rows(
                    client,
                    resolved_root_page_ids,
                    resolved_root_data_source_ids,
                    resolved_workspace_id,
                    previous_pages,
                    state,
                    stats,
                    check_active,
                ),
                check_active,
            )
        finally:
            if owned_client is not None:
                owned_client.close()

    resource = notion_pages()
    # Opt into the document ingestion path: each page row (id/title/content/
    # url, plus cognee_node_set) becomes a text document that flows through
    # normal cognify. resolve_dlt_sources reads this marker; it never imports
    # this connector.
    setattr(resource, DOCUMENT_SOURCE_ATTR, "notion")
    setattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR, resource_name)
    # Host-readable diagnostics contain counts only, never page titles or content.
    resource.cognee_sync_stats = stats
    return resource
