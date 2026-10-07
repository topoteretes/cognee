"""Unit tests for the Notion dlt source's tree walk, rendering, and sync logic.

No network: the Notion API is faked with ``httpx.MockTransport``. These tests
exercise ``_iter_rows``/``_resolve_roots``/rendering directly, the same shape
as ``test_google_drive_source.py``.
"""

import json

import httpx
import pytest

from cognee.tasks.ingestion.connectors import notion as notion_module
from cognee.tasks.ingestion.connectors.notion import (
    NotionAPIError,
    _iter_rows,
    _NotionClient,
    _resolve_workspace_id,
    notion_source,
)
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN

WORKSPACE_ID = "11111111-1111-1111-1111-111111111111"
ROOT_PAGE_ID = "page-root"
CHILD_PAGE_ID = "page-child"
DB_BLOCK_ID = "db-block"
DATA_SOURCE_ID = "ds-1"
ROW_PAGE_ID = "row-1"


def _title_prop(text):
    return {"type": "title", "title": [{"plain_text": text}]}


def _rich_text_block(block_id, block_type, text, *, has_children=False):
    return {
        "id": block_id,
        "type": block_type,
        "has_children": has_children,
        block_type: {"rich_text": [{"plain_text": text}]},
    }


def _page_object(
    page_id, title, *, parent, last_edited="2024-01-01T00:00:00.000Z", properties=None
):
    props = {"Name": _title_prop(title)}
    if properties:
        props.update(properties)
    return {
        "object": "page",
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "last_edited_time": last_edited,
        "archived": False,
        "in_trash": False,
        "parent": parent,
        "properties": props,
    }


class FakeNotion:
    """In-memory Notion backend driving an httpx.MockTransport."""

    def __init__(self):
        self.pages: dict[str, dict] = {}
        self.blocks: dict[str, list[dict]] = {}
        self.databases: dict[str, dict] = {}
        self.data_sources: dict[str, dict] = {}
        self.data_source_rows: dict[str, list[str]] = {}
        self.gone: set[str] = set()
        self.fail_next: dict[str, int] = {}
        self.requests: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1/")
        self.requests.append((request.method, path))

        if path == "users/me":
            return httpx.Response(200, json={"bot": {"workspace_id": WORKSPACE_ID}})

        for prefix, store in (
            ("pages/", self.pages),
            ("databases/", self.databases),
            ("data_sources/", self.data_sources),
        ):
            if path.startswith(prefix) and not path.endswith(("/children", "/query")):
                object_id = path[len(prefix) :]
                return self._object_response(object_id, store)

        if path.startswith("blocks/") and path.endswith("/children"):
            block_id = path[len("blocks/") : -len("/children")]
            return self._object_response(block_id, self.blocks, wrap_list=True)

        if path.startswith("data_sources/") and path.endswith("/query"):
            data_source_id = path[len("data_sources/") : -len("/query")]
            if data_source_id in self.gone:
                return httpx.Response(404, json={"message": "not found"})
            ids = self.data_source_rows.get(data_source_id, [])
            return httpx.Response(
                200,
                json={
                    "results": [self.pages[pid] for pid in ids],
                    "has_more": False,
                    "next_cursor": None,
                },
            )

        return httpx.Response(404, json={"message": f"unhandled path {path}"})

    def _object_response(self, object_id, store, *, wrap_list=False):
        if self.fail_next.get(object_id, 0) > 0:
            self.fail_next[object_id] -= 1
            return httpx.Response(
                429, headers={"Retry-After": "0"}, json={"message": "rate limited"}
            )
        if object_id in self.gone:
            return httpx.Response(404, json={"message": "not found"})
        if object_id not in store:
            return httpx.Response(404, json={"message": "not found"})
        if wrap_list:
            return httpx.Response(
                200, json={"results": store[object_id], "has_more": False, "next_cursor": None}
            )
        return httpx.Response(200, json=store[object_id])


def _client(fake: FakeNotion) -> _NotionClient:
    transport = httpx.MockTransport(fake.handler)
    http_client = httpx.Client(transport=transport)
    return _NotionClient(http_client, "secret_token")


def _basic_tree():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    fake.blocks[ROOT_PAGE_ID] = [
        _rich_text_block("b1", "paragraph", "hello root"),
        {
            "id": CHILD_PAGE_ID,
            "type": "child_page",
            "has_children": False,
            "child_page": {"title": "Child"},
        },
    ]
    fake.pages[CHILD_PAGE_ID] = _page_object(
        CHILD_PAGE_ID, "Child", parent={"type": "page_id", "page_id": ROOT_PAGE_ID}
    )
    fake.blocks[CHILD_PAGE_ID] = [_rich_text_block("b2", "paragraph", "hello child")]
    return fake


def test_walk_tags_every_page_with_its_root_node_set():
    fake = _basic_tree()
    client = _client(fake)
    stats = {}
    rows = list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, stats))

    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {ROOT_PAGE_ID, CHILD_PAGE_ID}
    for row in rows:
        assert row[NODE_SET_COLUMN] == [f"notion:{WORKSPACE_ID}:{ROOT_PAGE_ID}"]
    assert "hello root" in by_id[ROOT_PAGE_ID]["content"]
    assert stats["pages_scanned"] == 2
    assert stats["pages_changed"] == 2


def test_database_row_gets_its_properties_rendered():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    fake.blocks[ROOT_PAGE_ID] = [
        {
            "id": DB_BLOCK_ID,
            "type": "child_database",
            "has_children": False,
            "child_database": {"title": "Tasks"},
        }
    ]
    fake.databases[DB_BLOCK_ID] = {"data_sources": [{"id": DATA_SOURCE_ID, "name": "Tasks"}]}
    fake.data_source_rows[DATA_SOURCE_ID] = [ROW_PAGE_ID]
    fake.pages[ROW_PAGE_ID] = _page_object(
        ROW_PAGE_ID,
        "Row one",
        parent={
            "type": "data_source_id",
            "data_source_id": DATA_SOURCE_ID,
            "database_id": DB_BLOCK_ID,
        },
        properties={
            "Status": {"type": "status", "status": {"name": "Done"}},
            "Tags": {"type": "multi_select", "multi_select": [{"name": "a"}, {"name": "b"}]},
            "Done": {"type": "checkbox", "checkbox": True},
        },
    )
    fake.blocks[ROW_PAGE_ID] = [_rich_text_block("b3", "paragraph", "row body")]

    client = _client(fake)
    rows = list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    row = next(r for r in rows if r["id"] == ROW_PAGE_ID)

    assert "Status: Done" in row["content"]
    assert "Tags: a, b" in row["content"]
    assert "Done: Yes" in row["content"]
    assert "row body" in row["content"]


def test_table_and_toggle_rendering():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    table_id = "table-1"
    toggle_id = "toggle-1"
    fake.blocks[ROOT_PAGE_ID] = [
        {
            "id": table_id,
            "type": "table",
            "has_children": True,
            "table": {"has_column_header": True},
        },
        {
            "id": toggle_id,
            "type": "toggle",
            "has_children": True,
            "toggle": {"rich_text": [{"plain_text": "More"}]},
        },
    ]
    fake.blocks[table_id] = [
        {
            "id": "r1",
            "type": "table_row",
            "has_children": False,
            "table_row": {"cells": [[{"plain_text": "A"}], [{"plain_text": "B"}]]},
        },
        {
            "id": "r2",
            "type": "table_row",
            "has_children": False,
            "table_row": {"cells": [[{"plain_text": "1"}], [{"plain_text": "2"}]]},
        },
    ]
    fake.blocks[toggle_id] = [_rich_text_block("t1", "paragraph", "hidden text")]

    client = _client(fake)
    rows = list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    content = rows[0]["content"]
    assert "| A | B |" in content
    assert "| --- | --- |" in content
    assert "| 1 | 2 |" in content
    assert "More" in content
    assert "hidden text" in content


def test_container_children_are_listed_once_per_run():
    # Toggle children are listed on every run (sub-pages can hide there) and the
    # renderer reuses that listing, so a changed page does not fetch them twice.
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    toggle_id = "toggle-1"
    fake.blocks[ROOT_PAGE_ID] = [
        {
            "id": toggle_id,
            "type": "toggle",
            "has_children": True,
            "toggle": {"rich_text": [{"plain_text": "More"}]},
        }
    ]
    fake.blocks[toggle_id] = [_rich_text_block("t1", "paragraph", "hidden text")]

    client = _client(fake)
    state: dict = {}
    first = list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, state, {}))
    assert "hidden text" in first[0]["content"]
    assert fake.requests.count(("GET", f"blocks/{toggle_id}/children")) == 1

    previous_pages = dict(state["pages"])
    fake.requests.clear()
    second = list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, previous_pages, {}, {}))
    assert second == []
    assert fake.requests.count(("GET", f"blocks/{toggle_id}/children")) == 1


def test_sub_pages_inside_columns_are_synced_and_kept():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    fake.blocks[ROOT_PAGE_ID] = [
        {"id": "cols", "type": "column_list", "has_children": True, "column_list": {}}
    ]
    fake.blocks["cols"] = [{"id": "col1", "type": "column", "has_children": True, "column": {}}]
    fake.blocks["col1"] = [
        {"id": CHILD_PAGE_ID, "type": "child_page", "has_children": True, "child_page": {}}
    ]
    fake.pages[CHILD_PAGE_ID] = _page_object(
        CHILD_PAGE_ID, "Child", parent={"type": "block_id", "block_id": "col1"}
    )
    fake.blocks[CHILD_PAGE_ID] = [_rich_text_block("b2", "paragraph", "hello child")]

    state: dict = {}
    first = list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, state, {}))
    assert {row["id"] for row in first} == {ROOT_PAGE_ID, CHILD_PAGE_ID}

    second = list(
        _iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, dict(state["pages"]), {}, {})
    )
    assert not any(row.get("_deleted") for row in second)


def test_vanished_page_is_tombstoned():
    fake = _basic_tree()
    client = _client(fake)
    state: dict = {}
    list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, state, {}))
    previous_pages = dict(state["pages"])

    # The child page is no longer linked from root's blocks.
    fake.blocks[ROOT_PAGE_ID] = [_rich_text_block("b1", "paragraph", "hello root")]
    del fake.pages[CHILD_PAGE_ID]

    rows = list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, previous_pages, {}, {}))
    tombstones = [r for r in rows if r.get("_deleted")]
    assert tombstones == [{"id": CHILD_PAGE_ID, "_deleted": True}]


def test_a_page_reached_from_another_root_is_synced_again_under_it():
    """Narrowing the selection to the child page moves it to its own root.

    Its last_edited_time is unchanged, so only the root stored in state shows
    that the row needs a new cognee_node_set.
    """
    fake = _basic_tree()
    state: dict = {}
    list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, state, {}))

    rows = list(
        _iter_rows(_client(fake), [CHILD_PAGE_ID], [], WORKSPACE_ID, dict(state["pages"]), {}, {})
    )
    assert rows == [
        {
            "id": CHILD_PAGE_ID,
            "url": f"https://notion.so/{CHILD_PAGE_ID}",
            "title": "Child",
            "content": "hello child",
            "_deleted": False,
            NODE_SET_COLUMN: [f"notion:{WORKSPACE_ID}:{CHILD_PAGE_ID}"],
        },
        {"id": ROOT_PAGE_ID, "_deleted": True},
    ]


@pytest.mark.parametrize(
    "roots",
    [
        [ROOT_PAGE_ID, CHILD_PAGE_ID],
        # The child root listed last is what a depth-first walk would get wrong.
        [CHILD_PAGE_ID, ROOT_PAGE_ID],
    ],
)
def test_a_page_under_two_selected_roots_belongs_to_the_nearest(roots):
    fake = _basic_tree()
    rows = list(_iter_rows(_client(fake), roots, [], WORKSPACE_ID, {}, {}, {}))
    assert {row["id"]: row[NODE_SET_COLUMN] for row in rows} == {
        ROOT_PAGE_ID: [f"notion:{WORKSPACE_ID}:{ROOT_PAGE_ID}"],
        CHILD_PAGE_ID: [f"notion:{WORKSPACE_ID}:{CHILD_PAGE_ID}"],
    }


@pytest.mark.parametrize("flag", ["in_trash", "archived"])
def test_a_trashed_or_archived_page_is_tombstoned(flag):
    fake = _basic_tree()
    state: dict = {}
    list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, state, {}))

    fake.pages[CHILD_PAGE_ID][flag] = True
    rows = list(
        _iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, dict(state["pages"]), {}, {})
    )
    assert rows == [{"id": CHILD_PAGE_ID, "_deleted": True}]


def _paginate(fake, path, results):
    """Serve ``results`` for ``path`` one per response, following start_cursor.

    A walk that ignores the cursor asks for more pages than exist and gets a
    400, so the test fails instead of looping.
    """
    original = fake.handler
    served = []

    def handler(request):
        if request.url.path.removeprefix("/v1/") != path:
            return original(request)
        served.append(request)
        if len(served) > len(results):
            return httpx.Response(400, json={"message": "cursor not followed"})
        if request.method == "POST":
            cursor = json.loads(request.content).get("start_cursor")
        else:
            cursor = request.url.params.get("start_cursor")
        index = int(cursor or 0)
        more = index + 1 < len(results)
        return httpx.Response(
            200,
            json={
                "results": [results[index]],
                "has_more": more,
                "next_cursor": str(index + 1) if more else None,
            },
        )

    fake.handler = handler


def test_block_listing_follows_the_cursor_to_every_page():
    fake = _basic_tree()
    _paginate(fake, f"blocks/{ROOT_PAGE_ID}/children", fake.blocks[ROOT_PAGE_ID])
    rows = list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    assert {row["id"] for row in rows} == {ROOT_PAGE_ID, CHILD_PAGE_ID}


def test_data_source_query_follows_the_cursor_to_every_page():
    fake = _tree_with_database()
    second_row = "row-2"
    fake.pages[second_row] = dict(fake.pages[ROW_PAGE_ID], id=second_row)
    fake.blocks[second_row] = []
    _paginate(
        fake,
        f"data_sources/{DATA_SOURCE_ID}/query",
        [fake.pages[ROW_PAGE_ID], fake.pages[second_row]],
    )
    rows = list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    assert {ROW_PAGE_ID, second_row} <= {row["id"] for row in rows}


def test_an_incomplete_data_source_query_aborts_without_tombstones():
    """Notion stops a query at 10,000 rows with has_more false; the rest must not be forgotten."""
    fake = _tree_with_database()
    original = fake.handler

    def handler(request):
        if request.url.path.endswith(f"data_sources/{DATA_SOURCE_ID}/query"):
            return httpx.Response(
                200,
                json={
                    "results": [],
                    "has_more": False,
                    "next_cursor": None,
                    "request_status": {"type": "incomplete"},
                },
            )
        return original(request)

    fake.handler = handler
    previous = {ROOT_PAGE_ID: {}, CHILD_PAGE_ID: {}, ROW_PAGE_ID: {}}
    state: dict = {}
    rows = []
    with pytest.raises(NotionAPIError, match="incomplete"):
        rows.extend(
            _iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, previous, state, {})
        )
    assert not any(row.get("_deleted") for row in rows)
    assert state == {}


def test_a_wiki_data_source_in_query_results_is_walked_as_a_data_source():
    """A wiki's query returns its child data sources next to its pages."""
    fake = _tree_with_database()
    nested_source, nested_row = "ds-2", "row-2"
    fake.data_source_rows[DATA_SOURCE_ID] = []
    fake.data_source_rows[nested_source] = [nested_row]
    fake.pages[nested_row] = dict(fake.pages[ROW_PAGE_ID], id=nested_row)
    fake.blocks[nested_row] = []
    original = fake.handler

    def handler(request):
        if request.url.path.endswith(f"data_sources/{DATA_SOURCE_ID}/query"):
            return httpx.Response(
                200,
                json={
                    "results": [{"object": "data_source", "id": nested_source}],
                    "has_more": False,
                    "next_cursor": None,
                },
            )
        return original(request)

    fake.handler = handler
    rows = list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    assert nested_row in {row["id"] for row in rows}
    assert ("GET", f"blocks/{nested_source}/children") not in fake.requests


def test_transient_error_aborts_without_tombstones(monkeypatch):
    fake = _basic_tree()
    client = _client(fake)
    state: dict = {}
    list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, state, {}))
    previous_pages = dict(state["pages"])

    # The child page now 500s on every attempt: retries are exhausted, and the
    # generator must raise instead of yielding tombstones for it.
    def always_500(request: httpx.Request) -> httpx.Response:
        if "pages/" + CHILD_PAGE_ID in str(request.url):
            return httpx.Response(500, json={"message": "boom"})
        return fake.handler(request)

    client._http = httpx.Client(transport=httpx.MockTransport(always_500))
    monkeypatch.setattr(notion_module, "_MAX_RETRIES", 1)
    with pytest.raises(NotionAPIError):
        list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, previous_pages, {}, {}))


def test_gone_child_page_is_tombstoned_not_raised():
    fake = _basic_tree()
    client = _client(fake)
    state: dict = {}
    list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, state, {}))
    previous_pages = dict(state["pages"])

    fake.gone.add(CHILD_PAGE_ID)
    rows = list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, previous_pages, {}, {}))
    assert {"id": CHILD_PAGE_ID, "_deleted": True} in rows


def test_explicit_root_gone_raises():
    fake = FakeNotion()
    client = _client(fake)
    with pytest.raises(NotionAPIError):
        list(_iter_rows(client, ["missing-root"], [], WORKSPACE_ID, {}, {}, {}))


def test_429_is_retried_then_succeeds():
    fake = _basic_tree()
    fake.fail_next[ROOT_PAGE_ID] = 2
    client = _client(fake)
    rows = list(_iter_rows(client, [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    assert any(r["id"] == ROOT_PAGE_ID for r in rows if not r.get("_deleted"))


def test_refuses_core_without_document_row_contract(monkeypatch):
    pytest.importorskip("dlt")
    monkeypatch.setattr(notion_module.dlt_utils, "DOCUMENT_SYNC_VERSION", 1)
    with pytest.raises(RuntimeError, match="DOCUMENT_SYNC_VERSION"):
        notion_source(token="secret", root_page_ids=["x"])


def test_workspace_id_resolution_failure_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"bot": {}})

    client = _NotionClient(httpx.Client(transport=httpx.MockTransport(handler)), "secret")
    with pytest.raises(RuntimeError, match="workspace"):
        _resolve_workspace_id(client)


def test_requires_a_root():
    pytest.importorskip("dlt")
    with pytest.raises(ValueError, match="root_page_ids"):
        notion_source(token="secret")


def test_requires_a_token(monkeypatch):
    pytest.importorskip("dlt")
    monkeypatch.delenv("NOTION_API_KEY", raising=False)
    with pytest.raises(ValueError, match="token"):
        notion_source(root_page_ids=["x"])


def test_moving_database_root_resolves_data_sources():
    fake = FakeNotion()
    fake.databases[DB_BLOCK_ID] = {"data_sources": [{"id": DATA_SOURCE_ID, "name": "Tasks"}]}
    fake.data_source_rows[DATA_SOURCE_ID] = [ROW_PAGE_ID]
    fake.pages[ROW_PAGE_ID] = _page_object(
        ROW_PAGE_ID,
        "Row one",
        parent={
            "type": "data_source_id",
            "data_source_id": DATA_SOURCE_ID,
            "database_id": DB_BLOCK_ID,
        },
    )
    fake.blocks[ROW_PAGE_ID] = []
    client = _client(fake)

    rows = list(_iter_rows(client, [], [DB_BLOCK_ID], WORKSPACE_ID, {}, {}, {}))
    assert rows[0][NODE_SET_COLUMN] == [f"notion:{WORKSPACE_ID}:{DB_BLOCK_ID}"]


def test_unreadable_children_of_a_readable_page_abort_without_tombstones():
    fake = _basic_tree()
    grandchild = "page-grandchild"
    fake.blocks[CHILD_PAGE_ID].append(
        {"id": grandchild, "type": "child_page", "has_children": False, "child_page": {}}
    )
    fake.pages[grandchild] = _page_object(
        grandchild, "Grandchild", parent={"type": "page_id", "page_id": CHILD_PAGE_ID}
    )
    fake.blocks[grandchild] = []
    previous = {ROOT_PAGE_ID: "2024-01-01T00:00:00.000Z", CHILD_PAGE_ID: "x", grandchild: "y"}
    original = fake.handler

    def handler(request):
        if request.url.path.endswith(f"blocks/{CHILD_PAGE_ID}/children"):
            return httpx.Response(403, json={"message": "restricted"})
        return original(request)

    fake.handler = handler
    state: dict = {}
    rows = []
    with pytest.raises(NotionAPIError):
        rows.extend(
            _iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, previous, state, {})
        )
    assert not any(row.get("_deleted") for row in rows)
    assert state == {}


def test_dashed_and_undashed_root_ids_are_one_page():
    page_id = "12345678-1234-1234-1234-123456789abc"
    fake = FakeNotion()
    fake.pages[page_id] = _page_object(
        page_id, "P", parent={"type": "workspace", "workspace": True}
    )
    fake.blocks[page_id] = []
    rows = list(
        _iter_rows(_client(fake), [page_id.replace("-", ""), page_id], [], WORKSPACE_ID, {}, {}, {})
    )
    assert [row["id"] for row in rows] == [page_id]
    assert rows[0][NODE_SET_COLUMN] == [f"notion:{WORKSPACE_ID}:{page_id}"]


def test_has_more_without_a_cursor_aborts_instead_of_tombstoning():
    fake = _basic_tree()
    original = fake.handler

    def handler(request):
        if request.url.path.endswith(f"blocks/{ROOT_PAGE_ID}/children"):
            return httpx.Response(200, json={"results": [], "has_more": True, "next_cursor": None})
        return original(request)

    fake.handler = handler
    previous = {ROOT_PAGE_ID: "2024-01-01T00:00:00.000Z", CHILD_PAGE_ID: "x"}
    with pytest.raises(NotionAPIError):
        list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, previous, {}, {}))


def test_check_active_runs_for_every_walked_page_even_when_nothing_changed():
    fake = _basic_tree()
    unchanged = {"last_edited_time": "2024-01-01T00:00:00.000Z", "root_id": ROOT_PAGE_ID}
    previous = {ROOT_PAGE_ID: unchanged, CHILD_PAGE_ID: unchanged}
    calls = []
    rows = list(
        _iter_rows(
            _client(fake),
            [ROOT_PAGE_ID],
            [],
            WORKSPACE_ID,
            previous,
            {},
            {},
            lambda: calls.append(1),
        )
    )
    assert rows == []
    assert len(calls) == 2


def test_table_cells_with_pipes_and_newlines_keep_the_table_shape():
    fake = FakeNotion()
    fake.blocks["table-1"] = [
        {
            "id": "r1",
            "type": "table_row",
            "table_row": {"cells": [[{"plain_text": "a|b"}], [{"plain_text": "x"}]]},
        },
        {
            "id": "r2",
            "type": "table_row",
            "table_row": {"cells": [[{"plain_text": "line1\nline2"}], [{"plain_text": "y"}]]},
        },
    ]
    rendered = notion_module._render_table(_client(fake), {"id": "table-1"})
    assert rendered.splitlines() == [
        "| a\\|b | x |",
        "| --- | --- |",
        "| line1 line2 | y |",
    ]


def test_owned_http_client_is_closed_after_the_sync(monkeypatch):
    fake = _basic_tree()
    closed = []

    class TrackingClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            super().__init__(transport=httpx.MockTransport(fake.handler))

        def close(self):
            closed.append(True)
            super().close()

    monkeypatch.setattr(httpx, "Client", TrackingClient)
    resource = notion_source(
        token="secret", root_page_ids=[ROOT_PAGE_ID], workspace_id=WORKSPACE_ID
    )
    assert closed == []
    rows = list(resource)
    assert {row["id"] for row in rows} == {ROOT_PAGE_ID, CHILD_PAGE_ID}
    assert closed == [True]


def _tree_with_database():
    fake = _basic_tree()
    fake.blocks[ROOT_PAGE_ID].append(
        {"id": DB_BLOCK_ID, "type": "child_database", "has_children": False, "child_database": {}}
    )
    fake.databases[DB_BLOCK_ID] = {"data_sources": [{"id": DATA_SOURCE_ID, "name": "Tasks"}]}
    fake.data_source_rows[DATA_SOURCE_ID] = [ROW_PAGE_ID]
    fake.pages[ROW_PAGE_ID] = _page_object(
        ROW_PAGE_ID,
        "Row",
        parent={
            "type": "data_source_id",
            "data_source_id": DATA_SOURCE_ID,
            "database_id": DB_BLOCK_ID,
        },
    )
    fake.blocks[ROW_PAGE_ID] = []
    return fake


def test_failing_query_of_a_readable_database_aborts_without_tombstones():
    fake = _tree_with_database()
    fake.gone.add(DATA_SOURCE_ID)
    previous = {ROOT_PAGE_ID: "x", CHILD_PAGE_ID: "x", ROW_PAGE_ID: "x"}
    state: dict = {}
    rows = []
    with pytest.raises(NotionAPIError):
        rows.extend(
            _iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, previous, state, {})
        )
    assert not any(row.get("_deleted") for row in rows)
    assert state == {}


def test_query_has_more_without_a_cursor_aborts():
    fake = _tree_with_database()
    original = fake.handler

    def handler(request):
        if request.url.path.endswith(f"data_sources/{DATA_SOURCE_ID}/query"):
            return httpx.Response(200, json={"results": [], "has_more": True, "next_cursor": None})
        return original(request)

    fake.handler = handler
    with pytest.raises(NotionAPIError):
        list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))


def test_child_page_and_query_result_ids_are_canonical():
    root = "11111111-2222-3333-4444-555555555555"
    child = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    data_source = "99999999-8888-7777-6666-555555555555"
    row = "12121212-3434-5656-7878-909090909090"
    fake = FakeNotion()
    fake.pages[root] = _page_object(root, "Root", parent={"type": "workspace", "workspace": True})
    fake.blocks[root] = [
        {
            "id": child.replace("-", ""),
            "type": "child_page",
            "has_children": False,
            "child_page": {},
        }
    ]
    fake.pages[child] = _page_object(child, "Child", parent={"type": "page_id", "page_id": root})
    fake.blocks[child] = []
    fake.data_sources[data_source] = {"id": data_source}
    fake.pages[row] = dict(
        _page_object(row, "Row", parent={"type": "data_source_id", "data_source_id": data_source}),
        id=row.replace("-", ""),
    )
    fake.data_source_rows[data_source] = [row]
    fake.blocks[row] = []

    rows = list(_iter_rows(_client(fake), [root], [data_source], WORKSPACE_ID, {}, {}, {}))
    assert sorted(r["id"] for r in rows) == sorted([root, child, row])


def test_workspace_id_is_canonical_in_node_set():
    fake = _basic_tree()
    resource = notion_source(
        token="secret",
        root_page_ids=[ROOT_PAGE_ID],
        workspace_id=WORKSPACE_ID.replace("-", ""),
        http_client=httpx.Client(transport=httpx.MockTransport(fake.handler)),
    )
    rows = list(resource)
    assert {tuple(r[NODE_SET_COLUMN]) for r in rows} == {(f"notion:{WORKSPACE_ID}:{ROOT_PAGE_ID}",)}


def _nested_toggles(fake, levels, leaf):
    parent = ROOT_PAGE_ID
    for level in range(levels):
        toggle = f"toggle-{level}"
        fake.blocks[parent] = [
            {"id": toggle, "type": "toggle", "has_children": True, "toggle": {"rich_text": []}}
        ]
        parent = toggle
    fake.blocks[parent] = [leaf]


def test_a_tree_nested_past_the_discovery_cap_aborts_instead_of_tombstoning():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    _nested_toggles(
        fake,
        notion_module._MAX_DISCOVERY_DEPTH + 1,
        {"id": CHILD_PAGE_ID, "type": "child_page", "has_children": False, "child_page": {}},
    )
    previous = {ROOT_PAGE_ID: "2024-01-01T00:00:00.000Z", CHILD_PAGE_ID: "x"}
    rows = []
    with pytest.raises(NotionAPIError, match="nested deeper"):
        rows.extend(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, previous, {}, {}))
    assert not any(row.get("_deleted") for row in rows)


def test_a_page_deeper_than_the_render_cap_is_still_kept():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    _nested_toggles(
        fake,
        notion_module._MAX_RENDER_DEPTH + 2,
        {"id": CHILD_PAGE_ID, "type": "child_page", "has_children": False, "child_page": {}},
    )
    fake.pages[CHILD_PAGE_ID] = _page_object(
        CHILD_PAGE_ID, "Deep", parent={"type": "block_id", "block_id": "toggle-11"}
    )
    fake.blocks[CHILD_PAGE_ID] = []
    rows = list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    assert {row["id"] for row in rows} == {ROOT_PAGE_ID, CHILD_PAGE_ID}


def test_a_synced_copy_of_an_unreadable_original_neither_aborts_nor_is_searched():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    fake.blocks[ROOT_PAGE_ID] = [
        _rich_text_block("b1", "paragraph", "before"),
        {
            "id": "copy-1",
            "type": "synced_block",
            "has_children": True,
            "synced_block": {"synced_from": {"type": "block_id", "block_id": "original-1"}},
        },
    ]
    fake.gone.add("copy-1")
    rows = list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    assert [row["id"] for row in rows] == [ROOT_PAGE_ID]
    assert "before" in rows[0]["content"]
    assert ("GET", "blocks/copy-1/children") in fake.requests


def test_resource_carries_its_pipeline_scope_and_document_marker():
    from cognee.tasks.ingestion.dlt_utils import PIPELINE_SCOPE_ATTR

    resource = notion_source(
        token="secret", root_page_ids=[ROOT_PAGE_ID], resource_name="notion_conn_1"
    )
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == "notion_conn_1"
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "notion"


def test_a_page_chain_past_the_walk_cap_aborts_instead_of_tombstoning(monkeypatch):
    monkeypatch.setattr(notion_module, "_MAX_WALK_DEPTH", 2)
    fake = FakeNotion()
    chain = [ROOT_PAGE_ID, "p1", "p2", "p3"]
    for index, page_id in enumerate(chain):
        parent = (
            {"type": "workspace", "workspace": True}
            if index == 0
            else {"type": "page_id", "page_id": chain[index - 1]}
        )
        fake.pages[page_id] = _page_object(page_id, page_id, parent=parent)
        fake.blocks[page_id] = (
            [
                {
                    "id": chain[index + 1],
                    "type": "child_page",
                    "has_children": False,
                    "child_page": {},
                }
            ]
            if index + 1 < len(chain)
            else []
        )
    previous = {page_id: "2024-01-01T00:00:00.000Z" for page_id in chain}
    rows = []
    with pytest.raises(NotionAPIError, match="nested deeper"):
        rows.extend(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, previous, {}, {}))
    assert not any(row.get("_deleted") for row in rows)


def test_an_original_synced_block_is_searched_for_sub_pages():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    fake.blocks[ROOT_PAGE_ID] = [
        {
            "id": "original-1",
            "type": "synced_block",
            "has_children": True,
            "synced_block": {"synced_from": None},
        }
    ]
    fake.blocks["original-1"] = [
        {"id": CHILD_PAGE_ID, "type": "child_page", "has_children": False, "child_page": {}}
    ]
    fake.pages[CHILD_PAGE_ID] = _page_object(
        CHILD_PAGE_ID, "Child", parent={"type": "block_id", "block_id": "original-1"}
    )
    fake.blocks[CHILD_PAGE_ID] = []
    rows = list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    assert {row["id"] for row in rows} == {ROOT_PAGE_ID, CHILD_PAGE_ID}


def test_a_readable_synced_copy_with_an_unreadable_block_inside_still_renders():
    fake = FakeNotion()
    fake.pages[ROOT_PAGE_ID] = _page_object(
        ROOT_PAGE_ID, "Root", parent={"type": "workspace", "workspace": True}
    )
    fake.blocks[ROOT_PAGE_ID] = [
        _rich_text_block("b1", "paragraph", "before"),
        {
            "id": "copy-1",
            "type": "synced_block",
            "has_children": True,
            "synced_block": {"synced_from": {"type": "block_id", "block_id": "original-1"}},
        },
    ]
    fake.blocks["copy-1"] = [_rich_text_block("inner", "toggle", "toggle", has_children=True)]
    fake.gone.add("inner")
    rows = list(_iter_rows(_client(fake), [ROOT_PAGE_ID], [], WORKSPACE_ID, {}, {}, {}))
    assert [row["id"] for row in rows] == [ROOT_PAGE_ID]
    assert "before" in rows[0]["content"]
