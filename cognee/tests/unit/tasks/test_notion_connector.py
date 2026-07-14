"""Unit tests for the Notion dlt connector.

Two layers, all runnable in CI without a live Notion token:

* DB-free tests for block→markdown rendering, page→row flattening, and the
  generic document DataItem tagging (``source="notion"``) that routes pages
  through normal cognify.
* dlt-pipeline tests (mocked notion-client, temp sqlite destination) covering
  the acceptance criteria: re-sync reflects edits, and archived/vanished pages
  drop out of the full-snapshot load (forget-on-delete).
"""

from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.tasks.ingestion.connectors.notion import (
    NOTION_SOURCE_NAME,
    _paginate,
    _page_title,
    _page_to_row,
    _render_block,
    _render_blocks,
    _rich_text,
)

# The row → document-DataItem mapping is generic and owned by the ingestion
# layer (any document source uses it), not the connector.
from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _rt(text):
    """A minimal Notion rich_text array with one plain_text span."""
    return [{"plain_text": text}]


def _block(block_type, text=None, **payload):
    body = dict(payload)
    if text is not None:
        body["rich_text"] = _rt(text)
    return {"type": block_type, block_type: body, "has_children": False}


def _page(page_id, last_edited, title, archived=False, url=None):
    return {
        "id": page_id,
        "last_edited_time": last_edited,
        "archived": archived,
        "url": url or f"https://notion.so/{page_id}",
        "properties": {"Name": {"type": "title", "title": _rt(title)}},
    }


class FakeNotionClient:
    """Stand-in for notion_client.Client backed by in-memory fixtures."""

    def __init__(self, pages, blocks=None):
        self._pages = pages
        self._blocks = blocks or {}
        self.blocks = SimpleNamespace(children=SimpleNamespace(list=self._blocks_list))
        self.pages = SimpleNamespace(retrieve=self._pages_retrieve)
        self.databases = SimpleNamespace(query=self._db_query)

    def search(self, **kwargs):
        # The real Notion API omits archived/trashed pages from search results,
        # so the connector detects deletion by their absence — mirror that here.
        return {"results": self._live_pages(), "has_more": False}

    def _db_query(self, **kwargs):
        return {"results": self._live_pages(), "has_more": False}

    def _live_pages(self):
        return [p for p in self._pages if not (p.get("archived") or p.get("in_trash"))]

    def _pages_retrieve(self, page_id=None):
        # pages.retrieve DOES return a trashed page (flagged), unlike search.
        return next(p for p in self._pages if p["id"] == page_id)

    def _blocks_list(self, block_id=None, start_cursor=None, **kwargs):
        return {"results": self._blocks.get(block_id, []), "has_more": False}


# ---------------------------------------------------------------------------
# Rendering (DB-free)
# ---------------------------------------------------------------------------


def test_rich_text_concatenates_plain_text():
    rich = [{"plain_text": "Hello "}, {"plain_text": "world"}]
    assert _rich_text(rich) == "Hello world"


def test_rich_text_handles_non_list():
    assert _rich_text(None) == ""


def test_render_block_covers_common_types():
    assert _render_block(_block("heading_1", "Title")) == "# Title"
    assert _render_block(_block("heading_2", "Sub")) == "## Sub"
    assert _render_block(_block("bulleted_list_item", "point")) == "- point"
    assert _render_block(_block("numbered_list_item", "step")) == "1. step"
    assert _render_block(_block("to_do", "task", checked=True)) == "- [x] task"
    assert _render_block(_block("to_do", "task", checked=False)) == "- [ ] task"
    assert _render_block(_block("paragraph", "prose")) == "prose"
    assert _render_block(_block("code", "print(1)", language="python")) == (
        "```python\nprint(1)\n```"
    )


def test_render_blocks_recurses_into_children():
    client = FakeNotionClient(
        pages=[],
        blocks={
            "root": [
                {
                    "type": "paragraph",
                    "paragraph": {"rich_text": _rt("parent")},
                    "has_children": True,
                    "id": "child",
                },
            ],
            "child": [_block("bulleted_list_item", "nested")],
        },
    )
    rendered = _render_blocks(client, "root")
    assert "parent" in rendered
    assert "- nested" in rendered


def test_render_block_unsupported_or_empty_returns_blank():
    assert _render_block({"type": None}) == ""  # missing type
    assert _render_block({"type": "divider", "divider": {}}) == ""  # no rich_text
    assert _render_block({"type": "image", "image": {}}) == ""  # unsupported, no text


def test_render_blocks_depth_guard_terminates():
    # Past the recursion cap the renderer bails out instead of looping.
    client = FakeNotionClient(pages=[], blocks={"x": [_block("paragraph", "deep")]})
    assert _render_blocks(client, "x", depth=11) == ""


def test_paginate_follows_cursor():
    pages = {
        None: {"results": [{"id": "a"}], "has_more": True, "next_cursor": "c1"},
        "c1": {"results": [{"id": "b"}], "has_more": False, "next_cursor": None},
    }
    seen = []

    def method(start_cursor=None, **kwargs):
        seen.append(start_cursor)
        return pages[start_cursor]

    assert [item["id"] for item in _paginate(method)] == ["a", "b"]
    assert seen == [None, "c1"]  # second call used the cursor from the first


def test_paginate_stops_on_null_cursor():
    # Contract violation (has_more=True but no next_cursor) must terminate, not spin.
    def method(start_cursor=None, **kwargs):
        return {"results": [{"id": "a"}], "has_more": True, "next_cursor": None}

    assert [item["id"] for item in _paginate(method)] == ["a"]


# ---------------------------------------------------------------------------
# Page → row / DataItem (DB-free)
# ---------------------------------------------------------------------------


def test_page_title_reads_title_property():
    assert _page_title(_page("p1", "2024-01-01T00:00:00.000Z", "My Page")) == "My Page"


def test_page_title_missing_returns_empty():
    assert _page_title({"properties": {}}) == ""
    assert _page_title({}) == ""


def test_page_to_row_flattens_page():
    client = FakeNotionClient(pages=[], blocks={"p1": [_block("paragraph", "body text")]})
    page = _page("p1", "2024-01-01T00:00:00.000Z", "My Page")

    row = _page_to_row(client, page)

    # Only identity/provenance + text are kept — no volatile last_edited_time,
    # so a metadata-only edit does not churn the content-hash data_id.
    assert row["id"] == "p1"
    assert row["title"] == "My Page"
    assert row["url"] == "https://notion.so/p1"
    assert "body text" in row["content"]
    assert "last_edited_time" not in row


def test_build_document_data_item_tags_source():
    row = SimpleNamespace(
        row_data={
            "id": "p1",
            "url": "https://notion.so/p1",
            "title": "My Page",
            "content": "body text",
        },
        content_hash="abc123",
    )
    data_id = uuid5(NAMESPACE_OID, "p1")

    item = _build_document_data_item(row, data_id, "notion")

    # source="notion" (not "dlt") is what routes the page through normal cognify.
    assert item.external_metadata["source"] == "notion"
    assert item.external_metadata["url"] == "https://notion.so/p1"
    assert item.external_metadata["external_id"] == "p1"
    assert item.data_id == data_id
    assert item.data.startswith("# My Page")
    assert "body text" in item.data


def test_notion_source_declares_document_marker():
    # resolve_dlt_sources routes on the document-source marker (not on this name),
    # but the tag it carries is the source name; keep it stable.
    from cognee.tasks.ingestion.connectors.notion import notion_source
    from cognee.tasks.ingestion.dlt_utils import document_source_tag

    source = notion_source(token="test-token")
    assert NOTION_SOURCE_NAME == "notion"
    assert document_source_tag(source) == "notion"


# ---------------------------------------------------------------------------
# dlt pipeline: full-snapshot sync + forget-on-delete (needs dlt + notion-client)
# ---------------------------------------------------------------------------


def _run_sync(dlt, tmp_path, monkeypatch, pages, blocks):
    """Run notion_source through a dlt pipeline into a temp sqlite destination."""
    from cognee.tasks.ingestion.connectors.notion import notion_source

    db_path = (tmp_path / "notion.db").as_posix()
    pipeline = dlt.pipeline(
        pipeline_name="notion_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{db_path}"),
        dataset_name="notion_ds",
        pipelines_dir=str(tmp_path / "state"),
    )
    pipeline.run(notion_source(client=FakeNotionClient(pages, blocks)))
    return pipeline


def _read_pages(pipeline):
    """Return {id: row-dict} for the notion_pages table.

    Reads positionally (the SELECT fixes the column order) since dlt's
    sqlalchemy cursor exposes a SQLAlchemy Result without DB-API ``description``.
    """
    with pipeline.sql_client() as client:
        with client.execute_query("SELECT id, title, content FROM notion_pages") as cursor:
            rows = cursor.fetchall()
    return {row[0]: {"id": row[0], "title": row[1], "content": row[2]} for row in rows}


@pytest.fixture
def dlt_mod():
    return pytest.importorskip("dlt")


@pytest.fixture(autouse=True)
def _need_notion_client():
    pytest.importorskip("notion_client")


def test_first_sync_loads_pages_with_rendered_content(dlt_mod, tmp_path, monkeypatch):
    pages = [_page("p1", "2024-01-01T00:00:00.000Z", "Alpha")]
    blocks = {"p1": [_block("paragraph", "alpha body")]}

    pipeline = _run_sync(dlt_mod, tmp_path, monkeypatch, pages, blocks)

    rows = _read_pages(pipeline)
    assert set(rows) == {"p1"}
    assert "alpha body" in rows["p1"]["content"]


def test_edit_is_reflected_on_resync(dlt_mod, tmp_path, monkeypatch):
    pages = [_page("p1", "2024-01-01T00:00:00.000Z", "Alpha")]
    _run_sync(dlt_mod, tmp_path, monkeypatch, pages, {"p1": [_block("paragraph", "v1")]})

    # Edit bumps last_edited_time so the incremental cursor picks it up.
    edited = [_page("p1", "2024-02-01T00:00:00.000Z", "Alpha")]
    pipeline = _run_sync(
        dlt_mod, tmp_path, monkeypatch, edited, {"p1": [_block("paragraph", "v2")]}
    )

    rows = _read_pages(pipeline)
    assert "v2" in rows["p1"]["content"]
    assert "v1" not in rows["p1"]["content"]


def test_archived_page_is_removed_on_resync(dlt_mod, tmp_path, monkeypatch):
    pages = [
        _page("p1", "2024-01-01T00:00:00.000Z", "Alpha"),
        _page("p2", "2024-01-01T00:00:00.000Z", "Beta"),
    ]
    blocks = {"p1": [_block("paragraph", "a")], "p2": [_block("paragraph", "b")]}
    _run_sync(dlt_mod, tmp_path, monkeypatch, pages, blocks)

    # p1 archived: the real API drops it from search results (the fake mirrors
    # that), and the connector skips it on the page_ids path — either way it is
    # absent from the replace load, so it falls out of staging.
    resync = [
        _page("p1", "2024-03-01T00:00:00.000Z", "Alpha", archived=True),
        _page("p2", "2024-01-01T00:00:00.000Z", "Beta"),
    ]
    pipeline = _run_sync(dlt_mod, tmp_path, monkeypatch, resync, blocks)

    rows = _read_pages(pipeline)
    # Absent from staging → orphan cleanup forgets it downstream.
    assert "p1" not in rows
    assert "p2" in rows


def test_vanished_page_is_removed_on_resync(dlt_mod, tmp_path, monkeypatch):
    # A page that simply disappears from the listing (deleted, or unshared from
    # the integration) must be forgotten too — not just explicitly-archived
    # pages. This is the case merge + a hard_delete hint could never catch,
    # because a vanished page is never yielded to flag it.
    pages = [
        _page("p1", "2024-01-01T00:00:00.000Z", "Alpha"),
        _page("p2", "2024-01-01T00:00:00.000Z", "Beta"),
    ]
    blocks = {"p1": [_block("paragraph", "a")], "p2": [_block("paragraph", "b")]}
    _run_sync(dlt_mod, tmp_path, monkeypatch, pages, blocks)

    # p1 no longer returned by the API at all.
    pipeline = _run_sync(
        dlt_mod, tmp_path, monkeypatch, [_page("p2", "2024-01-01T00:00:00.000Z", "Beta")], blocks
    )

    rows = _read_pages(pipeline)
    assert "p1" not in rows
    assert "p2" in rows
