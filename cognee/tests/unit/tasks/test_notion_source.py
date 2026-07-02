"""Unit tests for the Notion dlt connector.

Two layers, all runnable in CI without a live Notion token:

* DB-free tests for block→markdown rendering, page→row flattening, and the
  document DataItem tagging (``source="notion"``) that routes pages through
  normal cognify.
* dlt-pipeline tests (mocked notion-client, temp sqlite destination) covering
  the acceptance criteria: incremental re-sync reflects edits, and archiving a
  page removes it from staging (forget-on-delete) via the ``hard_delete`` hint.
"""

from types import SimpleNamespace
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.tasks.ingestion.notion_source import (
    NOTION_SOURCE_NAME,
    _build_notion_data_item,
    _page_title,
    _page_to_row,
    _render_block,
    _render_blocks,
    _rich_text,
    expand_notion_rows,
)


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
        return {"results": self._pages, "has_more": False}

    def _db_query(self, **kwargs):
        return {"results": self._pages, "has_more": False}

    def _pages_retrieve(self, page_id=None):
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


# ---------------------------------------------------------------------------
# Page → row / DataItem (DB-free)
# ---------------------------------------------------------------------------


def test_page_title_reads_title_property():
    assert _page_title(_page("p1", "2024-01-01T00:00:00.000Z", "My Page")) == "My Page"


def test_page_to_row_flattens_page_and_flags_archive():
    client = FakeNotionClient(pages=[], blocks={"p1": [_block("paragraph", "body text")]})
    page = _page("p1", "2024-01-01T00:00:00.000Z", "My Page")
    page["in_trash"] = True  # newer trash flag should also mark it archived

    row = _page_to_row(client, page)

    assert row["id"] == "p1"
    assert row["title"] == "My Page"
    assert row["last_edited_time"] == "2024-01-01T00:00:00.000Z"
    assert row["archived"] is True
    assert "body text" in row["content"]


def test_build_notion_data_item_tags_source_notion():
    row = SimpleNamespace(
        row_data={
            "id": "p1",
            "url": "https://notion.so/p1",
            "title": "My Page",
            "last_edited_time": "2024-01-01T00:00:00.000Z",
            "content": "body text",
        },
        content_hash="abc123",
    )
    data_id = uuid5(NAMESPACE_OID, "p1")

    item = _build_notion_data_item(row, data_id)

    # source="notion" (not "dlt") is what routes the page through normal cognify.
    assert item.external_metadata["source"] == "notion"
    assert item.external_metadata["notion_page_id"] == "p1"
    assert item.external_metadata["notion_url"] == "https://notion.so/p1"
    assert item.data_id == data_id
    assert item.data.startswith("# My Page")
    assert "body text" in item.data


@pytest.mark.asyncio
async def test_expand_notion_rows_builds_items_and_fresh_ids(monkeypatch):
    async def fake_unique_id(identifier, user):
        return uuid5(NAMESPACE_OID, identifier)

    monkeypatch.setattr("cognee.tasks.ingestion.notion_source.get_unique_data_id", fake_unique_id)

    rows = [
        SimpleNamespace(
            table_name="notion_pages",
            primary_key_value="p1",
            content_hash="h1",
            row_data={"id": "p1", "title": "A", "content": "one", "url": "u1"},
        ),
        SimpleNamespace(
            table_name="notion_pages",
            primary_key_value="p2",
            content_hash="h2",
            row_data={"id": "p2", "title": "B", "content": "two", "url": "u2"},
        ),
    ]

    items, fresh_ids = await expand_notion_rows(rows, user=SimpleNamespace(id="u"))

    assert len(items) == 2
    assert len(fresh_ids) == 2
    assert all(i.external_metadata["source"] == "notion" for i in items)
    # fresh_ids must match the items' data_ids so orphan cleanup keeps them.
    assert {i.data_id for i in items} == fresh_ids


def test_notion_source_name_constant():
    # resolve_dlt_sources routes on this name; keep it stable.
    assert NOTION_SOURCE_NAME == "notion"


# ---------------------------------------------------------------------------
# dlt pipeline: incremental + hard_delete (needs dlt + notion-client)
# ---------------------------------------------------------------------------


def _run_sync(dlt, tmp_path, monkeypatch, pages, blocks):
    """Run notion_source through a dlt pipeline into a temp sqlite destination."""
    import notion_client

    from cognee.tasks.ingestion.notion_source import notion_source

    monkeypatch.setattr(notion_client, "Client", lambda *a, **k: FakeNotionClient(pages, blocks))

    db_path = (tmp_path / "notion.db").as_posix()
    pipeline = dlt.pipeline(
        pipeline_name="notion_test",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{db_path}"),
        dataset_name="notion_ds",
        pipelines_dir=str(tmp_path / "state"),
    )
    pipeline.run(notion_source(token="test-token"))
    return pipeline


def _read_pages(pipeline):
    """Return {id: row-dict} for the notion_pages table.

    Reads positionally (the SELECT fixes the column order) since dlt's
    sqlalchemy cursor exposes a SQLAlchemy Result without DB-API ``description``.
    """
    with pipeline.sql_client() as client:
        with client.execute_query(
            "SELECT id, title, content, archived FROM notion_pages"
        ) as cursor:
            rows = cursor.fetchall()
    return {
        row[0]: {"id": row[0], "title": row[1], "content": row[2], "archived": row[3]}
        for row in rows
    }


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

    # p1 archived (with a bumped timestamp so it comes through the window).
    resync = [
        _page("p1", "2024-03-01T00:00:00.000Z", "Alpha", archived=True),
        _page("p2", "2024-01-01T00:00:00.000Z", "Beta"),
    ]
    pipeline = _run_sync(dlt_mod, tmp_path, monkeypatch, resync, blocks)

    rows = _read_pages(pipeline)
    # hard_delete drops the archived page from staging → orphan cleanup would
    # then forget it downstream.
    assert "p1" not in rows
    assert "p2" in rows
