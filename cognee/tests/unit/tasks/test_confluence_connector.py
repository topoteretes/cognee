"""Unit tests for the Confluence connector (cognee/tasks/ingestion/connectors/confluence.py).

The Confluence Cloud REST API is fully mocked via ``FakeConfluenceSession`` — no
``requests`` traffic and no live credentials are required, so these run in CI.
Coverage:

  - storage-format HTML is stripped to plain text
  - the incremental cursor reads version.createdAt (v2) or version.when (v1)
  - full backfill yields every page and records the cursor + id set
  - incremental re-sync yields ONLY pages modified since the cursor
  - pages that vanish from the sweep become hard-delete markers (forget-on-delete)
  - footer comments are folded into their page's text when requested
  - space keys are pushed down as an API filter
  - the dlt resource is wired with merge + id PK + the hard_delete column
  - a real dlt merge removes the marked row (end-to-end forget-on-delete)

The end-to-end "deletion removes from memory" guarantee is provided by the
existing ``orphan_cleanup`` path (see test_dlt_p0_correctness.py); here we prove
the connector emits the markers that drive it, and that dlt acts on them.
"""

import re

import pytest

from cognee.tasks.ingestion.connectors.confluence import (
    _clean_html,
    _version_when,
    confluence_source,
    sync_pages,
)

BASE_URL = "https://test.atlassian.net"


# ---------------------------------------------------------------------------
# Fake Confluence REST API
# ---------------------------------------------------------------------------
def _page(page_id, *, when, title="", space_id="1", body="", comments=None):
    """Build a page as the API returns it, plus test-only ``_body`` / ``_comments``."""
    return {
        "id": page_id,
        "title": title,
        "spaceId": space_id,
        "version": {"createdAt": when},
        "_links": {"webui": f"/spaces/ENG/pages/{page_id}"},
        "_body": body,
        "_comments": comments or [],
    }


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeConfluenceSession:
    """Minimal stand-in for a ``requests`` session hitting Confluence v2."""

    def __init__(self, spaces, pages_by_space):
        # spaces: [{"id": "1", "key": "ENG"}]; pages_by_space: {"1": [page, ...]}
        self.spaces = spaces
        self.pages_by_space = pages_by_space
        self.calls = []

    def _find(self, page_id):
        for pages in self.pages_by_space.values():
            for page in pages:
                if page["id"] == page_id:
                    return page
        return None

    def get(self, url, params=None):
        params = params or {}
        self.calls.append((url, params))

        if url.endswith("/spaces"):
            spaces = self.spaces
            if params.get("keys"):
                # v2 `keys` is a repeated array param → requests passes a list.
                # Model that contract (not comma-joining) so the fake fails loudly
                # if the connector ever regresses to keys="ENG,DOCS".
                keys = params["keys"]
                assert isinstance(keys, list), f"keys must be a list, got {keys!r}"
                wanted = set(keys)
                spaces = [s for s in self.spaces if s["key"] in wanted]
            return _Resp({"results": spaces, "_links": {}})

        m = re.search(r"/spaces/([^/]+)/pages$", url)
        if m:
            listed = [
                {k: v for k, v in page.items() if not k.startswith("_") or k == "_links"}
                for page in self.pages_by_space.get(m.group(1), [])
            ]
            return _Resp({"results": listed, "_links": {}})

        m = re.search(r"/pages/([^/]+)/footer-comments$", url)
        if m:
            page = self._find(m.group(1)) or {}
            results = [{"body": {"storage": {"value": c}}} for c in page.get("_comments", [])]
            return _Resp({"results": results, "_links": {}})

        m = re.search(r"/pages/([^/]+)$", url)
        if m:
            page = self._find(m.group(1)) or {}
            return _Resp({"body": {"storage": {"value": page.get("_body", "")}}})

        raise AssertionError(f"unexpected URL: {url}")


def _make_session(pages):
    return FakeConfluenceSession(
        spaces=[{"id": "1", "key": "ENG"}],
        pages_by_space={"1": pages},
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_clean_html_strips_tags_unescapes_and_collapses_whitespace():
    raw = "<p>Hello&nbsp;<b>world</b></p>\n<p>  second   line </p>"
    assert _clean_html(raw) == "Hello world second line"
    assert _clean_html("") == ""
    assert _clean_html(None) == ""


def test_version_when_prefers_created_at_then_falls_back_to_when():
    assert _version_when({"version": {"createdAt": "2024-05-01T00:00:00.000Z"}}) == (
        "2024-05-01T00:00:00.000Z"
    )
    assert _version_when({"version": {"when": "2020-01-01T00:00:00.000Z"}}) == (
        "2020-01-01T00:00:00.000Z"
    )
    assert _version_when({}) == ""


# ---------------------------------------------------------------------------
# sync_pages — backfill / incremental / deletion
# ---------------------------------------------------------------------------
def test_backfill_yields_all_pages_and_records_cursor_and_ids():
    session = _make_session(
        [
            _page("1", when="2024-01-01T10:00:00.000Z", body="<p>alpha</p>"),
            _page("2", when="2024-01-02T10:00:00.000Z", body="<p>beta</p>"),
        ]
    )
    state = {}
    rows = list(sync_pages(session, BASE_URL, state, include_comments=False))

    assert {r["id"] for r in rows} == {"1", "2"}
    assert all(r["_deleted"] is False for r in rows)
    assert {r["body"] for r in rows} == {"alpha", "beta"}
    # Cursor + id set captured for the next incremental run.
    assert state["last_when"] == "2024-01-02T10:00:00.000Z"
    assert state["known_ids"] == ["1", "2"]
    # Absolute page URL is reconstructed for citations.
    assert rows[0]["url"].startswith(f"{BASE_URL}/wiki/spaces/ENG/pages/")


def test_incremental_yields_only_pages_modified_since_cursor():
    session = _make_session(
        [
            _page("1", when="2024-01-01T10:00:00.000Z", body="<p>old</p>"),
            _page("2", when="2024-02-01T10:00:00.000Z", body="<p>new</p>"),
        ]
    )
    state = {"known_ids": ["1", "2"], "last_when": "2024-01-15T00:00:00.000Z"}
    rows = list(sync_pages(session, BASE_URL, state, include_comments=False))

    assert [r["id"] for r in rows] == ["2"]  # only the page newer than the cursor
    assert state["last_when"] == "2024-02-01T10:00:00.000Z"


def test_incremental_no_changes_is_a_noop():
    session = _make_session([_page("1", when="2024-01-01T10:00:00.000Z")])
    state = {"known_ids": ["1"], "last_when": "2024-01-01T10:00:00.000Z"}
    rows = list(sync_pages(session, BASE_URL, state, include_comments=False))
    assert rows == []


def test_deleted_page_emits_hard_delete_marker():
    # Page "2" was known last run but is gone from the sweep now.
    session = _make_session([_page("1", when="2024-01-01T10:00:00.000Z")])
    state = {"known_ids": ["1", "2"], "last_when": "2024-01-01T10:00:00.000Z"}
    rows = list(sync_pages(session, BASE_URL, state, include_comments=False))

    assert rows == [{"id": "2", "_deleted": True}]
    assert state["known_ids"] == ["1"]  # sweep now reflects reality


def test_empty_sweep_does_not_mass_delete_and_preserves_state():
    # A sweep that returns zero pages while pages were known is treated as a
    # transient failure, NOT "everything deleted" — otherwise a benign blip would
    # wipe the whole dataset and overwrite known_ids permanently.
    session = FakeConfluenceSession(spaces=[], pages_by_space={})
    state = {"known_ids": ["1", "2"], "last_when": "2024-01-01T10:00:00.000Z"}
    rows = list(sync_pages(session, BASE_URL, state, include_comments=False))

    assert rows == []  # no hard-delete markers emitted
    assert state["known_ids"] == ["1", "2"]  # prior id set preserved


def test_new_page_below_cursor_is_still_ingested():
    # A page new to the corpus is fetched regardless of its version timestamp
    # (moved into a tracked space / restored / tied at the cursor boundary),
    # while an already-known unchanged page at the same old timestamp is skipped.
    session = _make_session(
        [
            _page("1", when="2024-01-01T10:00:00.000Z", body="<p>known</p>"),  # old + known
            _page("3", when="2024-01-01T10:00:00.000Z", body="<p>moved-in</p>"),  # old + new
            _page("4", when="2024-06-01T00:00:00.000Z", body="<p>tie</p>"),  # boundary tie + new
        ]
    )
    state = {"known_ids": ["1"], "last_when": "2024-06-01T00:00:00.000Z"}
    rows = list(sync_pages(session, BASE_URL, state, include_comments=False))

    assert sorted(r["id"] for r in rows) == ["3", "4"]  # page "1" skipped, new pages ingested


def test_comments_are_folded_into_page_body_when_requested():
    session = _make_session(
        [
            _page(
                "1",
                when="2024-01-01T10:00:00.000Z",
                body="<p>page</p>",
                comments=["<p>first</p>", "<p>second</p>"],
            )
        ]
    )
    rows_with = list(sync_pages(session, BASE_URL, dict(), include_comments=True))
    assert rows_with[0]["body"] == "page\n\nComments:\nfirst\n\nsecond"

    rows_without = list(
        sync_pages(
            _make_session([_page("1", when="x", body="<p>page</p>", comments=["<p>c</p>"])]),
            BASE_URL,
            dict(),
            include_comments=False,
        )
    )
    assert rows_without[0]["body"] == "page"


def test_space_keys_are_pushed_down_as_a_repeated_array_param():
    session = FakeConfluenceSession(
        spaces=[{"id": "1", "key": "ENG"}, {"id": "2", "key": "DOCS"}, {"id": "3", "key": "OPS"}],
        pages_by_space={
            "1": [_page("1", when="2024-01-01T10:00:00.000Z")],
            "2": [_page("2", when="2024-01-02T10:00:00.000Z")],
            "3": [_page("9", when="x")],
        },
    )
    # Multi-space filter must resolve BOTH requested spaces (comma-joining would
    # match neither and silently sync nothing / mass-delete on re-run).
    rows = list(
        sync_pages(session, BASE_URL, dict(), space_keys=["ENG", "DOCS"], include_comments=False)
    )

    assert sorted(r["id"] for r in rows) == ["1", "2"]  # OPS excluded, ENG+DOCS included
    spaces_call = next(params for url, params in session.calls if url.endswith("/spaces"))
    assert spaces_call["keys"] == ["ENG", "DOCS"]  # repeated array param, not "ENG,DOCS"


# ---------------------------------------------------------------------------
# confluence_source — dlt wiring — requires dlt
# ---------------------------------------------------------------------------
def test_confluence_source_resource_is_configured_for_merge_and_hard_delete():
    pytest.importorskip("dlt")

    resource = confluence_source(base_url=BASE_URL, session=_make_session([]))
    assert resource.name == "confluence_pages"

    schema = resource.compute_table_schema()
    write_disposition = schema.get("write_disposition")
    if isinstance(write_disposition, dict):  # dlt may normalize to a config dict
        write_disposition = write_disposition.get("disposition")
    assert write_disposition == "merge"

    columns = schema["columns"]
    assert columns["id"].get("primary_key") is True
    assert columns["_deleted"].get("hard_delete") is True


def test_confluence_source_requires_credentials_or_session():
    pytest.importorskip("dlt")
    with pytest.raises(ValueError, match="email and api_token"):
        confluence_source(base_url=BASE_URL)


def test_confluence_source_requires_dlt(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dlt":
            raise ImportError("no dlt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="cognee\\[confluence\\]"):
        confluence_source(base_url=BASE_URL, session=object())


# ---------------------------------------------------------------------------
# End-to-end: a real dlt merge acts on the hard-delete marker
# ---------------------------------------------------------------------------
def test_forget_on_delete_end_to_end_through_a_real_dlt_merge(tmp_path):
    dlt = pytest.importorskip("dlt")
    pytest.importorskip("duckdb")

    pipeline = dlt.pipeline(
        pipeline_name="test_confluence_e2e",
        destination=dlt.destinations.duckdb(str(tmp_path / "confluence.duckdb")),
        dataset_name="wiki",
    )

    # Sync #1: two live pages land in the destination.
    session1 = _make_session(
        [
            _page("1", when="2024-01-01T10:00:00.000Z", body="<p>a</p>"),
            _page("2", when="2024-01-02T10:00:00.000Z", body="<p>b</p>"),
        ]
    )
    pipeline.run(confluence_source(base_url=BASE_URL, session=session1, include_comments=False))
    with pipeline.sql_client() as client:
        assert client.execute_sql("SELECT count(*) FROM confluence_pages")[0][0] == 2

    # Sync #2: page "2" deleted upstream, page "1" unchanged. The connector emits
    # a hard-delete marker for "2"; dlt's merge removes it from the destination.
    session2 = _make_session([_page("1", when="2024-01-01T10:00:00.000Z", body="<p>a</p>")])
    pipeline.run(confluence_source(base_url=BASE_URL, session=session2, include_comments=False))
    with pipeline.sql_client() as client:
        rows = client.execute_sql("SELECT id FROM confluence_pages")
    assert [r[0] for r in rows] == ["1"]  # page "2" forgotten, page "1" retained
