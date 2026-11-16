"""Unit tests for the arXiv DLT source: the ingest path, the incremental cursor and
forget-on-delete, all against a fake service — no network, no waiting.

The tests drive ``_iter_rows`` directly with a plain dict as the resource state, the
way obsidian's and asana's unit tests do, and use ``arxiv_source`` itself where the
factory's validation and the dlt resource attributes are what is under test.
"""

import re
from datetime import datetime, timezone
from xml.etree import ElementTree

import pytest

import cognee_connector_arxiv.source as source_module
from cognee_connector_arxiv import arxiv_source
from cognee_connector_arxiv.source import (
    ArxivClient,
    ArxivResponseError,
    ArxivSourceError,
    _iter_rows,
    _parse_page,
    parse_window_start,
)

WINDOW_RE = re.compile(r"submittedDate:\[(\d{12}) TO (\d{12})\]")


def _entry(
    bare_id: str,
    *,
    published: str,
    updated: str | None = None,
    title: str = "A Paper",
    summary: str = "An abstract.",
    authors: tuple[str, ...] = ("Alice Example",),
    categories: tuple[str, ...] = ("cs.AI",),
    comment: str = "",
    journal_ref: str = "",
    doi: str = "",
) -> str:
    updated = updated or published
    parts = [
        f"<id>http://arxiv.org/abs/{bare_id}v2</id>",
        f"<published>{published}</published>",
        f"<updated>{updated}</updated>",
        f"<title>{title}</title>",
        f"<summary>{summary}</summary>",
        *[f"<author><name>{name}</name></author>" for name in authors],
        *[
            f'<category term="{term}" scheme="http://arxiv.org/schemas/atom"/>'
            for term in categories
        ],
    ]
    if comment:
        parts.append(
            f'<arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">{comment}</arxiv:comment>'
        )
    if journal_ref:
        parts.append(
            f'<arxiv:journal_ref xmlns:arxiv="http://arxiv.org/schemas/atom">{journal_ref}</arxiv:journal_ref>'
        )
    if doi:
        parts.append(f'<arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">{doi}</arxiv:doi>')
    return f"<entry>{''.join(parts)}</entry>"


def _feed(total: int, *entries: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <title>Query</title>
  <opensearch:totalResults>{total}</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>100</opensearch:itemsPerPage>
  {"".join(entries)}
</feed>"""


class FakeArxiv:
    """The duck-typed service ``_iter_rows`` takes: one Atom feed per ``start``."""

    def __init__(self, pages: dict[int, str]):
        self.pages = pages
        self.calls: list[dict] = []

    def get_feed(self, params: dict) -> ElementTree.Element:
        self.calls.append(dict(params))
        page = self.pages[params["start"]]
        if isinstance(page, Exception):
            raise page
        return ElementTree.fromstring(page)


def _run(service, state, *, categories=("cs.AI",), authors=(), window_start=None, max_results=100):
    # The same keys the dlt resource body initializes before iterating.
    stats: dict = {"synced": 0, "deleted": 0, "requests": 0}
    rows = list(
        _iter_rows(
            service,
            list(categories),
            list(authors),
            window_start,
            max_results,
            state,
            stats,
        )
    )
    return rows, stats


def _window(service, call: int = 0) -> tuple[str, str]:
    match = WINDOW_RE.search(service.calls[call]["search_query"])
    assert match is not None
    return match.group(1), match.group(2)


def test_first_run_ingests_every_paper_as_a_document_row():
    service = FakeArxiv(
        {
            0: _feed(
                2,
                _entry(
                    "2401.00001",
                    published="2024-01-10T12:00:00Z",
                    title="Attention Is All You Need",
                    summary="We propose the Transformer.",
                    authors=("Ashish Vaswani", "Noam Shazeer"),
                    categories=("cs.CL", "cs.LG"),
                    comment="Accepted at NeurIPS",
                    journal_ref="Advances in NeurIPS 30 (2017)",
                    doi="10.48550/arXiv.1706.03762",
                ),
                _entry(
                    "2401.00002",
                    published="2024-01-12T08:30:00Z",
                    title="A Second Paper",
                ),
            )
        }
    )
    state: dict = {}

    rows, stats = _run(service, state)

    assert [row["id"] for row in rows] == ["arxiv:2401.00001", "arxiv:2401.00002"]
    first = rows[0]
    assert first["title"] == "Attention Is All You Need"
    assert first["url"] == "https://arxiv.org/abs/2401.00001"
    assert first["_deleted"] is False
    assert first["content"].startswith("arXiv paper 2401.00001 in cs.CL, cs.LG.")
    assert "Title: Attention Is All You Need" in first["content"]
    assert "Authors: Ashish Vaswani, Noam Shazeer" in first["content"]
    assert "Abstract: We propose the Transformer." in first["content"]
    assert "Comment: Accepted at NeurIPS" in first["content"]
    assert "Journal reference: Advances in NeurIPS 30 (2017)" in first["content"]
    assert "DOI: 10.48550/arXiv.1706.03762" in first["content"]
    assert first["cognee_node_set"] == ["arxiv:category:cs.CL", "arxiv:category:cs.LG"]
    assert stats["synced"] == 2
    # The cursor is the newest published timestamp; the query is the whole history.
    assert state["cursor"] == "2024-01-12T08:30:00+00:00"
    start, _ = _window(service)
    assert start == "000000000000"
    assert service.calls[0]["sortBy"] == "submittedDate"
    assert service.calls[0]["sortOrder"] == "ascending"


def test_selection_combines_categories_and_authors_with_and():
    service = FakeArxiv({0: _feed(0)})

    _run(service, {}, categories=("cs.AI",), authors=("lecun_y",))

    assert service.calls[0]["search_query"].startswith(
        "cat:cs.AI AND au:lecun_y AND submittedDate:"
    )


def test_a_listing_pages_until_the_feed_runs_dry():
    pages = {
        start: _feed(
            250,
            *[
                _entry(f"2401.{i:05d}", published="2024-01-10T12:00:00Z")
                for i in range(start, min(start + 100, 250))
            ],
        )
        for start in (0, 100, 200)
    }
    service = FakeArxiv(pages)

    rows, _ = _run(service, {}, max_results=100)

    assert len(rows) == 250
    assert [call["start"] for call in service.calls] == [0, 100, 200]
    assert all(call["max_results"] == 100 for call in service.calls)


def test_a_listing_stops_when_a_page_returns_no_entries():
    # totalResults says one more paper than page 0 delivered: page 1 answers with an
    # empty entry list, and the listing stops there instead of paging on forever.
    service = FakeArxiv(
        {
            0: _feed(2, _entry("2401.00001", published="2024-01-10T12:00:00Z")),
            1: _feed(2),
        }
    )

    rows, _ = _run(service, {})

    assert [row["id"] for row in rows] == ["arxiv:2401.00001"]
    assert len(service.calls) == 2


def test_the_second_run_queries_only_from_the_stored_cursor():
    first = FakeArxiv(
        {
            0: _feed(
                2,
                _entry("2401.00001", published="2024-01-10T12:00:00Z"),
                _entry("2401.00002", published="2024-01-12T08:30:00Z"),
            )
        }
    )
    state: dict = {}
    _run(first, state)

    second = FakeArxiv(
        {
            0: _feed(
                2,
                _entry("2401.00002", published="2024-01-12T08:30:00Z"),
                _entry("2401.00003", published="2024-01-15T09:00:00Z"),
            )
        }
    )
    rows, stats = _run(second, state)

    # Only the new paper is yielded; the cursor paper is confirmed present, not
    # re-rendered; and the cursor advanced past the new paper.
    assert [row["id"] for row in rows] == ["arxiv:2401.00003"]
    assert stats["synced"] == 1
    assert state["cursor"] == "2024-01-15T09:00:00+00:00"
    start, end = _window(second)
    assert start == "202401120830"
    assert end > start


def test_a_paper_already_synced_is_not_yielded_again():
    entry = _entry("2401.00001", published="2024-01-10T12:00:00Z")
    state: dict = {}
    _run(FakeArxiv({0: _feed(1, entry)}), state)

    rows, stats = _run(FakeArxiv({0: _feed(1, entry)}), state)

    assert rows == []
    assert stats["synced"] == 0
    # The cursor still advanced over the confirmed paper.
    assert state["cursor"] == "2024-01-10T12:00:00+00:00"


def test_a_paper_whose_updated_timestamp_changed_is_re_rendered():
    state: dict = {}
    _run(
        FakeArxiv({0: _feed(1, _entry("2401.00001", published="2024-01-10T12:00:00Z"))}),
        state,
    )

    rows, _ = _run(
        FakeArxiv(
            {
                0: _feed(
                    1,
                    _entry(
                        "2401.00001",
                        published="2024-01-10T12:00:00Z",
                        updated="2024-01-11T00:00:00Z",
                        title="A Paper, Revised",
                    ),
                )
            }
        ),
        state,
    )

    assert [row["title"] for row in rows] == ["A Paper, Revised"]
    assert state["members"]["arxiv:2401.00001"]["updated"] == "2024-01-11T00:00:00+00:00"


def test_a_paper_that_left_the_window_listing_is_forgotten():
    state: dict = {}
    _run(
        FakeArxiv(
            {
                0: _feed(
                    2,
                    _entry("2401.00001", published="2024-01-10T12:00:00Z"),
                    _entry("2401.00002", published="2024-01-12T08:30:00Z"),
                )
            }
        ),
        state,
    )

    # The newest paper disappeared upstream before the next sync.
    rows, stats = _run(
        FakeArxiv(
            {
                0: _feed(
                    1,
                    _entry("2401.00001", published="2024-01-10T12:00:00Z"),
                )
            }
        ),
        state,
    )

    assert rows == [{"id": "arxiv:2401.00002", "_deleted": True}]
    assert stats["deleted"] == 1
    assert "arxiv:2401.00002" not in state["members"]
    assert "arxiv:2401.00001" in state["members"]


def test_a_paper_behind_the_cursor_is_not_rechecked():
    """arXiv has no deletion feed: deletions settle only inside the re-queried window."""
    state: dict = {}
    _run(
        FakeArxiv(
            {
                0: _feed(
                    2,
                    _entry("2401.00001", published="2024-01-10T12:00:00Z"),
                    _entry("2401.00002", published="2024-01-12T08:30:00Z"),
                )
            }
        ),
        state,
    )

    # 2401.00001 is behind the cursor (2024-01-12T08:30) and the second run's listing
    # does not contain it; it stays, because the window the run settled starts at the
    # cursor. The cursor paper 2401.00002 is inside the window and is checked.
    rows, _ = _run(FakeArxiv({0: _feed(0)}), state)

    assert rows == [{"id": "arxiv:2401.00002", "_deleted": True}]
    assert "arxiv:2401.00001" in state["members"]


def test_a_failed_listing_raises_and_leaves_the_state_untouched():
    state: dict = {
        "cursor": "2024-01-10T12:00:00+00:00",
        "members": {
            "arxiv:old": {
                "submitted": "2024-01-01T00:00:00+00:00",
                "updated": "2024-01-01T00:00:00+00:00",
            }
        },
    }
    snapshot = dict(state)
    service = FakeArxiv(
        {
            0: ArxivResponseError(
                "arXiv API answered HTTP 500; nothing was synced and nothing was forgotten."
            )
        }
    )

    with pytest.raises(ArxivResponseError, match="HTTP 500"):
        _run(service, state)

    assert state == snapshot


def test_a_feed_of_the_wrong_shape_raises_instead_of_looking_empty():
    with pytest.raises(ArxivResponseError, match="not an Atom feed"):
        _parse_page(ElementTree.fromstring("<html><body>nope</body></html>"))

    missing_total = _feed(0).replace("<opensearch:totalResults>0</opensearch:totalResults>", "")
    with pytest.raises(ArxivResponseError, match="totalResults"):
        _parse_page(ElementTree.fromstring(missing_total))

    no_id = _feed(
        1,
        _entry("2401.00001", published="2024-01-10T12:00:00Z").replace(
            "<id>http://arxiv.org/abs/2401.00001v2</id>", ""
        ),
    )
    with pytest.raises(ArxivResponseError, match="no id"):
        _parse_page(ElementTree.fromstring(no_id))


def test_a_cursor_in_the_future_refuses_to_run():
    state = {"cursor": "2999-01-10T12:00:00+00:00"}
    service = FakeArxiv({})

    with pytest.raises(ArxivSourceError, match="in the future"):
        _run(service, state)

    assert service.calls == []


def test_a_corrupt_cursor_refuses_to_run():
    state = {"cursor": "not-a-timestamp"}

    with pytest.raises(ArxivSourceError, match="not a timestamp"):
        _run(FakeArxiv({}), state)


def test_an_empty_selection_is_refused():
    with pytest.raises(ValueError, match="categories"):
        arxiv_source(categories=[], resource_name="arxiv_contract")
    with pytest.raises(ValueError, match="authors"):
        arxiv_source(authors="lecun_y", resource_name="arxiv_contract")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="categories or authors"):
        arxiv_source(resource_name="arxiv_contract")


def test_submitted_after_is_parsed_into_a_window_bound():
    assert parse_window_start(None) is None
    assert parse_window_start("2024-01-01") == "202401010000"
    assert parse_window_start("2024-01-01T12:30") == "202401011230"
    assert parse_window_start(datetime(2024, 1, 1, tzinfo=timezone.utc)) == "202401010000"
    with pytest.raises(ValueError, match="not an ISO date"):
        parse_window_start("01/01/2024")
    with pytest.raises(ValueError, match="ISO date/datetime string or a datetime"):
        parse_window_start(1704067200)  # type: ignore[arg-type]


def test_submitted_after_bounds_only_the_first_run():
    service = FakeArxiv({0: _feed(0)})

    _run(service, {}, window_start=parse_window_start("2024-06-01"))

    start, _ = _window(service)
    assert start == "202406010000"


def test_build_arxiv_service_builds_a_client_with_the_built_in_delay():
    sleeps: list[float] = []
    client = ArxivClient(request_delay=3.0, sleep=sleeps.append)
    calls: list[dict] = []

    class _Response:
        status_code = 200
        text = _feed(0)

    class _Requests:
        class RequestException(Exception):
            pass

        @staticmethod
        def get(url, params=None, timeout=None):
            calls.append({"url": url, "params": params, "timeout": timeout})
            return _Response()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(__import__("sys").modules, "requests", _Requests)
    try:
        client.get_feed({"search_query": "cat:cs.AI"})
        client.get_feed({"search_query": "cat:cs.AI"})
    finally:
        monkeypatch.undo()

    # The delay is built in, not discovered: exactly one wait, before the second
    # request, of the configured 3 seconds.
    assert sleeps == [3.0]
    assert len(calls) == 2
    assert calls[0]["url"] == source_module.API_BASE_URL
