"""The Dev.to source lists an author completely, and fetches and forgets only on that listing.

Every run lists all published articles; a body is fetched only for an article that is new or
whose ``published_at`` / ``edited_at`` changed. A previous member the listing no longer contains
is tombstoned. An empty listing after a sync, a malformed listing or an API error never forgets
anything. The walker is driven with a plain dict for state and a fake client.
"""

import pytest

from cognee.tasks.ingestion.connectors import devto as devto_module
from cognee.tasks.ingestion.connectors import devto_source
from cognee.tasks.ingestion.connectors.devto import (
    PAGE_SIZE,
    DevToAPIError,
    DevToClient,
    DevToEmptyListingError,
    DevToNotFoundError,
    DevToRateLimitedError,
    _iter_rows,
    _username,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

TABLE = "devto_test"
EMPTY_STATS = {"synced": 0, "unchanged": 0, "deleted": 0, "vanished": 0}
USER = {"name": "Ada Lovelace", "username": "ada"}


def _article(article_id: int, title: str = "", edited_at=None, **extra) -> dict:
    return {
        "id": article_id,
        "title": title or f"Article {article_id}",
        "url": f"https://dev.to/ada/article-{article_id}",
        "canonical_url": f"https://dev.to/ada/article-{article_id}",
        "published_at": "2026-10-01T10:00:00Z",
        "edited_at": edited_at,
        # The list endpoint's shape: tag_list is a list, tags a string.
        "tag_list": ["python", "ai"],
        "tags": "python, ai",
        "user": USER,
        "public_reactions_count": 3,
        **extra,
    }


class FakeDevTo:
    """Serves a listing and article bodies, and records every request."""

    def __init__(self, articles: list[dict], bodies: dict[int, str] | None = None):
        self.articles = articles
        self.bodies = bodies or {}
        self.requests: list[tuple[str, dict | None]] = []
        self.gone: set[int] = set()

    def get(self, path, params=None):
        self.requests.append((path, params))
        if path == "/articles":
            start = (params["page"] - 1) * params["per_page"]
            return self.articles[start : start + params["per_page"]]
        article_id = int(path.rsplit("/", 1)[1])
        if article_id in self.gone:
            raise DevToNotFoundError("Dev.to answered HTTP 404")
        (summary,) = [a for a in self.articles if a["id"] == article_id]
        # The single-article endpoint swaps the two tag fields and adds the body.
        detail = {**summary, "tags": summary["tag_list"], "tag_list": summary["tags"]}
        detail["body_markdown"] = self.bodies.get(article_id, f"Body of {article_id}.\n")
        return detail

    def fetched(self) -> list[int]:
        return [int(path.rsplit("/", 1)[1]) for path, _ in self.requests if path != "/articles"]


def _rows(client, state=None, stats=None, check_active=None) -> list[dict]:
    stats = dict(EMPTY_STATS) if stats is None else stats
    return list(_iter_rows(client, "ada", {} if state is None else state, stats, check_active))


# --- factory ---------------------------------------------------------------------------------


def test_the_resource_is_a_merge_resource_tagged_for_document_ingestion():
    resource = devto_source(username="ada", resource_name=TABLE, client=FakeDevTo([]))

    assert resource.name == TABLE
    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "devto"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == TABLE
    hints = resource.compute_table_schema()
    assert hints["write_disposition"] == "merge"
    assert hints["columns"]["_deleted"]["hard_delete"] is True
    assert resource.cognee_sync_stats == {}


def test_the_default_resource_name_follows_the_username():
    resource = devto_source(username="@Some.Org", client=FakeDevTo([]))

    assert resource.name == "devto_some_org"


def test_an_older_core_without_node_set_support_is_refused(monkeypatch):
    monkeypatch.setattr(devto_module.dlt_utils, "DOCUMENT_SYNC_VERSION", 1)

    with pytest.raises(RuntimeError, match="DOCUMENT_SYNC_VERSION >= 2"):
        devto_source(username="ada")


@pytest.mark.parametrize("username", ["", "  ", "ada lovelace", "ada/../x", None, 42])
def test_a_username_that_is_not_one_is_refused(username):
    with pytest.raises(ValueError):
        _username(username)


def test_a_leading_at_and_letter_case_are_normalized():
    assert _username(" @Ada_99 ") == "ada_99"


@pytest.mark.parametrize("resource_name", ["", "   ", 7])
def test_an_empty_resource_name_is_refused(resource_name):
    with pytest.raises(ValueError):
        devto_source(username="ada", resource_name=resource_name)


# --- rows ------------------------------------------------------------------------------------


def test_a_first_run_yields_one_row_per_article_with_tags_as_node_sets():
    client = FakeDevTo([_article(2, "Second"), _article(1, "First")], {1: "Hello *world*.\n"})
    stats = dict(EMPTY_STATS)

    rows = _rows(client, stats=stats)

    assert [row["id"] for row in rows] == ["devto:article:1", "devto:article:2"]
    first = rows[0]
    assert first["title"] == "First"
    assert first["url"] == "https://dev.to/ada/article-1"
    assert first["_deleted"] is False
    assert first["content"] == (
        "Author: Ada Lovelace (@ada)\n"
        "Published: 2026-10-01T10:00:00Z\n"
        "Tags: ai, python\n\n"
        "Hello *world*."
    )
    assert first[NODE_SET_COLUMN] == ["devto:tag:ai", "devto:tag:python"]
    assert stats == {**EMPTY_STATS, "synced": 2}


def test_volatile_counts_are_not_rendered():
    rows = _rows(FakeDevTo([_article(1, public_reactions_count=999, comments_count=5)]))

    assert "999" not in rows[0]["content"]
    assert set(rows[0]) == {"id", "title", "content", "url", "_deleted", NODE_SET_COLUMN}


def test_a_canonical_url_elsewhere_is_rendered():
    rows = _rows(FakeDevTo([_article(1, canonical_url="https://ada.example/post")]))

    assert "Canonical URL: https://ada.example/post" in rows[0]["content"]


def test_an_article_without_tags_has_no_node_set_column():
    rows = _rows(FakeDevTo([_article(1, tag_list=[], tags="")]))

    assert NODE_SET_COLUMN not in rows[0]
    assert "Tags:" not in rows[0]["content"]


def test_tags_given_only_as_a_string_are_split():
    client = FakeDevTo([_article(1, tag_list="Rust , WebDev,rust", tags="Rust , WebDev,rust")])

    assert _rows(client)[0][NODE_SET_COLUMN] == ["devto:tag:rust", "devto:tag:webdev"]


# --- incremental -----------------------------------------------------------------------------


def test_a_quiet_run_fetches_no_body_and_yields_nothing():
    client = FakeDevTo([_article(1), _article(2)])
    state: dict = {}
    _rows(client, state)
    client.requests.clear()
    stats = dict(EMPTY_STATS)

    assert _rows(client, state, stats) == []
    assert client.fetched() == []
    assert stats == {**EMPTY_STATS, "unchanged": 2}


def test_an_edit_refetches_only_that_article():
    client = FakeDevTo([_article(1), _article(2)])
    state: dict = {}
    _rows(client, state)
    client.articles[1] = _article(2, "Second, revised", edited_at="2026-10-20T08:00:00Z")
    client.bodies[2] = "New text.\n"
    client.requests.clear()

    rows = _rows(client, state)

    assert client.fetched() == [2]
    assert [(row["id"], row["title"]) for row in rows] == [("devto:article:2", "Second, revised")]
    assert rows[0]["content"].endswith("New text.")


def test_an_article_that_left_the_listing_is_forgotten():
    client = FakeDevTo([_article(1), _article(2), _article(3)])
    state: dict = {}
    _rows(client, state)
    client.articles = [_article(1), _article(3)]
    stats = dict(EMPTY_STATS)

    rows = _rows(client, state, stats)

    assert rows == [{"id": "devto:article:2", "_deleted": True}]
    assert stats["deleted"] == 1
    assert sorted(state["articles"]) == ["1", "3"]


def test_an_article_unpublished_between_listing_and_fetch_is_skipped_or_forgotten():
    client = FakeDevTo([_article(1), _article(2)])
    state: dict = {}
    _rows(client, state)
    client.articles[0] = _article(1, edited_at="2026-10-21T00:00:00Z")
    client.articles.append(_article(3))
    client.gone = {1, 3}
    stats = dict(EMPTY_STATS)

    rows = _rows(client, state, stats)

    # The synced one is forgotten, the new one never existed for memory.
    assert rows == [{"id": "devto:article:1", "_deleted": True}]
    assert stats == {**EMPTY_STATS, "unchanged": 1, "deleted": 1, "vanished": 2}
    assert list(state["articles"]) == ["2"]


def test_an_empty_listing_after_a_sync_forgets_nothing():
    client = FakeDevTo([_article(1)])
    state: dict = {}
    _rows(client, state)
    before = dict(state)
    client.articles = []

    with pytest.raises(DevToEmptyListingError):
        _rows(client, state)
    assert state == before


def test_an_empty_listing_on_a_first_run_is_fine():
    stats = dict(EMPTY_STATS)

    assert _rows(FakeDevTo([]), stats=stats) == []
    assert stats == EMPTY_STATS


def test_a_listing_entry_without_an_id_raises_before_any_row():
    client = FakeDevTo([_article(1), {"title": "no id"}])
    state: dict = {}

    with pytest.raises(DevToAPIError):
        _rows(client, state)
    assert client.fetched() == []
    assert state == {}


def test_a_listing_longer_than_one_page_is_read_to_the_end():
    client = FakeDevTo([_article(i) for i in range(1, PAGE_SIZE + 3)])

    rows = _rows(client)

    pages = [params["page"] for path, params in client.requests if path == "/articles"]
    assert pages == [1, 2]
    assert len(rows) == PAGE_SIZE + 2


def test_a_mismatched_article_body_raises_and_keeps_state():
    client = FakeDevTo([_article(1)])
    client.get = lambda path, params=None: [_article(1)] if path == "/articles" else {"id": 9}
    state: dict = {}

    with pytest.raises(DevToAPIError):
        _rows(client, state)
    assert state == {}


def test_check_active_runs_before_every_article():
    calls = []
    _rows(FakeDevTo([_article(1), _article(2)]), check_active=lambda: calls.append(1))

    assert len(calls) == 2


# --- transport -------------------------------------------------------------------------------


class _Response:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _serve(monkeypatch, *responses):
    import requests

    queue = list(responses)
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return queue.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_the_client_retries_a_rate_limit_and_honors_retry_after(monkeypatch):
    calls = _serve(
        monkeypatch, _Response(429, headers={"Retry-After": "2"}), _Response(200, [{"id": 1}])
    )
    slept = []

    assert DevToClient(sleep=slept.append).get("/articles", {"page": 1}) == [{"id": 1}]
    assert slept == [2.0]
    assert calls[0][0] == "https://dev.to/api/articles"
    assert calls[0][1]["headers"]["User-Agent"] == "cognee-devto-connector"


def test_the_client_gives_up_on_a_long_retry_after(monkeypatch):
    _serve(monkeypatch, _Response(429, headers={"Retry-After": "3600"}))

    with pytest.raises(DevToRateLimitedError):
        DevToClient(sleep=lambda _: None).get("/articles")


def test_the_client_retries_server_errors_then_raises_without_the_body(monkeypatch):
    _serve(monkeypatch, *[_Response(503, "secret body")] * 3)

    with pytest.raises(DevToAPIError) as raised:
        DevToClient(sleep=lambda _: None).get("/articles/1")
    assert "secret" not in str(raised.value)
    assert "503" in str(raised.value)


def test_the_client_maps_404_and_bad_json(monkeypatch):
    _serve(monkeypatch, _Response(404), _Response(200, ValueError("no json")))
    client = DevToClient(sleep=lambda _: None)

    with pytest.raises(DevToNotFoundError):
        client.get("/articles/1")
    with pytest.raises(DevToAPIError, match="not JSON"):
        client.get("/articles/1")
