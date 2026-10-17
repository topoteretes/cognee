import httpx

from cognee.tasks.ingestion.connectors.openalex import (
    OpenAlexClient,
    _iter_work_rows,
    _work_row,
    decode_abstract_inverted_index,
)


def test_decode_abstract_inverted_index_orders_positions():
    assert decode_abstract_inverted_index({"world": [1], "Hello": [0], "!": [2]}) == "Hello world !"


def test_work_row_contains_graph_identifiers_and_content():
    row = _work_row(
        {
            "id": "https://openalex.org/W1",
            "title": "A paper",
            "abstract_inverted_index": {"Useful": [0], "research": [1]},
            "authorships": [
                {
                    "author": {"id": "https://openalex.org/A1"},
                    "institutions": [{"id": "https://openalex.org/I1"}],
                }
            ],
            "topics": [{"id": "https://openalex.org/T1"}],
            "primary_location": {"source": {"id": "https://openalex.org/S1"}},
            "referenced_works": ["https://openalex.org/W2"],
        },
        "scope",
    )
    assert row["id"] == "W1"
    assert row["authors"] == ["A1"]
    assert row["institutions"] == ["I1"]
    assert row["topics"] == ["T1"]
    assert row["venue"] == "S1"
    assert "Useful research" in row["content"]


def test_cursor_pagination_and_incremental_filter():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        cursor = request.url.params["cursor"]
        if cursor == "*":
            return httpx.Response(
                200,
                json={
                    "meta": {"next_cursor": "next"},
                    "results": [{"id": "https://openalex.org/W1", "title": "One"}],
                },
            )
        return httpx.Response(
            200,
            json={"meta": {"next_cursor": None}, "results": [{"id": "https://openalex.org/W2"}]},
        )

    client = OpenAlexClient(httpx.Client(transport=httpx.MockTransport(handler)))
    state = {}
    rows = list(
        _iter_work_rows(
            client,
            scope="test",
            filters=[],
            state=state,
            page_size=2,
            from_updated_date="2026-01-01",
        )
    )
    assert [row["id"] for row in rows] == ["W1", "W2"]
    assert requests[0].url.params["filter"] == "from_updated_date:2026-01-01"
    assert state["seen_work_ids"] == ["W1", "W2"]


def test_client_retries_rate_limit(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"meta": {"next_cursor": None}, "results": []})

    monkeypatch.setattr("cognee.tasks.ingestion.connectors.openalex.time.sleep", lambda _: None)
    client = OpenAlexClient(httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.list_works(cursor="*", per_page=1)["results"] == []
    assert attempts == 2
