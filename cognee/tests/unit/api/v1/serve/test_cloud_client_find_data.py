"""CloudClient.list_data(content_hash=...) must never return false matches.

FastAPI drops an unknown query param silently, so a server that predates the
content-hash lookup answers ``?content_hash=`` with the whole dataset listing.
``find_data()`` would hand that back as matches, and the PR's own use case
("delete exactly what I stored") would then delete the dataset. These tests
run the client against a fake server of each generation.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient

LISTING = "/api/v1/datasets/{dataset_id}/data"
WANTED = "9e107d9d372bb6826bd81d3542a419d6"
OTHER = "d41d8cd98f00b204e9800998ecf8427e"


def _row(content_hash: str | None, *, new_server: bool) -> dict:
    row = {"id": str(uuid4()), "name": "note.md", "mimeType": "text/markdown"}
    if new_server:
        row["contentHash"] = content_hash
    return row


def _openapi(*, with_content_hash: bool) -> dict:
    parameters = [
        {"name": "limit", "in": "query"},
        {"name": "offset", "in": "query"},
    ]
    if with_content_hash:
        parameters.append({"name": "content_hash", "in": "query"})
    return {"paths": {LISTING: {"get": {"parameters": parameters}}}}


class _Response:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _Get:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeServer:
    """One generation of server: a schema (or none) and a dataset of rows.

    ``honours_filter`` is what a server that knows ``content_hash`` does;
    an old server ignores the param and returns every row.
    """

    def __init__(self, rows, *, openapi, honours_filter):
        self.rows = rows
        self.openapi = openapi
        self.honours_filter = honours_filter
        self.requests = []

    def get(self, url, params=None):
        self.requests.append((url, dict(params or {})))
        if url.endswith("/openapi.json"):
            if self.openapi is None:
                return _Get(_Response(404, {"detail": "Not Found"}))
            return _Get(_Response(200, self.openapi))
        rows = self.rows
        if self.honours_filter and "content_hash" in params:
            rows = [row for row in rows if row.get("contentHash") == params["content_hash"]]
        offset, limit = params["offset"], params["limit"]
        return _Get(_Response(200, rows[offset : offset + limit]))


def _client(server) -> CloudClient:
    client = CloudClient.__new__(CloudClient)
    client.service_url = "http://server"
    client._get_session = AsyncMock(return_value=server)
    return client


def _listing_requests(server):
    return [params for url, params in server.requests if url.endswith("/data")]


# --- a server that knows the filter --------------------------------------------


@pytest.mark.asyncio
async def test_new_server_filters_server_side_and_rows_are_checked_again():
    match, other = _row(WANTED, new_server=True), _row(OTHER, new_server=True)
    server = _FakeServer(
        [other, match], openapi=_openapi(with_content_hash=True), honours_filter=True
    )
    client = _client(server)

    rows = await client.list_data(uuid4(), content_hash=WANTED)

    assert rows == [match]
    assert await client.supports_content_hash_lookup() is True
    # The param went on the wire and the schema was read once.
    assert all(params["content_hash"] == WANTED for params in _listing_requests(server))
    assert sum(url.endswith("/openapi.json") for url, _ in server.requests) == 1


@pytest.mark.asyncio
async def test_rows_the_server_failed_to_filter_are_dropped_client_side():
    """Belt and braces: even a server that sends contentHash but ignores the
    param cannot turn a lookup into the dataset."""
    match, other = _row(WANTED, new_server=True), _row(OTHER, new_server=True)
    server = _FakeServer(
        [other, match, _row(None, new_server=True)],
        openapi=_openapi(with_content_hash=True),
        honours_filter=False,
    )

    rows = await _client(server).list_data(uuid4(), content_hash=WANTED)

    assert rows == [match]


# --- a server that predates the filter -----------------------------------------


@pytest.mark.asyncio
async def test_old_server_is_refused_before_the_dataset_is_fetched():
    server = _FakeServer(
        [_row(None, new_server=False) for _ in range(3)],
        openapi=_openapi(with_content_hash=False),
        honours_filter=False,
    )
    client = _client(server)

    with pytest.raises(RuntimeError, match="does not support content_hash lookup"):
        await client.list_data(uuid4(), content_hash=WANTED)
    assert _listing_requests(server) == [], "no listing request on a known-old server"

    # The verdict is cached: a second lookup does not re-read the schema.
    with pytest.raises(RuntimeError):
        await client.list_data(uuid4(), content_hash=WANTED)
    assert sum(url.endswith("/openapi.json") for url, _ in server.requests) == 1


@pytest.mark.asyncio
async def test_old_server_rows_fail_closed_when_the_schema_is_unavailable():
    """No schema to probe, and the server ignores the param and sends no
    contentHash: the rows must be refused, never returned as matches."""
    server = _FakeServer(
        [_row(None, new_server=False) for _ in range(3)], openapi=None, honours_filter=False
    )
    client = _client(server)

    with pytest.raises(RuntimeError, match="does not support content_hash lookup"):
        await client.list_data(uuid4(), content_hash=WANTED)

    # Unknown is not cached, so a later call probes the schema again.
    assert client._content_hash_lookup_supported is None
    with pytest.raises(RuntimeError):
        await client.list_data(uuid4(), content_hash=WANTED)
    assert sum(url.endswith("/openapi.json") for url, _ in server.requests) == 2


@pytest.mark.asyncio
async def test_new_server_rows_work_when_the_schema_is_unavailable():
    match = _row(WANTED, new_server=True)
    server = _FakeServer([_row(OTHER, new_server=True), match], openapi=None, honours_filter=True)

    rows = await _client(server).list_data(uuid4(), content_hash=WANTED)

    assert rows == [match]


@pytest.mark.asyncio
async def test_empty_dataset_is_an_empty_list_on_any_server():
    server = _FakeServer([], openapi=None, honours_filter=False)

    assert await _client(server).list_data(uuid4(), content_hash=WANTED) == []


@pytest.mark.asyncio
async def test_plain_listing_on_an_old_server_is_unchanged():
    """Without content_hash there is nothing to probe or filter."""
    rows = [_row(None, new_server=False) for _ in range(2)]
    server = _FakeServer(rows, openapi=_openapi(with_content_hash=False), honours_filter=False)

    assert await _client(server).list_data(uuid4()) == rows
    assert not any(url.endswith("/openapi.json") for url, _ in server.requests)
    assert all("content_hash" not in params for params in _listing_requests(server))
