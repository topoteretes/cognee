"""Notion OAuth, search and parent lookups: what is sent, and that errors stay log-safe."""

import asyncio
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from cognee.modules.integrations.notion import client


class FakeSession:
    """Stands in for ``aiohttp.ClientSession``; each request pops the next scripted reply."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return self._reply()

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def _reply(self):
        status, body, headers = self.replies.pop(0)
        response = AsyncMock()
        response.status = status
        response.headers = headers or {}
        if isinstance(body, Exception):
            response.json.side_effect = body
        else:
            response.json.return_value = body
        response.__aenter__.return_value = response
        return response


# base64 of "id:secret"
BASIC_ID_SECRET = "Basic aWQ6c2VjcmV0"


def with_session(replies):
    session = FakeSession(replies)
    return session, patch.object(client.aiohttp, "ClientSession", return_value=session)


def reply(body, status=200, headers=None):
    return status, body, headers


@pytest.fixture(autouse=True)
def no_sleep():
    with patch.object(client.asyncio, "sleep", AsyncMock()) as sleep:
        yield sleep


def test_canonical_id_ignores_dashes_and_case():
    assert client.canonical_id("ABCD-ef01-2345") == client.canonical_id("abcdef012345")


@pytest.mark.asyncio
async def test_code_exchange_sends_basic_auth_and_a_json_body():
    session, patched = with_session([reply({"access_token": "tok", "workspace_id": "ws-1"})])
    with patched:
        grant = await client.exchange_code(
            "the-code", client_id="id", client_secret="secret", redirect_uri="http://cb"
        )

    _method, url, kwargs = session.requests[0]
    assert url == client.TOKEN_URL
    assert kwargs["json"] == {
        "grant_type": "authorization_code",
        "code": "the-code",
        "redirect_uri": "http://cb",
    }
    assert kwargs["headers"]["Authorization"] == BASIC_ID_SECRET
    assert kwargs["headers"]["Notion-Version"] == client.NOTION_VERSION
    assert grant["access_token"] == "tok" and grant["refresh_token"] is None
    assert grant["expires_in"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_in, kept", [(3600, 3600), (0, None), (True, None), ("1", None)])
async def test_only_a_positive_whole_number_is_a_lifetime(expires_in, kept):
    body = {"access_token": "tok", "workspace_id": "ws", "expires_in": expires_in}
    _session, patched = with_session([reply(body)])
    with patched:
        grant = await client.exchange_code(
            "c", client_id="id", client_secret="s", redirect_uri="http://cb"
        )

    assert grant["expires_in"] == kept


@pytest.mark.asyncio
async def test_a_code_exchange_without_a_workspace_is_refused():
    _session, patched = with_session([reply({"access_token": "tok"})])
    with patched, pytest.raises(client.NotionError) as caught:
        await client.exchange_code("c", client_id="id", client_secret="s", redirect_uri="http://cb")

    assert caught.value.code == "no_workspace_id"


@pytest.mark.asyncio
async def test_an_oauth_error_carries_only_its_code():
    body = {"error": "invalid_grant", "error_description": "secret workspace content"}
    _session, patched = with_session([reply(body, status=400)])
    with patched, pytest.raises(client.NotionError) as caught:
        await client.refresh_access_token("old", client_id="id", client_secret="s")

    assert caught.value.code == "invalid_grant"
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ValueError("secret body"), aiohttp.ClientPayloadError("secret body"), asyncio.TimeoutError()],
)
async def test_an_unreadable_response_is_reduced_to_its_status(error):
    _session, patched = with_session([reply(error, status=502)])
    with patched, pytest.raises(client.NotionError) as caught:
        await client.refresh_access_token("old", client_id="id", client_secret="s")

    assert str(caught.value) == "Notion token refresh failed: http_502"
    assert caught.value.__suppress_context__ is True


@pytest.mark.asyncio
async def test_a_rate_limited_request_is_retried_after_notions_delay(no_sleep):
    session, patched = with_session(
        [reply({}, status=429, headers={"Retry-After": "2"}), reply({"results": []})]
    )
    with patched:
        assert await client.search_shared("tok") == []

    assert len(session.requests) == 2
    no_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_a_wait_longer_than_the_bound_is_cut_short(no_sleep):
    _session, patched = with_session(
        [reply({}, status=429, headers={"Retry-After": "600"}), reply({"results": []})]
    )
    with patched:
        await client.search_shared("tok")

    no_sleep.assert_awaited_once_with(client.MAX_RETRY_AFTER_SECONDS)


@pytest.mark.asyncio
async def test_a_request_still_limited_after_the_retries_is_reported():
    limited = reply({"code": "rate_limited"}, status=429)
    _session, patched = with_session([limited] * (client.MAX_RATE_LIMIT_RETRIES + 1))
    with patched, pytest.raises(client.NotionError) as caught:
        await client.search_shared("tok")

    assert caught.value.code == "rate_limited"


def page(object_id, *, parent=None, title="T", **extra):
    return {
        "object": "page",
        "id": object_id,
        "parent": parent or {"type": "workspace", "workspace": True},
        "properties": {"Name": {"type": "title", "title": [{"plain_text": title}]}},
        **extra,
    }


@pytest.mark.asyncio
async def test_search_pages_through_every_result_and_keeps_an_object_once():
    data_source = {
        "object": "data_source",
        "id": "ds-1",
        "title": [{"plain_text": "Tasks"}],
        "parent": {"type": "database_id", "database_id": "db-1"},
    }
    session, patched = with_session(
        [
            reply({"results": [page("p1")], "has_more": True, "next_cursor": "c2"}),
            reply({"results": [page("P1"), data_source, page("gone", archived=True)]}),
        ]
    )
    with patched:
        found = await client.search_shared("tok")

    assert [(hit["kind"], hit["id"], hit["title"]) for hit in found] == [
        ("page", "p1", "T"),
        ("data_source", "ds-1", "Tasks"),
    ]
    assert session.requests[1][2]["json"]["start_cursor"] == "c2"
    assert found[1]["parent_kind"] == "database_id" and found[1]["parent_id"] == "db-1"


@pytest.mark.asyncio
async def test_trashed_objects_are_not_offered():
    _session, patched = with_session([reply({"results": [page("p1", in_trash=True)]})])
    with patched:
        assert await client.search_shared("tok") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, status, code",
    [
        ({"results": [], "has_more": True, "next_cursor": None}, 200, "no_cursor"),
        ({"object": "list"}, 200, "no_results"),
        ({"code": "unauthorized"}, 401, "unauthorized"),
        ({}, 500, "http_500"),
    ],
)
async def test_a_bad_search_response_is_an_error_never_a_prefix(body, status, code):
    _session, patched = with_session([reply(body, status=status)])
    with patched, pytest.raises(client.NotionError) as caught:
        await client.search_shared("tok")

    assert caught.value.code == code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, parent, expected",
    [
        ("page", {"type": "page_id", "page_id": "up"}, ("page", "up")),
        ("page", {"type": "data_source_id", "data_source_id": "ds"}, ("data_source", "ds")),
        ("page", {"type": "block_id", "block_id": "b"}, ("block", "b")),
        ("data_source", {"type": "database_id", "database_id": "db"}, ("database", "db")),
        ("page", {"type": "workspace", "workspace": True}, None),
        ("page", {"type": "something_new", "something_new": "x"}, None),
    ],
)
async def test_the_parent_comes_back_as_the_kind_to_fetch_next(kind, parent, expected):
    session, patched = with_session([reply({"parent": parent})])
    with patched:
        assert await client.get_parent("tok", kind, "obj") == expected

    method, url, _kwargs = session.requests[0]
    assert method == "GET" and url.endswith(f"/{client._OBJECT_PATH[kind]}/obj")


@pytest.mark.asyncio
async def test_an_object_that_cannot_be_read_is_an_error_with_notions_code():
    _session, patched = with_session([reply({"code": "object_not_found"}, status=404)])
    with patched, pytest.raises(client.NotionError) as caught:
        await client.get_parent("tok", "page", "obj")

    assert caught.value.code == "object_not_found"


@pytest.mark.asyncio
async def test_revoke_sends_the_token_with_basic_auth():
    session, patched = with_session([reply({})])
    with patched:
        await client.revoke_token("tok", client_id="id", client_secret="secret")

    _method, url, kwargs = session.requests[0]
    assert url == client.REVOKE_URL and kwargs["json"] == {"token": "tok"}
    assert kwargs["headers"]["Authorization"] == BASIC_ID_SECRET


def test_a_page_title_comes_from_whichever_property_is_the_title():
    item = page("p", title="Roadmap")
    item["properties"] = {"Anything": item["properties"]["Name"]}

    assert client._title(item) == "Roadmap"
    assert client._title({"object": "page", "properties": {}}) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, code",
    [
        (OSError("Connection refused to api.notion.com"), "network_error"),
        (aiohttp.ClientConnectionError("secret host"), "network_error"),
        (asyncio.TimeoutError(), "request_timeout"),
    ],
)
async def test_a_failed_connection_is_a_notion_error_not_a_raw_exception(error, code):
    class Failing(FakeSession):
        def request(self, method, url, **kwargs):
            raise error

    with (
        patch.object(client.aiohttp, "ClientSession", return_value=Failing([])),
        pytest.raises(client.NotionError) as caught,
    ):
        await client.search_shared("tok")

    assert caught.value.code == code
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "-5", "x"])
def test_a_retry_after_that_is_not_a_usable_number_never_hangs_the_wait(value):
    delay = client._retry_delay(value)

    assert 0 <= delay <= client.MAX_RETRY_AFTER_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_in", [10**18, client._MAX_EXPIRES_IN + 1])
async def test_a_lifetime_no_timedelta_could_hold_is_not_kept(expires_in):
    body = {"access_token": "tok", "workspace_id": "ws", "expires_in": expires_in}
    _session, patched = with_session([reply(body)])
    with patched:
        grant = await client.exchange_code(
            "c", client_id="id", client_secret="s", redirect_uri="http://cb"
        )

    assert grant["expires_in"] is None


@pytest.mark.asyncio
async def test_a_search_that_never_ends_is_an_error_not_a_prefix(monkeypatch):
    monkeypatch.setattr(client, "MAX_SEARCH_PAGES", 3)
    looping = reply({"results": [], "has_more": True, "next_cursor": "same"})
    session, patched = with_session([looping] * 5)
    with patched, pytest.raises(client.NotionError) as caught:
        await client.search_shared("tok")

    assert caught.value.code == "too_many_results"
    assert len(session.requests) == 3


@pytest.mark.asyncio
async def test_a_search_past_its_deadline_is_an_error(monkeypatch):
    monkeypatch.setattr(client, "SEARCH_DEADLINE_SECONDS", 0.01)

    async def never(*args, **kwargs):
        # Not asyncio.sleep: the autouse fixture above replaces it.
        await asyncio.Event().wait()

    with patch.object(client, "_walk_search", never), pytest.raises(client.NotionError) as caught:
        await client.search_shared("tok")

    assert caught.value.code == "timeout"


@pytest.mark.asyncio
async def test_a_token_check_passes_on_a_200():
    session, patched = with_session([reply({"object": "user"})])
    with patched:
        await client.verify_token("tok")

    method, url, kwargs = session.requests[0]
    assert (method, url) == ("GET", client.USERS_ME_URL)
    assert kwargs["headers"]["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, body, code",
    [
        (401, {"code": "unauthorized", "message": "secret"}, "unauthorized"),
        (401, {}, "http_401"),
        (503, {"code": "service_unavailable"}, "service_unavailable"),
        (500, None, "http_500"),
    ],
)
async def test_a_token_check_that_fails_carries_only_a_code(status, body, code):
    _session, patched = with_session([reply(body, status=status)])
    with patched, pytest.raises(client.NotionError) as caught:
        await client.verify_token("tok")

    assert caught.value.code == code
    assert "secret" not in str(caught.value)


def test_both_spellings_of_a_rejection_are_recognised():
    assert {"unauthorized", "http_401"} <= client.UNAUTHORIZED_CODES
