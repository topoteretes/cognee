"""The Linear GraphQL client tells a rejected token apart from other failures."""

import pytest

from cognee.modules.integrations.linear import client
from cognee.tests.unit.modules.integrations.test_linear_adapter import _FakeResponse, _FakeSession


def _fake_session(monkeypatch, response):
    session = _FakeSession(response)
    monkeypatch.setattr(client.aiohttp, "ClientSession", lambda **_kwargs: session)


@pytest.mark.asyncio
async def test_a_401_raises_the_unauthorized_error(monkeypatch):
    _fake_session(monkeypatch, _FakeResponse(401))

    with pytest.raises(client.LinearUnauthorizedError, match="HTTP 401"):
        await client.graphql("tok", "query Viewer { viewer { id } }")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 500])
async def test_other_failures_are_not_reported_as_a_rejected_token(monkeypatch, status):
    _fake_session(monkeypatch, _FakeResponse(status))

    with pytest.raises(RuntimeError, match=f"HTTP {status}") as caught:
        await client.graphql("tok", "query Viewer { viewer { id } }")

    assert not isinstance(caught.value, client.LinearUnauthorizedError)


@pytest.mark.asyncio
async def test_a_400_with_the_ratelimited_code_is_a_rate_limit(monkeypatch):
    body = {"errors": [{"message": "secret text", "extensions": {"code": "RATELIMITED"}}]}
    _fake_session(monkeypatch, _FakeResponse(400, payload=body))

    with pytest.raises(client.LinearRateLimitedError, match="RATELIMITED") as caught:
        await client.graphql("tok", "query Viewer { viewer { id } }")

    assert isinstance(caught.value, RuntimeError)
    assert "secret text" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _FakeResponse(400, payload={"errors": [{"extensions": {"code": "INVALID_INPUT"}}]}),
        _FakeResponse(400, payload={}),
        _FakeResponse(400, json_error=ValueError("not json")),
    ],
)
async def test_any_other_400_stays_a_plain_failure(monkeypatch, response):
    _fake_session(monkeypatch, response)

    with pytest.raises(RuntimeError, match="HTTP 400") as caught:
        await client.graphql("tok", "query Viewer { viewer { id } }")

    assert not isinstance(caught.value, client.LinearRateLimitedError)
