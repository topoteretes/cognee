"""Unit tests for cognee_connectors_core.http.

``AiohttpClient`` is exercised against a fake session so nothing touches the
network; what matters is how it decodes a provider's answer, because that is
what ``DescriptorConnector`` reasons about. The last test ports the cloud's
"wrongly encoded body" case end to end (cognee-saas-backend, 2026-10-26).
"""

import json

import aiohttp
import pytest

from cognee_connectors_core import http as http_module
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.http import AiohttpClient
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeSettings


class _FakeResponse:
    """Stands in for aiohttp's response, raising whatever .json() would."""

    def __init__(self, status=200, *, json_result=None, json_error=None, text="<html>"):
        self.status = status
        self._json_result = json_result
        self._json_error = json_error
        self._text = text
        self.json_kwargs = None

    async def json(self, content_type="application/json"):
        self.json_kwargs = {"content_type": content_type}
        if self._json_error is not None:
            raise self._json_error
        return self._json_result

    async def text(self, errors="strict"):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.fixture
def session(monkeypatch):
    """Install a fake session factory; the test sets ``holder.response`` first."""
    holder = {"session": None}

    def factory(**kwargs):
        holder["kwargs"] = kwargs
        return holder["session"]

    monkeypatch.setattr(http_module.aiohttp, "ClientSession", factory)

    def install(response: _FakeResponse) -> _FakeSession:
        holder["session"] = _FakeSession(response)
        return holder["session"]

    install.holder = holder
    return install


@pytest.mark.asyncio
async def test_post_form_sends_the_form_and_decodes_json(session):
    fake = session(_FakeResponse(201, json_result={"access_token": "tok"}))

    status, body = await AiohttpClient().post_form(
        "https://p.example/token", {"code": "c"}, {"X-Extra": "1"}
    )

    assert (status, body) == (201, {"access_token": "tok"})
    method, url, kwargs = fake.calls[0]
    assert (method, url) == ("POST", "https://p.example/token")
    assert kwargs["data"] == {"code": "c"}
    assert kwargs["headers"] == {"Accept": "application/json", "X-Extra": "1"}


@pytest.mark.asyncio
async def test_json_is_decoded_whatever_the_content_type_says(session):
    """GitHub answers token exchanges as JSON under a form-encoded type."""
    response = _FakeResponse(json_result={"ok": True})
    session(response)

    await AiohttpClient().get_json("https://p.example/me")

    assert response.json_kwargs == {"content_type": None}


@pytest.mark.asyncio
async def test_get_json_asks_for_json_and_lets_a_caller_override(session):
    fake = session(_FakeResponse(json_result={}))

    await AiohttpClient().get_json("https://p.example/me", {"Accept": "text/plain"})

    assert fake.calls[0][2]["headers"] == {"Accept": "text/plain"}


@pytest.mark.parametrize(
    "error",
    [
        json.JSONDecodeError("Expecting value", "<html>", 0),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        aiohttp.ContentTypeError(request_info=None, history=()),
    ],
    ids=["not_json", "wrong_encoding", "content_type"],
)
@pytest.mark.asyncio
async def test_a_body_that_is_not_json_comes_back_as_text(session, error):
    session(_FakeResponse(502, json_error=error, text="<html>Bad gateway</html>"))

    status, body = await AiohttpClient().get_json("https://p.example/me")

    assert (status, body) == (502, "<html>Bad gateway</html>")


@pytest.mark.asyncio
async def test_the_timeout_is_applied_to_every_session(session):
    session(_FakeResponse(json_result={}))

    await AiohttpClient(timeout_seconds=7).get_json("https://p.example/me")

    assert session.holder["kwargs"]["timeout"].total == 7


def test_the_default_timeout_matches_the_slack_flow():
    assert http_module.DEFAULT_TIMEOUT_SECONDS == 30


@pytest.mark.asyncio
async def test_exchange_callback_treats_a_wrongly_encoded_body_like_a_non_json_one(session):
    """aiohttp's ``.json()`` raises ``UnicodeDecodeError`` for a body that is
    not valid in its declared charset — a different exception from the
    ``ContentTypeError``/``JSONDecodeError`` pair, but the same real-world
    cause (a provider mid-outage answering something malformed) and it must
    land on the same graceful failure, not an unhandled 500.
    """
    error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    session(_FakeResponse(200, json_error=error, text="�"))
    descriptor = ConnectorDescriptor(
        provider="testprov",
        display_name="Test Provider",
        authorize_url="https://testprov.example/oauth/authorize",
        token_url="https://testprov.example/oauth/token",
        account_id_path="account.id",
        poll_interval_seconds=900,
    )
    connector = DescriptorConnector(descriptor, FakeSettings(), AiohttpClient())

    with pytest.raises(RuntimeError, match=r"non-JSON body \(HTTP 200\)"):
        await connector.exchange_callback("code", {})


def test_the_default_client_is_aiohttp():
    descriptor = ConnectorDescriptor(provider="apollo", display_name="Apollo", auth="api_key")

    assert isinstance(DescriptorConnector(descriptor).http, AiohttpClient)


@pytest.mark.asyncio
async def test_a_307_from_the_token_endpoint_is_not_followed():
    """A followed 307/308 would POST the client secret and the refresh token to
    the host the response names, and hand back that host's answer. Exercised
    against real servers on loopback, so it is aiohttp's behaviour being checked,
    not a keyword argument.
    """
    from aiohttp import web

    received = []

    async def attacker(request):
        received.append(dict(await request.post()))
        return web.json_response({"access_token": "attacker-token"})

    async def token_endpoint(request):
        raise web.HTTPTemporaryRedirect(location=str(request.app["attacker_url"]))

    attacker_app = web.Application()
    attacker_app.router.add_post("/collect", attacker)
    attacker_runner = web.AppRunner(attacker_app)
    await attacker_runner.setup()
    attacker_site = web.TCPSite(attacker_runner, "127.0.0.1", 0)
    await attacker_site.start()
    attacker_port = attacker_runner.addresses[0][1]

    token_app = web.Application()
    token_app["attacker_url"] = f"http://127.0.0.1:{attacker_port}/collect"
    token_app.router.add_post("/token", token_endpoint)
    token_runner = web.AppRunner(token_app)
    await token_runner.setup()
    token_site = web.TCPSite(token_runner, "127.0.0.1", 0)
    await token_site.start()
    token_port = token_runner.addresses[0][1]

    try:
        status, _body = await AiohttpClient().post_form(
            f"http://127.0.0.1:{token_port}/token",
            {"client_secret": "dummy-secret", "refresh_token": "dummy-refresh"},
        )

        descriptor = ConnectorDescriptor(
            provider="testprov",
            display_name="Test Provider",
            authorize_url="https://testprov.example/oauth/authorize",
            token_url=f"http://127.0.0.1:{token_port}/token",
            account_id_path="account.id",
            poll_interval_seconds=900,
        )
        connector = DescriptorConnector(descriptor, FakeSettings(), AiohttpClient())
        with pytest.raises(RuntimeError, match=r"HTTP 307"):
            await connector.exchange_callback("code", {})
    finally:
        await token_runner.cleanup()
        await attacker_runner.cleanup()

    assert status == 307
    assert received == []


@pytest.mark.asyncio
async def test_get_json_does_not_follow_redirects(session):
    fake = session(_FakeResponse(302, json_result={}))

    status, _body = await AiohttpClient().get_json("https://testprov.example/me")

    assert status == 302
    assert fake.calls[0][2]["allow_redirects"] is False
