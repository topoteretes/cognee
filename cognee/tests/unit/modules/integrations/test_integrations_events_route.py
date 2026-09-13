"""Unit tests for the generic POST /{provider}/events route and the
callback extensions (callback_params passthrough, on_installed dispatch).

Same fake-provider approach as test_get_integrations_router.py: what's
under test is the router's generic logic, not any real provider. Detached
work is intercepted at the _spawn_background seam so the tests can run the
captured coroutines deterministically instead of racing the event loop.
"""

import asyncio
import importlib
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.modules.integrations.base import (
    OAuthInstallation,
    OAuthIntegration,
    WebhookVerifier,
)
from cognee.modules.integrations.oauth_flow import make_state
from cognee.modules.integrations.registry import supported_integrations, use_integration
from cognee.modules.users.methods import get_authenticated_user

USER_ID = uuid4()

_router_module = importlib.import_module(
    "cognee.api.v1.integrations.routers.get_integrations_router"
)


class _FakeUser:
    id = USER_ID


class _FakeVerifier(WebhookVerifier):
    def __init__(self, fail: bool = False):
        self._fail = fail

    async def verify(self, request):
        if self._fail:
            raise HTTPException(status_code=401, detail="Invalid signature")
        return await request.body()


class _FakeIntegration(OAuthIntegration):
    provider = "fake"
    settings_cls = None

    def __init__(self):
        self.webhook_calls = []
        self.callback_params_seen = []
        self.verifier: WebhookVerifier = _FakeVerifier()

    def authorize_url(self, state):
        return f"https://fake.example/authorize?state={state}"

    async def exchange_code(self, code):
        return {"code": code}

    async def exchange_callback(self, code, params):
        self.callback_params_seen.append(params)
        return await self.exchange_code(code)

    def parse_installation(self, token_response):
        return OAuthInstallation(provider_account_id="ACC1", token_payload={})

    def state_signing_secret(self):
        return "fake-secret"

    def frontend_base_url(self):
        return "https://app.example.com"

    def webhook_verifier(self):
        return self.verifier

    async def handle_webhook(self, raw_body, headers):
        self.webhook_calls.append((raw_body, headers))


class _NoWebhookIntegration(_FakeIntegration):
    provider = "mute"

    def webhook_verifier(self):
        return None


@pytest.fixture
def spawned():
    """Capture detached work instead of racing the TestClient's event loop."""
    captured = []

    def fake_spawn(coro, *, description):
        captured.append((coro, description))

    with patch.object(_router_module, "_spawn_background", side_effect=fake_spawn):
        yield captured

    for coro, _description in captured:
        coro.close()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: _FakeUser()

    before = dict(supported_integrations)
    supported_integrations.clear()
    use_integration(_FakeIntegration())
    use_integration(_NoWebhookIntegration())

    yield TestClient(app)

    supported_integrations.clear()
    supported_integrations.update(before)


def test_events_unknown_provider_404s(client, spawned):
    assert client.post("/api/v1/integrations/notreal/events").status_code == 404


def test_events_provider_without_verifier_404s(client, spawned):
    # Same answer as an unknown provider, so the route leaks nothing about
    # which providers exist but opted out of webhooks.
    assert client.post("/api/v1/integrations/mute/events").status_code == 404
    assert spawned == []


def test_events_failed_verification_is_rejected_before_any_handling(client, spawned):
    supported_integrations["fake"].verifier = _FakeVerifier(fail=True)

    response = client.post("/api/v1/integrations/fake/events", content=b"payload")

    assert response.status_code == 401
    assert spawned == []


def test_events_verified_delivery_is_acked_and_handled_detached(client, spawned):
    integration = supported_integrations["fake"]

    response = client.post(
        "/api/v1/integrations/fake/events",
        content=b'{"action":"created"}',
        headers={"X-GitHub-Event": "installation"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}

    (coro, description) = spawned.pop()
    assert description == "fake webhook handling"
    asyncio.run(coro)
    ((raw_body, headers),) = integration.webhook_calls
    assert raw_body == b'{"action":"created"}'
    # Header keys are lower-cased for the handler.
    assert headers["x-github-event"] == "installation"


def test_callback_passes_the_full_query_string_to_the_adapter(client, spawned):
    integration = supported_integrations["fake"]
    state = make_state(USER_ID, signing_secret="fake-secret")

    # complete_installation is real here — the point is the wiring from the
    # route's query string down to exchange_callback. Only the credential
    # upsert underneath it is stubbed out.
    with patch(
        "cognee.modules.integrations.connect.upsert_credential",
        new=AsyncMock(return_value=type("C", (), {"provider_account_id": "ACC1"})()),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback"
            f"?code=abc&state={state}&installation_id=12345&setup_action=install",
            follow_redirects=False,
        )

    assert "fake=connected" in response.headers["location"]
    (params,) = integration.callback_params_seen
    assert params["installation_id"] == "12345"
    assert params["setup_action"] == "install"


def test_callback_success_fires_on_installed_detached(client, spawned):
    state = make_state(USER_ID, signing_secret="fake-secret")
    credential = type("C", (), {"provider_account_id": "ACC1"})()

    with patch.object(
        _router_module,
        "complete_installation",
        new=AsyncMock(return_value=credential),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={state}",
            follow_redirects=False,
        )

    assert "fake=connected" in response.headers["location"]
    descriptions = [description for _coro, description in spawned]
    assert descriptions == ["fake on_installed hook"]


def test_callback_failure_fires_no_on_installed(client, spawned):
    state = make_state(USER_ID, signing_secret="fake-secret")

    with patch.object(
        _router_module,
        "complete_installation",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={state}",
            follow_redirects=False,
        )

    assert "fake=error_exchange_failed" in response.headers["location"]
    assert spawned == []
