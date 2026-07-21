"""Unit tests for the generic {provider} dispatch in get_integrations_router.

A fake OAuthIntegration is registered under provider "fake" so these tests
exercise the router's own generic logic (dispatch, error-to-redirect
mapping, unknown-provider 404) without depending on Slack or any real
network call. Slack's own OAuth mechanics are covered separately by
test_slack_adapter.py and test_oauth_state.py.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.credentials import CrossUserConflictError
from cognee.modules.integrations.registry import supported_integrations, use_integration
from cognee.modules.users.methods import get_authenticated_user

USER_ID = uuid4()


class _FakeUser:
    id = USER_ID


class _FakeIntegration(OAuthIntegration):
    provider = "fake"
    settings_cls = None

    def authorize_url(self, state):
        return f"https://fake.example/authorize?state={state}"

    async def exchange_code(self, code):
        return {"code": code}

    def parse_installation(self, token_response):
        return OAuthInstallation(provider_account_id="ACC1", token_payload={})

    def state_signing_secret(self):
        return "fake-secret"

    def frontend_base_url(self):
        return "https://app.example.com"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: _FakeUser()

    before = dict(supported_integrations)
    supported_integrations.clear()
    use_integration(_FakeIntegration())

    yield TestClient(app)

    supported_integrations.clear()
    supported_integrations.update(before)


def test_unknown_provider_404s_on_every_route(client):
    assert client.post("/api/v1/integrations/notreal/authorize").status_code == 404
    assert client.get("/api/v1/integrations/notreal/callback").status_code == 404
    assert client.get("/api/v1/integrations/notreal/connection").status_code == 404
    assert client.delete("/api/v1/integrations/notreal/connection").status_code == 404


def test_authorize_returns_the_integrations_own_url(client):
    response = client.post("/api/v1/integrations/fake/authorize")
    assert response.status_code == 200
    assert response.json()["authorizeUrl"].startswith("https://fake.example/authorize?state=")


def test_authorize_surfaces_missing_config_as_503(client):
    with patch(
        "cognee.api.v1.integrations.routers.get_integrations_router.make_state",
        side_effect=RuntimeError("FAKE_SIGNING_SECRET is not configured"),
    ):
        response = client.post("/api/v1/integrations/fake/authorize", follow_redirects=False)
    assert response.status_code == 503


def test_callback_with_error_param_redirects_cancelled(client):
    response = client.get(
        "/api/v1/integrations/fake/callback?error=access_denied", follow_redirects=False
    )
    assert response.status_code in (302, 307)
    assert "fake=cancelled" in response.headers["location"]


def test_callback_with_invalid_state_redirects_error(client):
    response = client.get(
        "/api/v1/integrations/fake/callback?code=abc&state=garbage", follow_redirects=False
    )
    assert "fake=error_invalid_state" in response.headers["location"]


def test_callback_surfaces_missing_frontend_url_as_503_not_a_raw_crash(client):
    # error="" so we hit the "cancelled" branch, the earliest _frontend_redirect
    # call in callback() — proves the guard applies before any real work runs,
    # not just on the success path.
    with patch.object(
        supported_integrations["fake"],
        "frontend_base_url",
        side_effect=RuntimeError("FAKE_FRONTEND_BASE_URL is not configured"),
    ):
        response = client.get(
            "/api/v1/integrations/fake/callback?error=access_denied", follow_redirects=False
        )
    assert response.status_code == 503


def test_callback_success_redirects_connected(client):
    from cognee.modules.integrations.oauth_flow import make_state

    state = make_state(user_id=USER_ID, signing_secret="fake-secret")
    with patch(
        "cognee.api.v1.integrations.routers.get_integrations_router.complete_installation",
        new=AsyncMock(return_value=type("C", (), {"provider_account_id": "ACC1"})()),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={state}", follow_redirects=False
        )
    assert "fake=connected" in response.headers["location"]


def test_callback_cross_user_conflict_redirects_already_connected(client):
    from cognee.modules.integrations.oauth_flow import make_state

    state = make_state(user_id=USER_ID, signing_secret="fake-secret")
    with patch(
        "cognee.api.v1.integrations.routers.get_integrations_router.complete_installation",
        new=AsyncMock(side_effect=CrossUserConflictError("ACC1")),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={state}", follow_redirects=False
        )
    assert "fake=error_already_connected" in response.headers["location"]


def test_callback_unexpected_error_redirects_exchange_failed(client):
    from cognee.modules.integrations.oauth_flow import make_state

    state = make_state(user_id=USER_ID, signing_secret="fake-secret")
    with patch(
        "cognee.api.v1.integrations.routers.get_integrations_router.complete_installation",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        response = client.get(
            f"/api/v1/integrations/fake/callback?code=abc&state={state}", follow_redirects=False
        )
    assert "fake=error_exchange_failed" in response.headers["location"]


def test_connection_status_reports_disconnected_by_default(client):
    with patch(
        "cognee.api.v1.integrations.routers.get_integrations_router.get_active_credential_for_user",
        new=AsyncMock(return_value=None),
    ):
        response = client.get("/api/v1/integrations/fake/connection")
    assert response.json() == {"connected": False}


def test_connection_status_reports_connected_with_generic_fields(client):
    fake_credential = type(
        "Cred",
        (),
        {
            "account_label": "Acme",
            "provider_account_id": "ACC1",
            "created_at": __import__("datetime").datetime(2026, 1, 1),
        },
    )()
    with patch(
        "cognee.api.v1.integrations.routers.get_integrations_router.get_active_credential_for_user",
        new=AsyncMock(return_value=fake_credential),
    ):
        response = client.get("/api/v1/integrations/fake/connection")
    body = response.json()
    assert body["connected"] is True
    assert body["accountLabel"] == "Acme"
    assert body["providerAccountId"] == "ACC1"


def test_disconnect_calls_revoke_remote_and_revokes_locally(client):
    fake_credential = type("Cred", (), {"provider_account_id": "ACC1"})()
    integration = supported_integrations["fake"]
    with (
        patch(
            "cognee.api.v1.integrations.routers.get_integrations_router.get_active_credential_for_user",
            new=AsyncMock(return_value=fake_credential),
        ),
        patch.object(integration, "revoke_remote", new=AsyncMock()) as revoke_remote,
        patch(
            "cognee.api.v1.integrations.routers.get_integrations_router.revoke_credential_by_account",
            new=AsyncMock(return_value=True),
        ) as revoke_local,
    ):
        response = client.delete("/api/v1/integrations/fake/connection")

    assert response.json() == {"disconnected": True}
    revoke_remote.assert_awaited_once_with(fake_credential)
    revoke_local.assert_awaited_once_with("fake", "ACC1")


def test_disconnect_with_no_active_credential_reports_false(client):
    with patch(
        "cognee.api.v1.integrations.routers.get_integrations_router.get_active_credential_for_user",
        new=AsyncMock(return_value=None),
    ):
        response = client.delete("/api/v1/integrations/fake/connection")
    assert response.json() == {"disconnected": False}
