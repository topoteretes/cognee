"""Who ended a connection, through the real router and credential store.

The owner disconnects, the provider ends a connection, the owner connects again: each
step is read back through ``GET /{provider}/connection``, nothing mocked between the
route and the database.
"""

import base64
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.modules.integrations import credentials as store
from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.registry import supported_integrations, use_integration
from cognee.modules.users.methods import get_authenticated_user

USER_ID = uuid4()


class _User:
    id = USER_ID


class _Integration(OAuthIntegration):
    provider = "fake"
    settings_cls = None

    def authorize_url(self, state):
        return f"https://fake.example/authorize?state={state}"

    async def exchange_code(self, code):
        return {}

    def parse_installation(self, token_response):
        return OAuthInstallation(provider_account_id="ACC1", token_payload={})

    def frontend_base_url(self):
        return "https://app.example.com"


@pytest.fixture
def api(database, monkeypatch):
    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"0" * 32).decode())
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: _User()
    before = dict(supported_integrations)
    supported_integrations.clear()
    use_integration(_Integration())
    yield httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test/api/v1/integrations"
    )
    supported_integrations.clear()
    supported_integrations.update(before)


async def connect(account_id, user_id=USER_ID):
    await store.upsert_credential(
        provider="fake",
        provider_account_id=account_id,
        user_id=user_id,
        token_payload={"access_token": "token"},
        account_label=f"Label {account_id}",
    )


@pytest.mark.asyncio
async def test_the_page_says_who_ended_the_connection(api):
    async with api as client:
        assert (await client.get("/fake/connection")).json() == {"connected": False}

        await connect("ACC1")
        assert (await client.get("/fake/connection")).json()["connected"] is True

        assert (await client.delete("/fake/connection")).json()["disconnected"] is True
        by_owner = (await client.get("/fake/connection")).json()
        assert (by_owner["connected"], by_owner["disconnectedBy"]) == (False, "owner")
        assert by_owner["providerAccountId"] == "ACC1" and "disconnectedAt" in by_owner

        await connect("ACC1")
        await store.revoke_credential_by_account(
            "fake", "ACC1", revoked_by=store.REVOKED_BY_PROVIDER
        )
        by_provider = (await client.get("/fake/connection?account_id=ACC1")).json()
        assert (by_provider["connected"], by_provider["disconnectedBy"]) == (False, "provider")

        await connect("ACC1")
        connected = (await client.get("/fake/connection")).json()
        assert connected["connected"] is True and "disconnectedBy" not in connected


@pytest.mark.asyncio
async def test_another_users_ended_account_is_a_404(api):
    await connect("THEIRS", user_id=uuid4())
    await store.revoke_credential_by_account("fake", "THEIRS", revoked_by=store.REVOKED_BY_PROVIDER)

    async with api as client:
        response = await client.get("/fake/connection?account_id=THEIRS")

    assert response.status_code == 404
