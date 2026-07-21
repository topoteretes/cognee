"""Unit tests for cognee.modules.integrations.connect.complete_installation.

upsert_credential itself is exercised by test_credentials.py — this only
checks that complete_installation wires exchange_code -> parse_installation
-> upsert_credential correctly, with the right fields going to the right
upsert_credential parameter.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.connect import complete_installation

USER_ID = uuid4()


class _FakeIntegration(OAuthIntegration):
    provider = "fake"
    settings_cls = None

    def __init__(self, installation: OAuthInstallation, code_seen: list):
        self._installation = installation
        self._code_seen = code_seen

    def authorize_url(self, state):
        raise NotImplementedError

    async def exchange_code(self, code):
        self._code_seen.append(code)
        return {"raw": "response"}

    def parse_installation(self, token_response):
        assert token_response == {"raw": "response"}
        return self._installation

    def state_signing_secret(self):
        raise NotImplementedError

    def frontend_base_url(self):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_wires_exchange_code_through_to_upsert_credential():
    installation = OAuthInstallation(
        provider_account_id="ACC1",
        token_payload={"access_token": "secret"},
        provider_metadata={"bot_user_id": "U1"},
        account_label="Acme",
        scopes="read,write",
    )
    code_seen: list = []
    integration = _FakeIntegration(installation, code_seen)

    with patch(
        "cognee.modules.integrations.connect.upsert_credential", new=AsyncMock(return_value="credential")
    ) as upsert:
        result = await complete_installation(integration, code="the-code", user_id=USER_ID)

    assert code_seen == ["the-code"]
    assert result == "credential"
    upsert.assert_awaited_once_with(
        provider="fake",
        user_id=USER_ID,
        provider_account_id="ACC1",
        token_payload={"access_token": "secret"},
        account_label="Acme",
        auth_type="oauth2",
        scopes="read,write",
        provider_metadata={"bot_user_id": "U1"},
        token_expires_at=None,
    )


@pytest.mark.asyncio
async def test_propagates_exchange_code_errors():
    class _FailingIntegration(_FakeIntegration):
        async def exchange_code(self, code):
            raise RuntimeError("provider rejected the code")

    integration = _FailingIntegration(OAuthInstallation(provider_account_id="x", token_payload={}), [])

    with pytest.raises(RuntimeError, match="provider rejected the code"):
        await complete_installation(integration, code="bad-code", user_id=USER_ID)
