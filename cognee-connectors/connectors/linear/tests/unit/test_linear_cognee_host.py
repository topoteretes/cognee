"""The cognee host binding: ``register()``, ``routers()`` and the adapter's agreement with
the contract object on everything a connected workspace is keyed on."""

import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from cognee.modules.integrations import registry as cognee_registry

from cognee_connector_linear import CONNECTOR
from cognee_connector_linear.cognee_host import register, routers
from cognee_connector_linear.cognee_host.adapter import LinearIntegration
from cognee_connector_linear.cognee_host.verify_linear_signature import (
    LinearWebhookVerifier as ShimVerifier,
)
from cognee_connector_linear.settings import linear_settings
from cognee_connector_linear.webhooks import LinearWebhookVerifier

_TOKEN_RESPONSE = {
    "access_token": "lin_oauth_tok",
    "refresh_token": "lin_refresh_tok",
    "token_type": "Bearer",
    "expires_in": 86400,
    "scope": "read,write,app:assignable,app:mentionable",
}

_INSTALL_CONTEXT = {
    "viewer": {"id": "app-user-1", "name": "cognee-agent"},
    "organization": {"id": "org-1", "name": "Acme Co", "urlKey": "acme-co"},
}


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(linear_settings, "client_id", "lin_client")
    monkeypatch.setattr(linear_settings, "client_secret", "shhh")
    monkeypatch.setattr(
        linear_settings, "redirect_uri", "http://localhost:8000/api/v1/integrations/linear/callback"
    )
    monkeypatch.setattr(linear_settings, "webhook_secret", "hook-secret")
    monkeypatch.setattr(linear_settings, "frontend_base_url", "http://localhost:3000")


def test_register_puts_the_integration_in_cognees_registry(monkeypatch):
    monkeypatch.setattr(cognee_registry, "supported_integrations", {})

    register()

    integration = cognee_registry.get_integration("linear")
    assert isinstance(integration, LinearIntegration)
    assert integration.provider == CONNECTOR.provider


def test_linear_mounts_no_routers():
    assert routers() == []


def test_the_adapter_and_the_contract_send_the_browser_to_the_same_place():
    host = urlsplit(LinearIntegration().authorize_url("the-state"))
    contract = urlsplit(CONNECTOR.authorize_url("the-state"))

    assert (host.scheme, host.netloc, host.path) == (
        contract.scheme,
        contract.netloc,
        contract.path,
    )
    assert parse_qs(host.query) == parse_qs(contract.query)


def test_the_adapter_and_the_contract_read_the_token_response_alike():
    token_response = {**_TOKEN_RESPONSE, **_INSTALL_CONTEXT}

    host = LinearIntegration().parse_installation(token_response)
    contract = CONNECTOR.parse_installation(token_response)

    assert host.provider_account_id == contract.provider_account_id == "org-1"
    assert host.token_payload == contract.token_payload
    assert host.account_label == contract.account_label
    assert host.scopes == contract.scopes
    # The adapter adds cognee's own sync markers on top of the shared identity.
    assert contract.provider_metadata.items() <= host.provider_metadata.items()


def test_the_adapter_and_the_contract_name_the_same_dataset():
    credential = SimpleNamespace(
        provider_account_id="org-1", provider_metadata={"organization_url_key": "Acme-Co"}
    )

    assert LinearIntegration().dataset_name(credential) == "linear_acme_co"
    assert CONNECTOR.dataset_name(credential) == "linear_acme_co"


class _FakeRequest:
    def __init__(self, raw_body: bytes, headers: dict):
        self._raw_body = raw_body
        self.headers = headers

    async def body(self) -> bytes:
        return self._raw_body


@pytest.mark.asyncio
async def test_the_shim_and_the_contracts_verifier_accept_the_same_delivery():
    body = json.dumps(
        {"type": "Issue", "action": "create", "webhookTimestamp": int(time.time() * 1000)}
    ).encode()
    headers = {"linear-signature": hmac.new(b"hook-secret", body, hashlib.sha256).hexdigest()}
    shim = LinearIntegration().webhook_verifier()

    assert isinstance(shim, ShimVerifier)
    assert isinstance(CONNECTOR.webhook_verifier(), LinearWebhookVerifier)
    assert await shim.verify(_FakeRequest(body, headers)) == body
    assert await CONNECTOR.webhook_verifier().verify(body, headers) == body
