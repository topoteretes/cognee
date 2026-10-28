"""The cognee host binding: ``register()`` replaces the import-time registration cognee had,
and the adapter agrees with the contract object on everything a connected account is keyed on."""

from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("cognee")

from cognee_connector_gmail import CONNECTOR
from cognee_connector_gmail.cognee_host import register, routers
from cognee_connector_gmail.cognee_host.adapter import GoogleGmailIntegration
from cognee_connector_gmail.settings import GoogleGmailSettings, google_gmail_settings

SUBJECT = "110000000000000000001"
TOKEN_RESPONSE = {
    "access_token": "ya29.access",
    "refresh_token": "1//refresh",
    "expires_in": 3599,
    "scope": "openid email https://www.googleapis.com/auth/gmail.readonly",
    "userinfo": {"sub": SUBJECT, "email": "goran@topoteretes.com", "hd": "topoteretes.com"},
}


@pytest.fixture
def integrations(monkeypatch):
    from cognee.modules.integrations import registry

    monkeypatch.setattr(registry, "supported_integrations", {})
    return registry


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(google_gmail_settings, "client_id", "gm_client")
    monkeypatch.setattr(google_gmail_settings, "client_secret", "shhh")
    monkeypatch.setattr(
        google_gmail_settings,
        "redirect_uri",
        "http://localhost:8000/api/v1/integrations/gmail/callback",
    )
    monkeypatch.setattr(google_gmail_settings, "state_secret", "state-secret")
    monkeypatch.setattr(google_gmail_settings, "frontend_base_url", "http://localhost:3000")


def test_importing_the_package_registers_nothing(integrations):
    import cognee_connector_gmail

    assert "gmail" not in integrations.supported_integrations


def test_register_puts_gmail_in_cognees_registry(integrations):
    register()

    integration = integrations.get_integration("gmail")
    assert isinstance(integration, GoogleGmailIntegration)
    assert integration.provider == "gmail" == CONNECTOR.provider
    assert integration.settings_cls is GoogleGmailSettings
    assert integration.resource_selection_key == "selected_label_ids"
    assert integration.resource_selection_allows_all is True


def test_registering_again_replaces_the_adapter_in_place(integrations):
    register()
    first = integrations.get_integration("gmail")
    register()

    assert integrations.get_integration("gmail") is not first
    assert list(integrations.supported_integrations) == ["gmail"]


def test_gmail_mounts_no_routers():
    assert routers() == []


def test_the_adapter_and_the_contract_send_the_browser_to_the_same_place():
    host = urlsplit(GoogleGmailIntegration().authorize_url("the-state"))
    contract = urlsplit(CONNECTOR.authorize_url("the-state"))

    assert (host.scheme, host.netloc, host.path) == (
        contract.scheme,
        contract.netloc,
        contract.path,
    )
    assert parse_qs(host.query) == parse_qs(contract.query)


def test_the_adapter_and_the_contract_read_the_token_response_alike():
    host = GoogleGmailIntegration().parse_installation(TOKEN_RESPONSE)
    contract = CONNECTOR.parse_installation(TOKEN_RESPONSE)

    assert host.provider_account_id == contract.provider_account_id == SUBJECT
    assert host.token_payload == contract.token_payload
    assert host.account_label == contract.account_label
    assert host.scopes == contract.scopes
    # The adapter adds cognee's own empty label selection on top of the shared identity.
    assert contract.provider_metadata.items() <= host.provider_metadata.items()
    assert host.provider_metadata["selected_label_ids"] == []
