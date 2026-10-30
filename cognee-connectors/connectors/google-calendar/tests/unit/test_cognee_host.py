"""The cognee host binding: ``register()`` replaces the import-time registration cognee had,
and the adapter agrees with the contract object on everything a connected account is keyed on."""

from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("cognee")

from cognee_connector_google_calendar import CONNECTOR
from cognee_connector_google_calendar.cognee_host import register, routers
from cognee_connector_google_calendar.cognee_host.adapter import (
    GoogleCalendarIntegration,
)
from cognee_connector_google_calendar.settings import (
    GoogleCalendarOAuthSettings,
    google_calendar_settings,
)

SUBJECT = "110000000000000000001"
TOKEN_RESPONSE = {
    "access_token": "ya29.access",
    "refresh_token": "1//refresh",
    "expires_in": 3599,
    "scope": "openid email https://www.googleapis.com/auth/calendar.readonly",
    "userinfo": {"sub": SUBJECT, "email": "goran@topoteretes.com", "hd": "topoteretes.com"},
}


@pytest.fixture
def integrations(monkeypatch):
    from cognee.modules.integrations import registry

    monkeypatch.setattr(registry, "supported_integrations", {})
    return registry


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(google_calendar_settings, "client_id", "cal_client")
    monkeypatch.setattr(google_calendar_settings, "client_secret", "shhh")
    monkeypatch.setattr(
        google_calendar_settings,
        "redirect_uri",
        "http://localhost:8000/api/v1/integrations/google_calendar/callback",
    )
    monkeypatch.setattr(google_calendar_settings, "state_secret", "state-secret")
    monkeypatch.setattr(google_calendar_settings, "frontend_base_url", "http://localhost:3000")


def test_importing_the_package_registers_nothing(integrations):
    import cognee_connector_google_calendar

    assert "google_calendar" not in integrations.supported_integrations


def test_register_puts_calendar_in_cognees_registry(integrations):
    register()

    integration = integrations.get_integration("google_calendar")
    assert isinstance(integration, GoogleCalendarIntegration)
    assert integration.provider == "google_calendar" == CONNECTOR.provider
    assert integration.settings_cls is GoogleCalendarOAuthSettings
    assert integration.resource_selection_key == "selected_calendar_ids"
    # Calendars are opt-in on a cognee server: a null selection is refused, never "all".
    assert integration.resource_selection_allows_all is False


def test_registering_again_replaces_the_adapter_in_place(integrations):
    register()
    first = integrations.get_integration("google_calendar")
    register()

    assert integrations.get_integration("google_calendar") is not first
    assert list(integrations.supported_integrations) == ["google_calendar"]


def test_calendar_mounts_no_routers():
    assert routers() == []


def test_the_adapter_and_the_contract_send_the_browser_to_the_same_place():
    host = urlsplit(GoogleCalendarIntegration().authorize_url("the-state"))
    contract = urlsplit(CONNECTOR.authorize_url("the-state"))

    assert (host.scheme, host.netloc, host.path) == (
        contract.scheme,
        contract.netloc,
        contract.path,
    )
    assert parse_qs(host.query) == parse_qs(contract.query)


def test_the_adapter_and_the_contract_read_the_token_response_alike():
    host = GoogleCalendarIntegration().parse_installation(TOKEN_RESPONSE)
    contract = CONNECTOR.parse_installation(TOKEN_RESPONSE)

    assert host.provider_account_id == contract.provider_account_id == SUBJECT
    assert host.token_payload == contract.token_payload
    assert host.account_label == contract.account_label
    assert host.scopes == contract.scopes
    # The adapter adds cognee's own empty calendar selection on top of the shared identity.
    assert contract.provider_metadata.items() <= host.provider_metadata.items()
    assert host.provider_metadata["selected_calendar_ids"] == []
