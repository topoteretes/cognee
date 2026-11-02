"""The cognee host binding: ``register()`` replaces the import-time registration cognee had,
and the adapter agrees with the contract object on everything a connected workspace is
keyed on."""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("cognee")

from cognee.modules.integrations import ingestion

from cognee_connector_notion import CONNECTOR
from cognee_connector_notion.cognee_host import register, routers
from cognee_connector_notion.cognee_host.adapter import NotionIntegration
from cognee_connector_notion.settings import NotionSettings, notion_settings

TOKEN_RESPONSE = {
    "access_token": "secret_access",
    "refresh_token": "secret_refresh",
    "workspace_id": "ws-1",
    "workspace_name": "Topoteretes",
    "workspace_icon": "https://example.com/icon.png",
    "bot_id": "bot-1",
}


@pytest.fixture
def integrations(monkeypatch):
    from cognee.modules.integrations import registry

    monkeypatch.setattr(registry, "supported_integrations", {})
    return registry


@pytest.fixture
def _settings(monkeypatch):
    monkeypatch.setattr(notion_settings, "client_id", "n_client")
    monkeypatch.setattr(notion_settings, "client_secret", "shhh")
    monkeypatch.setattr(
        notion_settings, "redirect_uri", "http://localhost:8000/api/v1/integrations/notion/callback"
    )
    monkeypatch.setattr(notion_settings, "frontend_base_url", "http://localhost:3000")


def test_importing_the_package_registers_nothing(integrations):
    import cognee_connector_notion

    assert "notion" not in integrations.supported_integrations


def test_register_puts_notion_in_cognees_registry(integrations):
    register()

    integration = integrations.get_integration("notion")
    assert isinstance(integration, NotionIntegration)
    assert integration.provider == "notion"
    assert integration.settings_cls is NotionSettings
    assert integration.resource_selection_key == "selected_roots"
    assert integration.resource_selection_allows_all is False


def test_registering_again_replaces_the_adapter_in_place(integrations):
    register()
    first = integrations.get_integration("notion")
    register()

    assert integrations.get_integration("notion") is not first
    assert list(integrations.supported_integrations) == ["notion"]


def test_register_after_cognees_own_registration_puts_this_adapter_in_charge(integrations):
    """cognee 1.6.x registers its bundled Notion adapter when ``cognee.api.client`` is
    imported (a plain assignment into the registry), so ``register()`` has to run after
    that import to be the adapter the routes use; before it, the bundled one wins."""
    bundled = pytest.importorskip("cognee.modules.integrations.notion.adapter")

    integrations.use_integration(bundled.NotionIntegration())
    register()

    assert type(integrations.get_integration("notion")) is NotionIntegration
    assert list(integrations.supported_integrations) == ["notion"]


def test_notion_mounts_no_routers():
    assert routers() == []


def test_the_adapter_and_the_contract_send_the_browser_to_the_same_place(_settings):
    host = urlsplit(NotionIntegration().authorize_url("the-state"))
    contract = urlsplit(CONNECTOR.authorize_url("the-state"))

    assert (host.scheme, host.netloc, host.path) == (
        contract.scheme,
        contract.netloc,
        contract.path,
    )
    assert parse_qs(host.query) == parse_qs(contract.query)


def test_the_adapter_and_the_contract_read_the_token_response_alike():
    host = NotionIntegration().parse_installation(TOKEN_RESPONSE)
    contract = CONNECTOR.parse_installation(TOKEN_RESPONSE)

    assert host.provider_account_id == contract.provider_account_id == "ws-1"
    assert host.token_payload == contract.token_payload
    assert host.account_label == contract.account_label
    assert host.provider_metadata == contract.provider_metadata
    assert host.token_expires_at is None and contract.token_expires_at is None


def test_the_adapter_and_the_contract_name_the_same_dataset():
    credential = SimpleNamespace(
        provider_account_id="ws-1", account_label="Topoteretes", provider_metadata={}
    )

    assert (
        NotionIntegration().dataset_name(credential)
        == CONNECTOR.dataset_name(credential)
        == ingestion.dataset_name_for_account("notion", "", "ws-1")
    )
