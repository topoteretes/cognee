"""The cognee host binding: ``register()`` replaces the import-time registration cognee had."""

import pytest

pytest.importorskip("cognee")

from cognee_connector_google_drive.cognee_host import register, routers
from cognee_connector_google_drive.cognee_host.adapter import GoogleDriveIntegration
from cognee_connector_google_drive.settings import GoogleDriveSettings


@pytest.fixture
def integrations(monkeypatch):
    from cognee.modules.integrations import registry

    monkeypatch.setattr(registry, "supported_integrations", {})
    return registry


def test_importing_the_package_registers_nothing(integrations):
    import cognee_connector_google_drive

    assert "google_drive" not in integrations.supported_integrations


def test_register_puts_drive_in_cognees_registry(integrations):
    register()

    integration = integrations.get_integration("google_drive")
    assert isinstance(integration, GoogleDriveIntegration)
    assert integration.provider == "google_drive"
    assert integration.settings_cls is GoogleDriveSettings
    assert integration.resource_selection_key == "selected_folder_ids"
    assert integration.resource_selection_allows_all is True


def test_registering_again_replaces_the_adapter_in_place(integrations):
    register()
    first = integrations.get_integration("google_drive")
    register()

    assert integrations.get_integration("google_drive") is not first
    assert list(integrations.supported_integrations) == ["google_drive"]


def test_drive_mounts_no_routers():
    assert routers() == []
