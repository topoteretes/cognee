"""Contract tests for the Obsidian connector.

Obsidian is a local provider: a vault on disk, no account, no token, and so no token response
to replay through the contract. These pin what the contract object does instead: it names the
provider for hosts and the catalog as a local one, derives a connection's dataset from whatever
account id a host stores for a vault, passes the folders a host names through as the scope's
items with no credential, refuses the OAuth members, and never calls anything (the scripted
``FakeHttpClient`` records no request). The rows the source emits carry the same provider tag,
and the host binding has nothing to register.
"""

import pytest

from cognee_connector_obsidian import CONNECTOR, obsidian_source
from cognee_connector_obsidian.cognee_host import register, routers
from cognee_connector_obsidian.connector import DESCRIPTOR
from cognee_connectors_core import registry
from cognee_connectors_core.dlt_conventions import DOCUMENT_SOURCE_ATTR
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient

# What a host stores for a vault: an id of its own choosing (a vault has no account of its own)
# and nothing to decrypt, so the token payload the host hands over is empty.
CREDENTIAL = FakeCredential(
    provider="obsidian",
    provider_account_id="vault-1",
    account_label="Work vault",
    auth_type="local",
)
TOKENS: dict = {}


@pytest.fixture(autouse=True)
def _empty_registry():
    registry.reset_registry()
    yield
    registry.reset_registry()


def _connector(http: FakeHttpClient) -> DescriptorConnector:
    return DescriptorConnector(DESCRIPTOR, http=http)


def test_descriptor_identity():
    assert DESCRIPTOR.provider == "obsidian"
    assert DESCRIPTOR.display_name == "Obsidian"
    assert DESCRIPTOR.auth == "local"
    assert DESCRIPTOR.signature_scheme is None
    assert DESCRIPTOR.poll_interval_seconds is None
    assert DESCRIPTOR.content_type is None


def test_connector_serves_the_descriptor():
    assert isinstance(CONNECTOR, DescriptorConnector)
    assert CONNECTOR.descriptor is DESCRIPTOR
    assert CONNECTOR.provider == DESCRIPTOR.provider
    assert CONNECTOR.display_name == DESCRIPTOR.display_name
    assert CONNECTOR.auth == DESCRIPTOR.auth
    assert CONNECTOR.settings is None


@pytest.mark.asyncio
async def test_the_oauth_members_refuse_and_nothing_is_ever_called():
    http = FakeHttpClient()
    connector = _connector(http)

    with pytest.raises(NotImplementedError, match="local"):
        connector.authorize_url("state-1")
    with pytest.raises(NotImplementedError, match="local"):
        await connector.exchange_callback("code", {})
    with pytest.raises(NotImplementedError, match="local"):
        connector.parse_installation({"access_token": "nothing-to-store"})
    assert await connector.refresh(CREDENTIAL, TOKENS) is None
    await connector.revoke_remote(CREDENTIAL, TOKENS)
    assert http.calls == []


@pytest.mark.asyncio
async def test_sync_scope_names_the_vaults_dataset_and_carries_no_credential():
    http = FakeHttpClient()

    scope = await _connector(http).sync_scope(
        CREDENTIAL, TOKENS, resources=["Projects/Work", "Journal"]
    )

    assert scope.dataset_name == "obsidian_vault_1"
    assert scope.items == ["Projects/Work", "Journal"]
    assert scope.credential is None, "a vault needs no token"
    assert scope.content_type is None
    assert http.calls == []


@pytest.mark.asyncio
async def test_a_full_sync_is_the_sources_job_not_the_descriptors():
    """A descriptor cannot enumerate a vault; obsidian_source walks it, one resource_name a vault."""
    with pytest.raises(NotImplementedError):
        await _connector(FakeHttpClient()).sync_scope(CREDENTIAL, TOKENS)


def test_the_dataset_is_derived_from_the_account_id():
    assert CONNECTOR.dataset_name(CREDENTIAL) == "obsidian_vault_1"
    other = FakeCredential(provider="obsidian", provider_account_id="My Vault")
    assert CONNECTOR.dataset_name(other) == "obsidian_my_vault"


def test_no_webhooks_no_polling_no_picker():
    assert CONNECTOR.webhook_verifier() is None
    assert CONNECTOR.webhook_delivery_key(b"{}", {}) is None
    assert CONNECTOR.poll_interval_seconds is None
    assert CONNECTOR.poll_targets(CREDENTIAL) is None
    assert type(CONNECTOR).supports_resource_selection() is False


def test_the_catalog_lists_obsidian_as_a_local_provider():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("obsidian") is CONNECTOR
    assert registry.catalog() == [
        {
            "provider": "obsidian",
            "display_name": "Obsidian",
            "auth": "local",
            "tier": 1,
            "has_webhooks": False,
            "supports_resource_selection": False,
            "poll_interval_seconds": None,
            "content_type": None,
            "degraded": False,
        }
    ]


def test_the_bare_descriptor_registers_too():
    """What an entry point may hand a host: use_connector wraps it the same way."""
    registered = registry.use_connector(DESCRIPTOR)

    assert isinstance(registered, DescriptorConnector)
    assert registered.descriptor is DESCRIPTOR


def test_the_package_exports_what_the_entry_points_name():
    assert obsidian_source.__module__ == "cognee_connector_obsidian.source"
    assert callable(obsidian_source)


def test_the_source_tags_its_rows_with_the_providers_name(tmp_path):
    """cognee stores the tag as ``system_metadata["source"]``; it must be the provider name."""
    source = obsidian_source(vault_path=tmp_path, resource_name="obsidian_contract")

    assert getattr(source, DOCUMENT_SOURCE_ATTR) == DESCRIPTOR.provider


def test_the_host_binding_has_nothing_to_register():
    assert register() is None
    assert routers() == []
