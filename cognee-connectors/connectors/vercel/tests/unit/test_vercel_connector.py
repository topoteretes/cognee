"""The Vercel descriptor exposes a caller-supplied token and selected project IDs.

Hosts construct sources with explicit project_id, team_id/account_id and resource_name;
the descriptor does not discover projects, refresh credentials, or provide OAuth.
"""

import pytest

from cognee_connector_vercel import CONNECTOR, build_vercel_service, vercel_source
from cognee_connector_vercel.connector import DESCRIPTOR
from cognee_connectors_core import registry
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient

TOKEN = "SECRET-vercel-token-123"
ACCOUNT = "1201234567890123"
PROJECTS = ["1201234567890123", "1201234567890124"]
# What a host stores for a connection: an account id of its choosing (a workspace or user
# gid), and the token under the field DescriptorConnector reads a bearer credential from.
CREDENTIAL = FakeCredential(
    provider="vercel", provider_account_id=ACCOUNT, account_label=None, auth_type="api_key"
)
TOKENS = {"access_token": TOKEN}


@pytest.fixture(autouse=True)
def _empty_registry():
    registry.reset_registry()
    yield
    registry.reset_registry()


def _connector(http: FakeHttpClient) -> DescriptorConnector:
    return DescriptorConnector(DESCRIPTOR, http=http)


def test_descriptor_identity():
    assert DESCRIPTOR.provider == "vercel"
    assert DESCRIPTOR.display_name == "Vercel"
    assert DESCRIPTOR.auth == "api_key"
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
async def test_the_oauth_members_refuse_and_vercel_is_never_called():
    http = FakeHttpClient()
    connector = _connector(http)

    with pytest.raises(NotImplementedError, match="api_key"):
        connector.authorize_url("state-1")
    with pytest.raises(NotImplementedError, match="api_key"):
        await connector.exchange_callback("code", {})
    with pytest.raises(NotImplementedError, match="api_key"):
        connector.parse_installation({"access_token": TOKEN})
    assert await connector.refresh(CREDENTIAL, TOKENS) is None
    await connector.revoke_remote(CREDENTIAL, TOKENS)
    assert http.calls == []


@pytest.mark.asyncio
async def test_sync_scope_names_the_dataset_and_hands_the_token_and_the_selection_out():
    http = FakeHttpClient()

    scope = await _connector(http).sync_scope(CREDENTIAL, TOKENS, resources=PROJECTS)

    assert scope.dataset_name == f"vercel_{ACCOUNT}"
    assert scope.items == PROJECTS
    assert scope.credential == TOKEN
    assert scope.content_type is None
    assert http.calls == []


@pytest.mark.asyncio
async def test_a_sync_without_a_selection_is_refused_as_the_source_refuses_it():
    """A descriptor cannot enumerate a workspace, and vercel_source takes no empty selection."""
    with pytest.raises(NotImplementedError):
        await _connector(FakeHttpClient()).sync_scope(CREDENTIAL, TOKENS)
    with pytest.raises(ValueError):
        await _connector(FakeHttpClient()).sync_scope(CREDENTIAL, TOKENS, resources=[])


def test_the_dataset_is_derived_from_the_account_id():
    assert CONNECTOR.dataset_name(CREDENTIAL) == f"vercel_{ACCOUNT}"
    other = FakeCredential(provider="vercel", provider_account_id="Acme Workspace")
    assert CONNECTOR.dataset_name(other) == "vercel_acme_workspace"


def test_no_webhooks_no_polling_no_picker():
    assert CONNECTOR.webhook_verifier() is None
    assert CONNECTOR.webhook_delivery_key(b"{}", {}) is None
    assert CONNECTOR.poll_interval_seconds is None
    assert CONNECTOR.poll_targets(CREDENTIAL) is None
    assert type(CONNECTOR).supports_resource_selection() is False


def test_the_catalog_lists_vercel_as_an_api_key_provider():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("vercel") is CONNECTOR
    assert registry.catalog() == [
        {
            "provider": "vercel",
            "display_name": "Vercel",
            "auth": "api_key",
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
    assert vercel_source.__module__ == "cognee_connector_vercel.source"
    assert build_vercel_service.__module__ == "cognee_connector_vercel.source"
    assert callable(vercel_source) and callable(build_vercel_service)
