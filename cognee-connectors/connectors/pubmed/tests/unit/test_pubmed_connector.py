"""Contract tests for the PubMed connector.

PubMed is an API-key provider with an *optional* key, so there is no token response to
replay through the contract. These pin what the contract object does instead: it names
the provider for hosts and the catalog, derives a connection's dataset from its account
id, hands a stored key out as the sync credential, refuses the OAuth members, and never
calls NCBI (the scripted ``FakeHttpClient`` records no request).
"""

import pytest

from cognee_connector_pubmed import CONNECTOR, build_pubmed_service, pubmed_source
from cognee_connector_pubmed.connector import DESCRIPTOR
from cognee_connectors_core import registry
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient

KEY = "SECRET-pubmed-key-123"
# What a host stores for a connection: the resource_name it chose as the account id, and
# the key under the field DescriptorConnector reads an API-key credential from.
CREDENTIAL = FakeCredential(
    provider="pubmed",
    provider_account_id="pubmed_alzheimers",
    account_label=None,
    auth_type="api_key",
)
TOKENS = {"access_token": KEY}


@pytest.fixture(autouse=True)
def _empty_registry():
    registry.reset_registry()
    yield
    registry.reset_registry()


def _connector(http: FakeHttpClient) -> DescriptorConnector:
    return DescriptorConnector(DESCRIPTOR, http=http)


def test_descriptor_identity():
    assert DESCRIPTOR.provider == "pubmed"
    assert DESCRIPTOR.display_name == "PubMed"
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
async def test_the_oauth_members_refuse_and_pubmed_is_never_called():
    http = FakeHttpClient()
    connector = _connector(http)

    with pytest.raises(NotImplementedError, match="api_key"):
        connector.authorize_url("state-1")
    with pytest.raises(NotImplementedError, match="api_key"):
        await connector.exchange_callback("code", {})
    with pytest.raises(NotImplementedError, match="api_key"):
        connector.parse_installation({"access_token": KEY})
    assert await connector.refresh(CREDENTIAL, TOKENS) is None
    await connector.revoke_remote(CREDENTIAL, TOKENS)
    assert http.calls == []


@pytest.mark.asyncio
async def test_sync_scope_names_the_connections_dataset_and_hands_the_stored_key_out():
    http = FakeHttpClient()

    scope = await _connector(http).sync_scope(CREDENTIAL, TOKENS, resources=["articles"])

    assert scope.dataset_name == "pubmed_pubmed_alzheimers"
    assert scope.items == ["articles"]
    assert scope.credential == KEY
    assert scope.content_type is None
    assert http.calls == []


@pytest.mark.asyncio
async def test_a_full_sync_is_the_sources_job_not_the_descriptors():
    """A descriptor cannot enumerate a query; pubmed_source lists it, one resource_name a query."""
    with pytest.raises(NotImplementedError):
        await _connector(FakeHttpClient()).sync_scope(CREDENTIAL, TOKENS)


def test_the_dataset_is_derived_from_the_account_id():
    assert CONNECTOR.dataset_name(CREDENTIAL) == "pubmed_pubmed_alzheimers"
    other = FakeCredential(provider="pubmed", provider_account_id="My Query")
    assert CONNECTOR.dataset_name(other) == "pubmed_my_query"


def test_no_webhooks_no_polling_no_picker():
    assert CONNECTOR.webhook_verifier() is None
    assert CONNECTOR.webhook_delivery_key(b"{}", {}) is None
    assert CONNECTOR.poll_interval_seconds is None
    assert CONNECTOR.poll_targets(CREDENTIAL) is None
    assert type(CONNECTOR).supports_resource_selection() is False


def test_the_catalog_lists_pubmed_as_an_api_key_provider():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("pubmed") is CONNECTOR
    assert registry.catalog() == [
        {
            "provider": "pubmed",
            "display_name": "PubMed",
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
    assert pubmed_source.__module__ == "cognee_connector_pubmed.source"
    assert build_pubmed_service.__module__ == "cognee_connector_pubmed.source"
    assert callable(pubmed_source) and callable(build_pubmed_service)
