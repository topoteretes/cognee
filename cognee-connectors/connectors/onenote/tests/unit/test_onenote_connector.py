"""Contract tests for the OneNote connector.

OneNote is an API-key provider on the contract: the caller brings a delegated Microsoft Graph
token, so there is no token response to replay through it. These pin what the contract object
does instead: it names the provider for hosts and the catalog, derives a connection's dataset
from the stable account id a host stores, hands a stored token out as the sync credential with
the notebook ids a host names as the scope's items, refuses the OAuth members, and never calls
Graph itself (the scripted ``FakeHttpClient`` records no request). The credential it hands out
is what the source sends as the bearer, the rows the source emits carry the same provider tag,
and the host binding has nothing to register.
"""

import pytest
from fake_onenote import Graph

from cognee_connector_onenote import CONNECTOR, list_onenote_notebooks, onenote_source
from cognee_connector_onenote.cognee_host import register, routers
from cognee_connector_onenote.connector import DESCRIPTOR
from cognee_connectors_core import registry
from cognee_connectors_core.dlt_conventions import DOCUMENT_SOURCE_ATTR
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient

TOKEN = "SECRET-graph-delegated-token"
# What a host stores for a connection: the stable account and tenant id the source takes as
# ``account_id``, and the delegated token under the field DescriptorConnector reads a bearer
# credential from.
CREDENTIAL = FakeCredential(
    provider="onenote", provider_account_id="tenant:user", account_label=None, auth_type="api_key"
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
    assert DESCRIPTOR.provider == "onenote"
    assert DESCRIPTOR.display_name == "OneNote"
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
async def test_the_oauth_members_refuse_and_graph_is_never_called():
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
async def test_sync_scope_names_the_accounts_dataset_and_hands_the_stored_token_out():
    http = FakeHttpClient()

    scope = await _connector(http).sync_scope(CREDENTIAL, TOKENS, resources=["notebook-a"])

    assert scope.dataset_name == "onenote_tenant_user"
    assert scope.items == ["notebook-a"]
    assert scope.credential == TOKEN
    assert scope.content_type is None
    assert http.calls == []


@pytest.mark.asyncio
async def test_a_full_sync_is_the_sources_job_not_the_descriptors():
    """A descriptor cannot enumerate an account; list_onenote_notebooks does, one source a notebook."""
    with pytest.raises(NotImplementedError):
        await _connector(FakeHttpClient()).sync_scope(CREDENTIAL, TOKENS)


@pytest.mark.asyncio
async def test_the_credential_the_scope_hands_out_is_the_bearer_the_source_sends():
    """The fake refuses any bearer but its own, so a listing proves the token went through."""
    scope = await _connector(FakeHttpClient()).sync_scope(
        CREDENTIAL, TOKENS, resources=["notebook-a"]
    )
    graph = Graph(token=TOKEN)

    with graph.client() as client:
        notebooks = list_onenote_notebooks(scope.credential, http_client=client)

    assert {notebook["id"] for notebook in notebooks} == {"a", "b"}


def test_the_dataset_is_derived_from_the_account_id():
    assert CONNECTOR.dataset_name(CREDENTIAL) == "onenote_tenant_user"
    other = FakeCredential(provider="onenote", provider_account_id="Contoso Tenant")
    assert CONNECTOR.dataset_name(other) == "onenote_contoso_tenant"


def test_no_webhooks_no_polling_no_picker():
    assert CONNECTOR.webhook_verifier() is None
    assert CONNECTOR.webhook_delivery_key(b"{}", {}) is None
    assert CONNECTOR.poll_interval_seconds is None
    assert CONNECTOR.poll_targets(CREDENTIAL) is None
    assert type(CONNECTOR).supports_resource_selection() is False


def test_the_catalog_lists_onenote_as_an_api_key_provider():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("onenote") is CONNECTOR
    assert registry.catalog() == [
        {
            "provider": "onenote",
            "display_name": "OneNote",
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
    assert onenote_source.__module__ == "cognee_connector_onenote.source"
    assert list_onenote_notebooks.__module__ == "cognee_connector_onenote.source"
    assert callable(onenote_source) and callable(list_onenote_notebooks)


def test_the_source_tags_its_rows_with_the_providers_name():
    """cognee stores the tag as ``system_metadata["source"]``; it must be the provider name."""
    source = onenote_source(TOKEN, notebook_id="a", account_id="tenant:user")

    assert getattr(source, DOCUMENT_SOURCE_ATTR) == DESCRIPTOR.provider


def test_the_host_binding_has_nothing_to_register():
    assert register() is None
    assert routers() == []
