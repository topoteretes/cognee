"""Contract tests for the arXiv connector.

arXiv is a public API: no account, no token, and so no token response to replay
through the contract. These pin what the contract object does instead: it names the
provider for hosts and the catalog as a local one (no credential), derives a
connection's dataset from whatever account id a host stores for a selection, passes
the categories or authors a host names through as the scope's items with no
credential, refuses the OAuth members, and never calls anything (the scripted
``FakeHttpClient`` records no request). The rows the source emits carry the same
provider tag, and the host binding has nothing to register.
"""

import pytest

from cognee_connector_arxiv import CONNECTOR, arxiv_source, build_arxiv_service
from cognee_connector_arxiv.cognee_host import register, routers
from cognee_connector_arxiv.connector import DESCRIPTOR
from cognee_connectors_core import registry
from cognee_connectors_core.dlt_conventions import DOCUMENT_SOURCE_ATTR
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient

# What a host stores for a selection: an id of its own choosing (arXiv has no account)
# and nothing to decrypt, so the token payload the host hands over is empty.
CREDENTIAL = FakeCredential(
    provider="arxiv",
    provider_account_id="cs-ai",
    account_label="cs.AI papers",
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
    assert DESCRIPTOR.provider == "arxiv"
    assert DESCRIPTOR.display_name == "arXiv"
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
async def test_sync_scope_names_the_selections_dataset_and_carries_no_credential():
    http = FakeHttpClient()

    scope = await _connector(http).sync_scope(CREDENTIAL, TOKENS, resources=["cs.AI"])

    assert scope.dataset_name == "arxiv_cs_ai"
    assert scope.items == ["cs.AI"]
    assert scope.credential is None, "the arXiv API needs no token"
    assert scope.content_type is None
    assert http.calls == []


@pytest.mark.asyncio
async def test_a_full_sync_is_the_sources_job_not_the_descriptors():
    """A descriptor cannot enumerate papers; arxiv_source pages the API, one resource_name
    per selection."""
    with pytest.raises(NotImplementedError):
        await _connector(FakeHttpClient()).sync_scope(CREDENTIAL, TOKENS)


def test_the_dataset_is_derived_from_the_account_id():
    assert CONNECTOR.dataset_name(CREDENTIAL) == "arxiv_cs_ai"
    other = FakeCredential(provider="arxiv", provider_account_id="My Selection")
    assert CONNECTOR.dataset_name(other) == "arxiv_my_selection"


def test_no_webhooks_no_polling_no_picker():
    assert CONNECTOR.webhook_verifier() is None
    assert CONNECTOR.webhook_delivery_key(b"{}", {}) is None
    assert CONNECTOR.poll_interval_seconds is None
    assert CONNECTOR.poll_targets(CREDENTIAL) is None
    assert type(CONNECTOR).supports_resource_selection() is False


def test_the_catalog_lists_arxiv_as_a_local_provider():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("arxiv") is CONNECTOR
    assert registry.catalog() == [
        {
            "provider": "arxiv",
            "display_name": "arXiv",
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
    assert arxiv_source.__module__ == "cognee_connector_arxiv.source"
    assert callable(arxiv_source)
    assert build_arxiv_service.__module__ == "cognee_connector_arxiv.source"
    assert callable(build_arxiv_service)


def test_the_source_tags_its_rows_with_the_providers_name():
    """cognee stores the tag as ``system_metadata["source"]``; it must be the provider name."""
    source = arxiv_source(categories=["cs.AI"], resource_name="arxiv_contract")

    assert getattr(source, DOCUMENT_SOURCE_ATTR) == DESCRIPTOR.provider


def test_the_host_binding_has_nothing_to_register():
    assert register() is None
    assert routers() == []
