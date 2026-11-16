"""Contract tests for the Todoist connector.

Todoist is an API-key provider here (a token the caller holds), so there is no token
response to replay through the contract. These pin what the contract object does
instead: it names the provider for hosts and the catalog, derives a connection's dataset
from its account id, hands a stored token out as the sync credential — for the whole
account when no selection is named, which the base descriptor refuses but Todoist's Sync
API supports (its ``"*"`` cursor enumerates the account) — refuses the OAuth members,
and never calls Todoist (the scripted ``FakeHttpClient`` records no request).
"""

import pytest

from cognee_connector_todoist import CONNECTOR, build_todoist_service, todoist_source
from cognee_connector_todoist.connector import DESCRIPTOR, TodoistConnector
from cognee_connectors_core import registry
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient

TOKEN = "SECRET-todoist-token-123"
ACCOUNT = "9988776655"
PROJECTS = ["22114433", "22114434"]
# What a host stores for a connection: an account id of its choosing, and the token under
# the field DescriptorConnector reads a bearer credential from.
CREDENTIAL = FakeCredential(
    provider="todoist", provider_account_id=ACCOUNT, account_label=None, auth_type="api_key"
)
TOKENS = {"access_token": TOKEN}


@pytest.fixture(autouse=True)
def _empty_registry():
    registry.reset_registry()
    yield
    registry.reset_registry()


def _connector(http: FakeHttpClient) -> TodoistConnector:
    return TodoistConnector(DESCRIPTOR, http=http)


def test_descriptor_identity():
    assert DESCRIPTOR.provider == "todoist"
    assert DESCRIPTOR.display_name == "Todoist"
    assert DESCRIPTOR.auth == "api_key"
    assert DESCRIPTOR.signature_scheme is None
    assert DESCRIPTOR.poll_interval_seconds is None
    assert DESCRIPTOR.content_type is None


def test_connector_serves_the_descriptor():
    assert isinstance(CONNECTOR, DescriptorConnector)
    assert type(CONNECTOR) is TodoistConnector
    assert CONNECTOR.descriptor is DESCRIPTOR
    assert CONNECTOR.provider == DESCRIPTOR.provider
    assert CONNECTOR.display_name == DESCRIPTOR.display_name
    assert CONNECTOR.auth == DESCRIPTOR.auth
    assert CONNECTOR.settings is None


@pytest.mark.asyncio
async def test_the_oauth_members_refuse_and_todoist_is_never_called():
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
async def test_sync_scope_without_a_selection_is_the_whole_account():
    """What the base descriptor refuses, Todoist allows: its sync enumerates the account."""
    http = FakeHttpClient()

    scope = await _connector(http).sync_scope(CREDENTIAL, TOKENS)

    assert scope.dataset_name == f"todoist_{ACCOUNT}"
    assert scope.items == []
    assert scope.credential == TOKEN
    assert scope.content_type is None
    assert http.calls == []


@pytest.mark.asyncio
async def test_sync_scope_with_a_selection_hands_the_token_and_the_projects_out():
    http = FakeHttpClient()

    scope = await _connector(http).sync_scope(CREDENTIAL, TOKENS, resources=PROJECTS)

    assert scope.dataset_name == f"todoist_{ACCOUNT}"
    assert scope.items == PROJECTS
    assert scope.credential == TOKEN
    assert http.calls == []


@pytest.mark.asyncio
async def test_an_empty_selection_is_refused_as_the_source_refuses_it():
    with pytest.raises(ValueError):
        await _connector(FakeHttpClient()).sync_scope(CREDENTIAL, TOKENS, resources=[])


def test_the_dataset_is_derived_from_the_account_id():
    assert CONNECTOR.dataset_name(CREDENTIAL) == f"todoist_{ACCOUNT}"
    other = FakeCredential(provider="todoist", provider_account_id="My Account")
    assert CONNECTOR.dataset_name(other) == "todoist_my_account"


def test_no_webhooks_no_polling_no_picker():
    assert CONNECTOR.webhook_verifier() is None
    assert CONNECTOR.webhook_delivery_key(b"{}", {}) is None
    assert CONNECTOR.poll_interval_seconds is None
    assert CONNECTOR.poll_targets(CREDENTIAL) is None
    assert type(CONNECTOR).supports_resource_selection() is False


def test_the_catalog_lists_todoist_as_an_api_key_provider():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("todoist") is CONNECTOR
    assert registry.catalog() == [
        {
            "provider": "todoist",
            "display_name": "Todoist",
            "auth": "api_key",
            "tier": 2,
            "has_webhooks": False,
            "supports_resource_selection": False,
            "poll_interval_seconds": None,
            "content_type": None,
            "degraded": False,
        }
    ]


def test_the_bare_descriptor_registers_as_a_plain_descriptor_connector():
    """What an entry point may hand a host: use_connector wraps it the standard way.

    The wrapper is a plain ``DescriptorConnector``: the bare descriptor does not carry
    the ``TodoistConnector.sync_scope`` override (a whole-account sync without a
    selection), so a host that syncs without one must register ``CONNECTOR`` itself.
    """
    registered = registry.use_connector(DESCRIPTOR)

    assert type(registered) is DescriptorConnector
    assert registered.descriptor is DESCRIPTOR


def test_the_package_exports_what_the_entry_points_name():
    assert todoist_source.__module__ == "cognee_connector_todoist.source"
    assert build_todoist_service.__module__ == "cognee_connector_todoist.source"
    assert callable(todoist_source) and callable(build_todoist_service)
