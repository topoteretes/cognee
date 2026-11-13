"""Contract tests for the Gmail connector: ``CONNECTOR`` against Gmail's real shapes.

The token-response and userinfo fixtures are the ones the cognee adapter tests carry
(``test_google_oauth_adapters.py``); Google's token and userinfo endpoints are faked with
``FakeHttpClient``, and the Gmail REST calls at the client module's own seam, as
``test_gmail_adapter.py`` fakes them. Nothing here imports cognee.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee_connector_gmail import CONNECTOR, client
from cognee_connector_gmail.connector import (
    DESCRIPTOR,
    MAILBOX_RESOURCE_ID,
    SCOPE,
    GmailConnector,
)
from cognee_connector_google import GoogleConnector
from cognee_connector_google import client as google_client
from cognee_connectors_core import registry
from cognee_connectors_core.base import ConnectorResource
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient, FakeSettings

SUBJECT = "110000000000000000001"
TOKEN_RESPONSE = {
    "access_token": "ya29.access",
    "refresh_token": "1//refresh",
    "expires_in": 3599,
    "scope": SCOPE,
    "token_type": "Bearer",
}
WORKSPACE_USERINFO = {"sub": SUBJECT, "email": "goran@topoteretes.com", "hd": "topoteretes.com"}
PERSONAL_USERINFO = {"sub": "110000000000000000002", "email": "someone@gmail.com"}
REDIRECT_URI = "http://localhost:8000/api/v1/integrations/gmail/callback"
# Gmail's labels.list answer, as the adapter tests carry it.
LABELS = {
    "labels": [
        {"id": "INBOX", "name": "Inbox", "type": "system", "messagesTotal": 120},
        {"id": "Label_1", "name": "Project", "type": "user"},
        {"name": "missing ID"},
    ]
}


def settings(**overrides):
    values = {
        "client_id": "gm_client",
        "client_secret": "shhh",
        "redirect_uri": REDIRECT_URI,
        "frontend_base_url": "http://localhost:3000",
        "env_prefix": "GOOGLE_GMAIL_",
    }
    values.update(overrides)
    return FakeSettings(**values)


def connector(http=None, **overrides) -> GmailConnector:
    return GmailConnector(DESCRIPTOR, settings=settings(**overrides), http=http)


def credential(**overrides) -> FakeCredential:
    values = {
        "provider": "gmail",
        "provider_account_id": SUBJECT,
        "account_label": "goran@topoteretes.com",
        "provider_metadata": {"email": "goran@topoteretes.com", "selected_label_ids": []},
    }
    values.update(overrides)
    return FakeCredential(**values)


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.reset_registry()
    yield
    registry.reset_registry()


# ---- Identity and catalog ------------------------------------------------------------


def test_the_descriptor_names_gmail_as_a_polled_oauth2_provider_with_a_read_only_scope():
    assert (DESCRIPTOR.provider, DESCRIPTOR.display_name, DESCRIPTOR.auth) == (
        "gmail",
        "Gmail",
        "oauth2",
    )
    assert DESCRIPTOR.authorize_url == "https://accounts.google.com/o/oauth2/v2/auth"
    assert DESCRIPTOR.token_url == google_client.TOKEN_URL
    # Anything wider than gmail.readonly would let a leaked token send or delete mail.
    assert DESCRIPTOR.scopes == "openid email https://www.googleapis.com/auth/gmail.readonly"
    assert DESCRIPTOR.signature_scheme is None
    assert DESCRIPTOR.poll_interval_seconds == 3600


def test_the_connector_is_tier_two_with_a_picker_and_no_webhooks():
    assert (CONNECTOR.provider, CONNECTOR.display_name, CONNECTOR.auth) == (
        "gmail",
        "Gmail",
        "oauth2",
    )
    assert isinstance(CONNECTOR, GmailConnector)
    assert isinstance(CONNECTOR, GoogleConnector)
    assert registry.tier_of(CONNECTOR) == 2
    assert CONNECTOR.webhook_verifier() is None
    assert type(CONNECTOR).supports_resource_selection() is True
    assert CONNECTOR.poll_targets(credential()) == []
    assert CONNECTOR.content_type is None


def test_the_connector_reads_the_real_settings_object():
    from cognee_connector_gmail.settings import google_gmail_settings

    assert CONNECTOR.settings is google_gmail_settings


def test_registering_the_connector_lists_it_in_the_catalog():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("gmail") is CONNECTOR
    (entry,) = registry.catalog()
    assert entry == {
        "provider": "gmail",
        "display_name": "Gmail",
        "auth": "oauth2",
        "tier": 2,
        "has_webhooks": False,
        "supports_resource_selection": True,
        "poll_interval_seconds": 3600,
        "content_type": None,
        "degraded": False,
    }


# ---- Connect ---------------------------------------------------------------------------


def test_authorize_url_asks_for_a_refresh_token_and_a_separate_grant():
    query = parse_qs(urlsplit(connector().authorize_url("signed-state")).query)

    # Without both of these Google reuses a prior consent and returns no refresh token,
    # which strands the connection an hour after install.
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    # Incremental authorization would fold this grant together with a Drive or Calendar
    # one, and Google's revoke kills a whole grant.
    assert "include_granted_scopes" not in query
    assert query["state"] == ["signed-state"]
    assert query["client_id"] == ["gm_client"]
    assert query["redirect_uri"] == [REDIRECT_URI]
    assert query["scope"] == [SCOPE]


def test_a_missing_setting_is_named_by_its_environment_variable():
    with pytest.raises(RuntimeError, match="GOOGLE_GMAIL_CLIENT_ID is not configured"):
        connector(client_id="").authorize_url("state")


@pytest.mark.asyncio
async def test_exchange_callback_attaches_the_identity_of_who_authorized():
    http = FakeHttpClient(
        {
            ("POST", google_client.TOKEN_URL): (200, TOKEN_RESPONSE),
            ("GET", google_client.USERINFO_URL): (200, WORKSPACE_USERINFO),
        }
    )

    account = await connector(http=http).exchange_callback("auth-code", {})
    installation = connector().parse_installation(account)

    exchange, lookup = http.calls
    assert exchange.data["redirect_uri"] == REDIRECT_URI
    assert lookup.headers == {"Authorization": "Bearer ya29.access"}
    assert installation.provider_account_id == SUBJECT
    assert installation.account_label == "goran@topoteretes.com"
    assert installation.provider_metadata == {
        "email": "goran@topoteretes.com",
        "hosted_domain": "topoteretes.com",
        "account_type": "workspace",
        "scope": SCOPE,
    }
    assert set(installation.token_payload) == {"access_token", "refresh_token"}
    assert "access_token" not in installation.provider_metadata
    assert installation.scopes == SCOPE
    assert installation.token_expires_at > datetime.now(timezone.utc) + timedelta(seconds=3500)


def test_parse_installation_marks_an_account_without_a_hosted_domain_as_personal():
    installation = CONNECTOR.parse_installation({**TOKEN_RESPONSE, "userinfo": PERSONAL_USERINFO})

    assert installation.provider_metadata["account_type"] == "personal"
    assert installation.provider_metadata["hosted_domain"] is None


def test_parse_installation_refuses_a_response_without_a_subject():
    with pytest.raises(RuntimeError, match="userinfo.sub"):
        CONNECTOR.parse_installation({**TOKEN_RESPONSE, "userinfo": {}})


def test_the_dataset_is_keyed_on_the_subject_and_survives_a_renamed_account():
    before = CONNECTOR.dataset_name(credential(account_label="goran@topoteretes.com"))
    after = CONNECTOR.dataset_name(credential(account_label="renamed@topoteretes.com"))

    assert before == after == f"gmail_{SUBJECT}"


# ---- Resources ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_picker_offers_every_label_with_an_id():
    with patch.object(client, "list_labels", AsyncMock(return_value=LABELS)) as list_labels:
        resources = await connector().list_resources(credential(), {"access_token": "tok"})

    list_labels.assert_awaited_once_with("tok")
    assert all(isinstance(resource, ConnectorResource) for resource in resources)
    assert [(item.id, item.name) for item in resources] == [
        ("INBOX", "Inbox"),
        ("Label_1", "Project"),
    ]
    assert resources[0].description is None
    assert resources[0].attributes == {"type": "system", "messages_total": 120}
    assert resources[1].attributes == {"type": "user", "messages_total": None}


@pytest.mark.asyncio
async def test_the_picker_names_a_label_without_a_name_by_its_id():
    labels = {"labels": [{"id": "Label_7"}]}
    with patch.object(client, "list_labels", AsyncMock(return_value=labels)):
        (resource,) = await connector().list_resources(credential(), {"access_token": "tok"})

    assert (resource.id, resource.name) == ("Label_7", "Label_7")


@pytest.mark.asyncio
async def test_the_picker_refuses_a_payload_without_an_access_token():
    with (
        patch.object(client, "list_labels", AsyncMock()) as list_labels,
        pytest.raises(RuntimeError, match=f"{SUBJECT} holds no access token"),
    ):
        await connector().list_resources(credential(), {"refresh_token": "r"})

    list_labels.assert_not_awaited()


# ---- Sync scope ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_scope_without_a_selection_is_the_whole_mailbox():
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    with patch.object(client, "list_labels", AsyncMock()) as list_labels:
        scope = await connector().sync_scope(
            credential(token_expires_at=expires), {"access_token": "tok", "refresh_token": "r"}
        )

    # The mailbox needs no listing: the source syncs it with label_ids=None.
    list_labels.assert_not_awaited()
    assert scope.items == [MAILBOX_RESOURCE_ID] == ["mailbox"]
    assert scope.item_resource_ids == scope.items
    assert scope.dataset_name == f"gmail_{SUBJECT}"
    assert scope.content_type is None
    assert scope.credential == "tok"
    assert scope.credential_expires_at == expires
    assert scope.credential_refreshable is True
    assert scope.options == {"label_ids": None}


@pytest.mark.asyncio
async def test_sync_scope_takes_the_labels_a_host_names_once_each():
    scope = await connector().sync_scope(
        credential(),
        {"access_token": "tok"},
        resources=["INBOX", "Label_1", "INBOX"],
    )

    assert scope.items == ["INBOX", "Label_1"]
    assert scope.item_resource_ids == ["INBOX", "Label_1"]
    assert scope.options == {"label_ids": ["INBOX", "Label_1"]}
    assert scope.credential == "tok"
    assert scope.credential_refreshable is False


@pytest.mark.asyncio
async def test_sync_scope_refuses_an_empty_selection():
    with pytest.raises(ValueError, match="empty"):
        await connector().sync_scope(credential(), {"access_token": "tok"}, resources=[])


@pytest.mark.asyncio
async def test_sync_scope_refuses_a_payload_without_an_access_token():
    with pytest.raises(RuntimeError, match="holds no access token"):
        await connector().sync_scope(credential(), {})
