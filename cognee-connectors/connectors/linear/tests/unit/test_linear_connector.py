"""Contract tests for the Linear connector.

The provider's real token-response shape — the fixtures the cognee adapter's tests carry —
driven through ``LinearConnector`` with the core's scripted ``FakeHttpClient`` for the token
endpoint and a recording GraphQL stand-in for the install-context and teams queries.
Nothing here needs cognee.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee_connector_linear import CONNECTOR
from cognee_connector_linear.connector import (
    AUTHORIZE_URL,
    DESCRIPTOR,
    INSTALL_CONTEXT_QUERY,
    REVOKE_URL,
    SCOPES,
    TOKEN_URL,
    LinearConnector,
)
from cognee_connector_linear.webhooks import LinearWebhookVerifier
from cognee_connector_linear.workspace import TEAMS_QUERY
from cognee_connectors_core import registry
from cognee_connectors_core.base import ConnectorResource, RefreshedTokens
from cognee_connectors_core.signatures import SignatureScheme
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient, FakeSettings, sign_body

# Linear's token response and the install context the connector merges into it, as the
# cognee adapter's tests carry them.
_TOKEN_RESPONSE = {
    "access_token": "lin_oauth_tok",
    "refresh_token": "lin_refresh_tok",
    "token_type": "Bearer",
    "expires_in": 86400,
    "scope": "read,write,app:assignable,app:mentionable",
}

_INSTALL_CONTEXT = {
    "viewer": {"id": "app-user-1", "name": "cognee-agent"},
    "organization": {"id": "org-1", "name": "Acme Co", "urlKey": "acme-co"},
}

_TOKENS = {"access_token": "lin_access", "refresh_token": "lin_refresh"}


def _team(team_id, **extra):
    return {"id": team_id, "key": team_id.upper(), "name": f"Team {team_id}", **extra}


class FakeGraphQL:
    """Records every GraphQL call and answers the install context and pages of teams."""

    def __init__(self, teams=None, context=None, page_size=100):
        self.calls = []
        self.teams = (
            teams
            if teams is not None
            else [_team("t1", visibility="public"), _team("t2", visibility="private")]
        )
        self.context = context if context is not None else _INSTALL_CONTEXT
        self.page_size = page_size

    async def __call__(self, access_token, query, variables=None):
        self.calls.append((access_token, query, variables))
        if query == INSTALL_CONTEXT_QUERY:
            return dict(self.context)
        if query == TEAMS_QUERY:
            start = int(variables.get("after") or 0)
            end = start + min(variables["first"], self.page_size)
            return {
                "teams": {
                    "nodes": self.teams[start:end],
                    "pageInfo": {"hasNextPage": end < len(self.teams), "endCursor": str(end)},
                }
            }
        raise AssertionError(query)

    def queries(self):
        return [query for _token, query, _variables in self.calls]


def _settings(**extra):
    return FakeSettings(
        client_id="lin_client",
        client_secret="shhh",
        redirect_uri="http://localhost:8000/api/v1/integrations/linear/callback",
        frontend_base_url="http://localhost:3000",
        env_prefix="LINEAR_",
        webhook_secret="hook-secret",
        **extra,
    )


def _connector(http=None, graphql=None, settings=None):
    return LinearConnector(
        DESCRIPTOR,
        settings=settings if settings is not None else _settings(),
        http=http if http is not None else FakeHttpClient(),
        graphql=graphql if graphql is not None else FakeGraphQL(),
    )


def _credential(**overrides):
    fields = {
        "provider": "linear",
        "provider_account_id": "org-1",
        "account_label": "Acme Co",
        "provider_metadata": {"organization_url_key": "Acme-Co"},
        "token_expires_at": datetime.now(timezone.utc) + timedelta(hours=12),
    }
    fields.update(overrides)
    return FakeCredential(**fields)


# -- identity ----------------------------------------------------------------
def test_descriptor_identity():
    assert DESCRIPTOR.provider == "linear"
    assert DESCRIPTOR.display_name == "Linear"
    assert DESCRIPTOR.auth == "oauth2"
    assert DESCRIPTOR.authorize_url == AUTHORIZE_URL
    assert DESCRIPTOR.token_url == TOKEN_URL
    assert DESCRIPTOR.scopes == SCOPES
    assert DESCRIPTOR.signature_scheme is SignatureScheme.BARE_HEX


def test_connector_serves_the_descriptor_as_a_tier_two_connector():
    assert CONNECTOR.provider == DESCRIPTOR.provider
    assert CONNECTOR.display_name == DESCRIPTOR.display_name
    assert CONNECTOR.auth == DESCRIPTOR.auth
    assert registry.tier_of(CONNECTOR) == 2
    assert LinearConnector.supports_resource_selection() is True


def test_the_catalog_entry_advertises_webhooks_and_a_picker():
    registry.reset_registry()
    try:
        registry.use_connector(CONNECTOR)
        (entry,) = registry.catalog()
    finally:
        registry.reset_registry()

    assert entry == {
        "provider": "linear",
        "display_name": "Linear",
        "auth": "oauth2",
        "tier": 2,
        "has_webhooks": True,
        "supports_resource_selection": True,
        "poll_interval_seconds": None,
        "content_type": None,
        "degraded": False,
    }


# -- connect -----------------------------------------------------------------
def test_authorize_url_is_the_agent_install_flow():
    url = _connector().authorize_url("the-state")
    scheme_host_path, _query = url.split("?", 1)

    assert scheme_host_path == AUTHORIZE_URL
    params = parse_qs(urlsplit(url).query)
    assert params["client_id"] == ["lin_client"]
    assert params["redirect_uri"] == ["http://localhost:8000/api/v1/integrations/linear/callback"]
    assert params["response_type"] == ["code"]
    assert params["state"] == ["the-state"]
    # actor=app is what makes this an agent install (an app user in the
    # workspace) rather than acting as the authorizing human.
    assert params["actor"] == ["app"]
    assert params["scope"] == ["read,write,app:assignable,app:mentionable"]


@pytest.mark.asyncio
async def test_exchange_callback_merges_workspace_identity_into_the_token_response():
    http = FakeHttpClient({("POST", TOKEN_URL): (200, dict(_TOKEN_RESPONSE))})
    graphql = FakeGraphQL()

    result = await _connector(http=http, graphql=graphql).exchange_callback("the-code", {})

    assert http.last_call.data == {
        "client_id": "lin_client",
        "client_secret": "shhh",
        "code": "the-code",
        "redirect_uri": "http://localhost:8000/api/v1/integrations/linear/callback",
        "grant_type": "authorization_code",
    }
    # The fresh token is spent on the install-context query so parse_installation
    # can key the credential on the organization id.
    assert graphql.calls == [("lin_oauth_tok", INSTALL_CONTEXT_QUERY, None)]
    assert result["access_token"] == "lin_oauth_tok"
    assert result["viewer"] == _INSTALL_CONTEXT["viewer"]
    assert result["organization"] == _INSTALL_CONTEXT["organization"]


@pytest.mark.asyncio
async def test_exchange_callback_rejects_a_failed_exchange_before_calling_linear():
    http = FakeHttpClient({("POST", TOKEN_URL): (400, {"error": "invalid_grant"})})
    graphql = FakeGraphQL()

    with pytest.raises(RuntimeError, match="token exchange failed: invalid_grant"):
        await _connector(http=http, graphql=graphql).exchange_callback("the-code", {})

    assert graphql.calls == []


@pytest.mark.asyncio
async def test_exchange_callback_rejects_a_response_without_access_token():
    http = FakeHttpClient({("POST", TOKEN_URL): (200, {"token_type": "Bearer"})})

    with pytest.raises(RuntimeError, match="no access_token"):
        await _connector(http=http).exchange_callback("the-code", {})


def test_parse_installation_splits_secrets_from_metadata():
    installation = _connector().parse_installation({**_TOKEN_RESPONSE, **_INSTALL_CONTEXT})

    assert installation.provider_account_id == "org-1"
    # Token material lives ONLY in the encrypted payload; the queryable
    # metadata carries identity, never secrets.
    assert installation.token_payload == {
        "access_token": "lin_oauth_tok",
        "refresh_token": "lin_refresh_tok",
    }
    assert installation.provider_metadata == {
        "app_user_id": "app-user-1",
        "app_user_name": "cognee-agent",
        "organization_name": "Acme Co",
        "organization_url_key": "acme-co",
        "scope": "read,write,app:assignable,app:mentionable",
    }
    assert installation.account_label == "Acme Co"
    assert installation.scopes == "read,write,app:assignable,app:mentionable"
    assert installation.auth_type == "oauth2"
    remaining = installation.token_expires_at - datetime.now(timezone.utc)
    assert timedelta(hours=23, minutes=59) < remaining <= timedelta(hours=24)


def test_parse_installation_without_refresh_token_stores_only_the_access_token():
    token_response = {**_TOKEN_RESPONSE, **_INSTALL_CONTEXT}
    del token_response["refresh_token"]

    installation = _connector().parse_installation(token_response)

    assert installation.token_payload == {"access_token": "lin_oauth_tok"}


def test_parse_installation_without_organization_id_raises():
    with pytest.raises(RuntimeError, match="organization"):
        _connector().parse_installation(dict(_TOKEN_RESPONSE))
    with pytest.raises(RuntimeError, match="organization"):
        _connector().parse_installation({**_TOKEN_RESPONSE, "organization": {"name": "Acme Co"}})


# -- refresh -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_refresh_stores_the_rotated_refresh_token_in_place_of_the_spent_one():
    rotated = {
        "access_token": "new_access",
        "refresh_token": "new_refresh",
        "expires_in": 86399,
        "scope": "read write",
    }
    http = FakeHttpClient({("POST", TOKEN_URL): (200, rotated)})

    refreshed = await _connector(http=http).refresh(_credential(), _TOKENS)

    assert isinstance(refreshed, RefreshedTokens)
    assert http.last_call.data == {
        "client_id": "lin_client",
        "client_secret": "shhh",
        "grant_type": "refresh_token",
        "refresh_token": "lin_refresh",
    }
    # Linear rotates the refresh token: the one just spent stops working.
    assert refreshed.token_payload == {"access_token": "new_access", "refresh_token": "new_refresh"}
    assert refreshed.scopes == "read write"
    remaining = refreshed.token_expires_at - datetime.now(timezone.utc)
    assert timedelta(hours=23, minutes=59) < remaining <= timedelta(seconds=86399)


@pytest.mark.asyncio
async def test_refresh_without_a_refresh_token_is_nothing_to_do():
    http = FakeHttpClient()

    assert await _connector(http=http).refresh(_credential(), {"access_token": "a"}) is None
    assert http.calls == []


@pytest.mark.asyncio
async def test_refresh_reports_a_rejected_refresh_token():
    http = FakeHttpClient({("POST", TOKEN_URL): (400, {"error": "invalid_grant"})})

    # The core vets a refresh answer exactly like a code exchange, so the refusal
    # carries the provider's error code, not a refresh-specific wording.
    with pytest.raises(RuntimeError, match="invalid_grant"):
        await _connector(http=http).refresh(_credential(), _TOKENS)


# -- sync --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sync_scope_on_connect_lists_every_granted_team():
    graphql = FakeGraphQL(teams=[_team("t1"), _team("t2"), _team("t3")], page_size=2)

    scope = await _connector(graphql=graphql).sync_scope(_credential(), _TOKENS)

    assert scope.dataset_name == "linear_acme_co"
    assert scope.items == ["t1", "t2", "t3"]
    assert scope.credential == "lin_access"
    assert scope.credential_refreshable is True
    assert scope.credential_expires_at is not None
    assert scope.content_type is None
    # Paged with the stored token, de-duplicated by the shared pager.
    assert {token for token, _query, _variables in graphql.calls} == {"lin_access"}
    assert graphql.queries() == [TEAMS_QUERY, TEAMS_QUERY]


@pytest.mark.asyncio
async def test_sync_scope_honours_the_stored_selection_without_calling_linear():
    graphql = FakeGraphQL()
    credential = _credential(
        provider_metadata={"organization_url_key": "acme", "selected_team_ids": ["t2", "t9"]}
    )

    scope = await _connector(graphql=graphql).sync_scope(credential, _TOKENS)

    assert scope.items == ["t2", "t9"]
    assert graphql.calls == []


@pytest.mark.asyncio
async def test_sync_scope_with_nothing_selected_has_nothing_to_ingest():
    credential = _credential(provider_metadata={"selected_team_ids": []})

    scope = await _connector().sync_scope(credential, _TOKENS)

    assert scope.items == []


@pytest.mark.asyncio
async def test_sync_scope_narrows_a_webhooks_teams_to_the_selection():
    credential = _credential(provider_metadata={"selected_team_ids": ["t1"]})

    scope = await _connector().sync_scope(credential, _TOKENS, resources=["t1", "t2"])

    assert scope.items == ["t1"]


@pytest.mark.asyncio
async def test_sync_scope_narrows_a_webhooks_teams_to_the_granted_ones():
    graphql = FakeGraphQL()

    scope = await _connector(graphql=graphql).sync_scope(
        _credential(), _TOKENS, resources=["t2", "foreign", "t2"]
    )

    assert scope.items == ["t2"]
    assert graphql.queries() == [TEAMS_QUERY]


@pytest.mark.asyncio
async def test_sync_scope_refuses_an_empty_resource_list():
    with pytest.raises(ValueError, match="empty"):
        await _connector().sync_scope(_credential(), _TOKENS, resources=[])


@pytest.mark.asyncio
async def test_sync_scope_refuses_a_malformed_selection():
    credential = _credential(provider_metadata={"selected_team_ids": "t1"})

    with pytest.raises(ValueError, match="list of team IDs"):
        await _connector().sync_scope(credential, _TOKENS)


@pytest.mark.asyncio
async def test_sync_scope_without_a_stored_token_raises():
    with pytest.raises(RuntimeError, match="holds no access token"):
        await _connector().sync_scope(_credential(), {})


@pytest.mark.asyncio
async def test_list_resources_describes_the_teams_for_the_picker():
    resources = await _connector().list_resources(_credential(), _TOKENS)

    assert resources == [
        ConnectorResource(
            id="t1", name="Team t1", description=None, attributes={"key": "T1", "private": False}
        ),
        ConnectorResource(
            id="t2", name="Team t2", description=None, attributes={"key": "T2", "private": True}
        ),
    ]


def test_dataset_name_is_the_workspaces_url_key_and_falls_back_to_the_account_id():
    connector = _connector()

    assert connector.dataset_name(_credential()) == "linear_acme_co"
    assert connector.dataset_name(_credential(provider_metadata={})) == "linear_org_1"


# -- webhooks ----------------------------------------------------------------
def _delivery(payload: dict) -> bytes:
    return json.dumps({"webhookTimestamp": int(time.time() * 1000), **payload}).encode()


@pytest.mark.asyncio
async def test_the_verifier_is_linears_and_accepts_a_delivery_signed_with_the_secret():
    connector = _connector()
    verifier = connector.webhook_verifier()
    body = _delivery({"type": "Issue", "action": "create", "organizationId": "org-1"})
    headers = {"Linear-Signature": sign_body(SignatureScheme.BARE_HEX, "hook-secret", body)}

    assert isinstance(verifier, LinearWebhookVerifier)
    assert await verifier.verify(body, headers) == body


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "update", "remove"])
@pytest.mark.parametrize(
    ("event_type", "data", "teams"),
    [
        ("Issue", {"teamId": "t1"}, ["t1"]),
        ("Comment", {"issue": {"teamId": "t1"}}, ["t1"]),
        ("Project", {"teamIds": ["t1", "t2"]}, ["t1", "t2"]),
    ],
)
async def test_team_events_name_the_teams_to_resync(action, event_type, data, teams):
    body = _delivery(
        {"type": event_type, "action": action, "organizationId": "org-1", "data": data}
    )

    dispatch = await _connector().handle_webhook(body, {})

    assert dispatch.provider_account_id == "org-1"
    assert dispatch.resources == teams
    assert dispatch.revoke is False
    assert dispatch.delivery_id is None


@pytest.mark.asyncio
async def test_an_app_revoked_delivery_revokes():
    body = _delivery({"type": "OAuthApp", "action": "revoked", "organizationId": "org-1"})

    dispatch = await _connector().handle_webhook(body, {})

    assert dispatch.provider_account_id == "org-1"
    assert dispatch.revoke is True
    assert dispatch.resources == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"type": "AgentSessionEvent", "action": "created", "organizationId": "org-1"},
        {"type": "Issue", "action": "archive", "organizationId": "org-1", "data": {}},
        {"type": "Cycle", "action": "create", "organizationId": "org-1", "data": {"teamId": "t"}},
    ],
)
async def test_other_deliveries_are_understood_but_imply_no_sync(payload):
    dispatch = await _connector().handle_webhook(_delivery(payload), {})

    assert dispatch.provider_account_id == "org-1"
    assert dispatch.resources == []
    assert dispatch.revoke is False


@pytest.mark.asyncio
async def test_a_delivery_without_an_organization_or_a_body_is_dropped():
    connector = _connector()

    dropped = await connector.handle_webhook(_delivery({"type": "Issue", "action": "create"}), {})
    assert dropped.provider_account_id is None

    unparseable = await connector.handle_webhook(b"not json", {})
    assert unparseable.provider_account_id is None


def test_linear_sends_nothing_to_key_dedupe_on():
    # webhookId is the id of the webhook itself, not of a delivery; the verifier's
    # timestamp window is the replay protection.
    body = _delivery({"type": "Issue", "action": "create", "organizationId": "org-1"})

    assert DESCRIPTOR.webhook_delivery_id_path is None
    assert _connector().webhook_delivery_key(body, {}) is None


# -- disconnect --------------------------------------------------------------
@pytest.mark.asyncio
async def test_revoke_remote_revokes_the_refresh_token_then_the_access_token():
    http = FakeHttpClient({("POST", REVOKE_URL): (200, {})})

    await _connector(http=http).revoke_remote(_credential(), _TOKENS)

    assert [call.data for call in http.calls_to("POST", REVOKE_URL)] == [
        {"token": "lin_refresh", "token_type_hint": "refresh_token"},
        {"token": "lin_access", "token_type_hint": "access_token"},
    ]


@pytest.mark.asyncio
async def test_revoke_remote_sends_nothing_when_no_token_is_stored():
    http = FakeHttpClient()

    await _connector(http=http).revoke_remote(_credential(), {})

    assert http.calls == []


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_and_a_failed_revoke_does_not_stop_the_other():
    http = FakeHttpClient({("POST", REVOKE_URL): [ConnectionError("network down"), (400, "nope")]})

    # Best-effort by contract: a network blip must never block a disconnect.
    await _connector(http=http).revoke_remote(_credential(), _TOKENS)

    assert len(http.calls_to("POST", REVOKE_URL)) == 2
