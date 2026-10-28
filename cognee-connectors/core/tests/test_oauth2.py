"""Unit tests for cognee_connectors_core.oauth2.

Ported from cloud-backend tests/modules/integrations/test_framework_oauth2.py
(cognee-saas-backend, 2026-10-26) onto the host-agnostic contract: settings
come from a ``FakeSettings``, HTTP from a scripted ``FakeHttpClient``, token
material is passed in as ``tokens``, and delivery dedupe — which the cloud did
in-process — is reported to the host on the dispatch instead of acted on.
The slug tests moved to ``test_base.py`` with the functions they test.

``DescriptorConnector`` is what makes Tier 1 real — a standard OAuth2 provider
served with no Python of its own — so these tests stand in for the tests each
of those providers will never write.

Two behaviours get the most attention. Refusing to store a credential with no
account id, because the alternative failure is "connected, silently never
syncs". And dropping every unrecognized webhook shape instead of raising,
because the provider controls this body and can change it without notice;
malformed input should be acknowledged as a no-op rather than retried forever.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from cognee_connectors_core.base import RefreshedTokens
from cognee_connectors_core.descriptor import ConnectorDescriptor
from cognee_connectors_core.errors import WebhookVerificationError
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.signatures import SignatureScheme
from cognee_connectors_core.testing import (
    FakeCredential,
    FakeHttpClient,
    FakeSettings,
    webhook_headers,
)

_TOKEN_URL = "https://testprov.example/oauth/token"
_WEBHOOK_SECRET = "whsec-1"
_TOKENS = {"access_token": "tok"}


def _descriptor(**overrides) -> ConnectorDescriptor:
    base = {
        "provider": "testprov",
        "display_name": "Test Provider",
        "authorize_url": "https://testprov.example/oauth/authorize",
        "token_url": _TOKEN_URL,
        "account_id_path": "account.id",
        "account_label_path": "account.name",
        "signature_scheme": SignatureScheme.SHA256_PREFIX,
        "signature_header": "x-signature",
        "webhook_account_id_path": "account.id",
        "webhook_event_header": "x-event",
    }
    return ConnectorDescriptor(**{**base, **overrides})


def _connector(http=None, settings=None, **overrides) -> DescriptorConnector:
    return DescriptorConnector(
        _descriptor(**overrides),
        settings if settings is not None else FakeSettings(webhook_secret=_WEBHOOK_SECRET),
        http if http is not None else FakeHttpClient(),
    )


def _credential(**overrides) -> FakeCredential:
    """A stand-in for the host's credential row; only the read attributes matter here."""
    return FakeCredential(**overrides)


def _polling_overrides() -> dict:
    return {
        "signature_scheme": None,
        "signature_header": None,
        "webhook_account_id_path": None,
        "webhook_event_header": None,
        "poll_interval_seconds": 900,
    }


# ---- identity ----------------------------------------------------------


def test_the_connector_takes_its_identity_from_the_descriptor():
    connector = _connector()

    assert connector.provider == "testprov"
    assert connector.display_name == "Test Provider"
    assert connector.auth == "oauth2"
    assert repr(connector) == "<DescriptorConnector provider='testprov'>"


# ---- authorize_url -----------------------------------------------------


def test_authorize_url_uses_the_settings_client_id_and_redirect_uri():
    settings = FakeSettings(client_id="cid", redirect_uri="https://host.example/cb")

    url = _connector(settings=settings, scopes="read").authorize_url("signed-state")

    assert url.startswith("https://testprov.example/oauth/authorize?")
    assert "client_id=cid" in url
    assert "redirect_uri=https%3A%2F%2Fhost.example%2Fcb" in url
    assert "state=signed-state" in url
    assert "scope=read" in url


def test_authorize_url_names_the_missing_setting():
    with pytest.raises(RuntimeError, match="TESTPROV_CLIENT_ID is not configured"):
        _connector(settings=FakeSettings(client_id="")).authorize_url("s")


def test_settings_are_read_at_call_time_not_captured():
    """A value the host rotates on the settings object takes effect next request."""
    settings = FakeSettings(client_id="first")
    connector = _connector(settings=settings)

    assert "client_id=first" in connector.authorize_url("s")
    settings.client_id = "second"
    assert "client_id=second" in connector.authorize_url("s")


def test_a_connector_built_without_settings_says_so():
    connector = DescriptorConnector(_descriptor(), None, FakeHttpClient())

    with pytest.raises(RuntimeError, match="built without settings"):
        connector.authorize_url("s")


# ---- exchange_callback -------------------------------------------------


@pytest.mark.asyncio
async def test_exchange_callback_posts_the_authorization_code_grant():
    http = FakeHttpClient({("POST", _TOKEN_URL): (200, {"access_token": "tok"})})
    settings = FakeSettings(
        client_id="cid", client_secret="csec", redirect_uri="https://host.example/cb"
    )

    body = await _connector(http=http, settings=settings).exchange_callback("the-code", {})

    assert body == {"access_token": "tok"}
    assert http.last_call.method == "POST"
    assert http.last_call.url == _TOKEN_URL
    assert http.last_call.data == {
        "client_id": "cid",
        "client_secret": "csec",
        "code": "the-code",
        "redirect_uri": "https://host.example/cb",
        "grant_type": "authorization_code",
    }
    assert http.last_call.headers == {"Accept": "application/json"}


@pytest.mark.asyncio
async def test_exchange_callback_ignores_the_callback_params():
    """A standard OAuth2 callback carries nothing load-bearing beyond the code."""
    http = FakeHttpClient([(200, {"access_token": "tok"})])

    await _connector(http=http).exchange_callback("code", {"installation_id": "evil"})

    assert "installation_id" not in http.last_call.data


@pytest.mark.asyncio
async def test_exchange_callback_rejects_a_non_json_body():
    """A provider mid-outage answers HTML; the client hands it back as text."""
    http = FakeHttpClient([(502, "<html>Bad gateway</html>")])

    with pytest.raises(RuntimeError, match=r"non-JSON body \(HTTP 502\)"):
        await _connector(http=http).exchange_callback("code", {})


@pytest.mark.asyncio
async def test_exchange_callback_rejects_a_non_object_response():
    http = FakeHttpClient([(200, ["not", "an", "object"])])

    with pytest.raises(RuntimeError, match="non-object response"):
        await _connector(http=http).exchange_callback("code", {})


@pytest.mark.asyncio
async def test_exchange_callback_rejects_an_error_reported_with_http_200():
    """The body is checked before the status: GitHub reports failure this way."""
    http = FakeHttpClient([(200, {"error": "bad_verification_code"})])

    with pytest.raises(RuntimeError, match="testprov token exchange failed: bad_verification_code"):
        await _connector(http=http).exchange_callback("code", {})


@pytest.mark.asyncio
async def test_exchange_callback_rejects_an_error_status_with_a_clean_body():
    http = FakeHttpClient([(400, {"access_token": "tok"})])

    with pytest.raises(RuntimeError, match="returned HTTP 400"):
        await _connector(http=http).exchange_callback("code", {})


@pytest.mark.asyncio
async def test_exchange_callback_names_the_missing_secret():
    http = FakeHttpClient([(200, {"access_token": "tok"})])

    with pytest.raises(RuntimeError, match="TESTPROV_CLIENT_SECRET is not configured"):
        await _connector(http=http, settings=FakeSettings(client_secret="")).exchange_callback(
            "code", {}
        )


@pytest.mark.asyncio
async def test_exchange_callback_merges_what_resolve_account_returns():
    """The Tier 2 hook: identity the token response omits, read with one more call."""

    class _Tier2(DescriptorConnector):
        async def resolve_account(self, token_response):
            _status, body = await self.http.get_json(
                "https://testprov.example/me",
                {"Authorization": f"Bearer {token_response['access_token']}"},
            )
            return {"account": body}

    http = FakeHttpClient(
        {
            ("POST", _TOKEN_URL): (200, {"access_token": "tok"}),
            ("GET", "https://testprov.example/me"): (200, {"id": "org-1", "name": "Org"}),
        }
    )
    connector = _Tier2(_descriptor(), FakeSettings(), http)

    account_data = await connector.exchange_callback("code", {})
    installation = connector.parse_installation(account_data)

    assert account_data == {"access_token": "tok", "account": {"id": "org-1", "name": "Org"}}
    assert installation.provider_account_id == "org-1"
    assert installation.account_label == "Org"
    assert http.calls_to("GET", "https://testprov.example/me")[0].headers == {
        "Authorization": "Bearer tok"
    }


@pytest.mark.asyncio
async def test_resolve_account_keys_win_over_the_token_response():
    class _Tier2(DescriptorConnector):
        async def resolve_account(self, token_response):
            return {"account": {"id": "resolved"}}

    http = FakeHttpClient([(200, {"access_token": "tok", "account": {"id": "from-token"}})])

    account_data = await _Tier2(_descriptor(), FakeSettings(), http).exchange_callback("c", {})

    assert account_data["account"] == {"id": "resolved"}


@pytest.mark.asyncio
async def test_resolve_account_defaults_to_leaving_the_response_alone():
    http = FakeHttpClient([(200, {"access_token": "tok", "account": {"id": "a"}})])

    account_data = await _connector(http=http).exchange_callback("c", {})

    assert account_data == {"access_token": "tok", "account": {"id": "a"}}


# ---- parse_installation ------------------------------------------------


def test_parse_installation_maps_the_token_response():
    installation = _connector().parse_installation(
        {
            "access_token": "tok-abc",
            "account": {"id": "acct-1", "name": "Acme Corp"},
        }
    )

    assert installation.provider_account_id == "acct-1"
    assert installation.token_payload == {"access_token": "tok-abc"}
    assert installation.account_label == "Acme Corp"
    assert installation.auth_type == "oauth2"


def test_parse_installation_coerces_a_numeric_account_id():
    """Several providers send the id as a JSON number; storage wants a string."""
    installation = _connector().parse_installation({"access_token": "tok", "account": {"id": 42}})

    assert installation.provider_account_id == "42"


def test_parse_installation_refuses_a_response_with_no_account_id():
    """A credential with no account id can be stored but never reached again.

    A webhook carries only the account id, so such a connection would look
    healthy in the UI and silently never sync — refuse it at connect instead.
    """
    with pytest.raises(RuntimeError, match="account.id"):
        _connector().parse_installation({"access_token": "tok"})


def test_parse_installation_copies_only_known_token_fields():
    """Sweeping the whole response into the encrypted blob protects nothing."""
    installation = _connector().parse_installation(
        {
            "access_token": "tok",
            "account": {"id": "a"},
            "team_name": "public info",
            "api_domain": "example.com",
        }
    )

    assert installation.token_payload == {"access_token": "tok"}


def test_parse_installation_keeps_a_bot_token_too():
    installation = _connector().parse_installation(
        {"access_token": "user-tok", "bot_access_token": "bot-tok", "account": {"id": "a"}}
    )

    assert installation.token_payload == {
        "access_token": "user-tok",
        "bot_access_token": "bot-tok",
    }


def test_parse_installation_refuses_an_oversized_token_field():
    """A compromised or misconfigured token endpoint must not get an unbounded
    write into the host's ciphertext column. The message must not carry the
    actual size.
    """
    oversized = "x" * 65537

    with pytest.raises(RuntimeError, match="too large to store") as excinfo:
        _connector().parse_installation({"access_token": oversized, "account": {"id": "a"}})

    assert "65537" not in str(excinfo.value)


def test_parse_installation_accepts_a_token_field_at_the_boundary():
    at_limit = "x" * 65536

    installation = _connector().parse_installation(
        {"access_token": at_limit, "account": {"id": "a"}}
    )

    assert installation.token_payload == {"access_token": at_limit}


def test_parse_installation_keeps_a_refresh_token_for_later_rotation():
    installation = _connector().parse_installation(
        {"access_token": "tok", "refresh_token": "refresh-abc", "account": {"id": "a"}}
    )

    assert installation.token_payload["refresh_token"] == "refresh-abc"


def test_parse_installation_converts_expires_in_to_an_absolute_time():
    """Relative expiry is meaningless once persisted."""
    installation = _connector().parse_installation(
        {"access_token": "tok", "expires_in": 3600, "account": {"id": "a"}}
    )

    assert installation.token_expires_at is not None
    assert installation.token_expires_at.tzinfo is not None
    assert installation.token_expires_at > datetime.now(timezone.utc)


def test_parse_installation_ignores_an_unparseable_expires_in():
    """A shape surprise in one optional field must not fail the whole connect."""
    installation = _connector().parse_installation(
        {"access_token": "tok", "expires_in": "soon", "account": {"id": "a"}}
    )

    assert installation.token_expires_at is None


def test_parse_installation_treats_an_expires_in_of_zero_as_already_expired():
    """``0`` means "already expired, right now" — real information from the
    provider, not "nothing to report".
    """
    installation = _connector().parse_installation(
        {"access_token": "tok", "expires_in": 0, "account": {"id": "a"}}
    )

    assert installation.token_expires_at is not None
    assert installation.token_expires_at <= datetime.now(timezone.utc)


def test_parse_installation_ignores_an_expires_in_too_large_to_add_to_a_datetime():
    """``int()`` is arbitrary precision, so a huge ``expires_in`` parses cleanly
    and only blows up in the ``datetime + timedelta`` addition, which raises
    ``OverflowError``. Must degrade the same as any other unparseable value.
    """
    installation = _connector().parse_installation(
        {"access_token": "tok", "expires_in": 10**12, "account": {"id": "a"}}
    )

    assert installation.token_expires_at is None


def test_parse_installation_collects_declared_metadata_paths():
    connector = _connector(
        metadata_paths=(("account_type", "account.type"), ("region", "account.region"))
    )

    installation = connector.parse_installation(
        {
            "access_token": "tok",
            "account": {"id": "a", "type": "organization", "region": "eu"},
        }
    )

    assert installation.provider_metadata == {"account_type": "organization", "region": "eu"}


def test_parse_installation_omits_absent_metadata_paths():
    connector = _connector(metadata_paths=(("missing", "account.nope"),))

    installation = connector.parse_installation({"access_token": "tok", "account": {"id": "a"}})

    assert installation.provider_metadata == {}


def test_parse_installation_reads_the_scopes_path():
    connector = _connector(scopes_path="scope")

    installation = connector.parse_installation(
        {"access_token": "tok", "scope": "read write", "account": {"id": "a"}}
    )

    assert installation.scopes == "read write"


def test_parse_installation_refuses_an_empty_token_payload():
    """A provider that renamed its token field (camelCase, a wrapper object)
    must not be stored as a credential with nothing to authenticate with.
    """
    with pytest.raises(RuntimeError, match="access_token"):
        _connector().parse_installation({"accessToken": "tok", "account": {"id": "a"}})


def test_parse_installation_refuses_a_non_scalar_account_id():
    """A dict/list at account_id_path means the path is wrong, not that the
    id is that object — str()-ing it would silently become the routing key
    every webhook for this account is matched against.
    """
    with pytest.raises(RuntimeError, match="non-scalar"):
        _connector().parse_installation(
            {"access_token": "tok", "account": {"id": {"nested": "object"}}}
        )


def test_parse_installation_drops_a_non_scalar_optional_field(caplog):
    connector = _connector(account_label_path="account.name")
    account_data = {
        "access_token": "tok",
        "account": {"id": "A1", "name": {"nested": 1}},
    }

    with caplog.at_level("WARNING"):
        installation = connector.parse_installation(account_data)

    assert installation.account_label is None
    assert "non-scalar" in caplog.text


def test_parse_installation_drops_a_non_scalar_metadata_path(caplog):
    connector = _connector(metadata_paths=(("region", "account.region"),))
    account_data = {
        "access_token": "tok",
        "account": {"id": "A1", "region": {"deep": {"x": 1}}},
    }

    with caplog.at_level("WARNING"):
        installation = connector.parse_installation(account_data)

    assert installation.provider_metadata == {}
    assert "non-scalar" in caplog.text


def test_parse_installation_keeps_a_non_string_scalar_metadata_value():
    """Shape-checked, not stringified — a number stays a number."""
    connector = _connector(metadata_paths=(("seats", "account.seats"),))
    installation = connector.parse_installation(
        {"access_token": "tok", "account": {"id": "A1", "seats": 12}}
    )

    assert installation.provider_metadata == {"seats": 12}


def test_parse_installation_refuses_an_empty_account_id():
    """An empty id resolves, so neither the None nor the shape check sees it.

    ``account_slug("")`` falls back to the literal ``"account"``, so every
    connection whose id came back blank would land in one shared
    ``{provider}_account`` dataset.
    """
    with pytest.raises(RuntimeError, match="empty value"):
        _connector().parse_installation({"access_token": "tok", "account": {"id": "   "}})


def test_parse_installation_refuses_a_boolean_account_id():
    """``str(False)`` is the literal ``"False"``, which would become the
    provider_account_id every webhook routes against.
    """
    with pytest.raises(RuntimeError, match="boolean"):
        _connector().parse_installation({"access_token": "tok", "account": {"id": False}})


def test_parse_installation_still_accepts_a_zero_account_id():
    """``bool`` is checked before falsiness on purpose: ``0`` is a plausible id."""
    installation = _connector().parse_installation({"access_token": "tok", "account": {"id": 0}})

    assert installation.provider_account_id == "0"


def test_parse_installation_drops_a_non_finite_float_metadata_value(caplog):
    """``json.loads`` accepts the non-standard ``NaN``/``Infinity`` literals,
    and a non-finite float fails the host's JSON column INSERT.
    """
    connector = _connector(metadata_paths=(("ratio", "account.ratio"),))
    account_data = json.loads('{"access_token": "tok", "account": {"id": "A1", "ratio": NaN}}')

    with caplog.at_level("WARNING"):
        installation = connector.parse_installation(account_data)

    assert installation.provider_metadata == {}
    assert "non-finite" in caplog.text
    # The whole point: what survives must be strict-JSON serializable.
    json.dumps(installation.provider_metadata, allow_nan=False)


def test_parse_installation_drops_a_nul_byte_text_value(caplog):
    """Postgres rejects a NUL byte in a text column outright."""
    connector = _connector(account_label_path="account.name")
    account_data = {"access_token": "tok", "account": {"id": "A1", "name": "ac\x00me"}}

    with caplog.at_level("WARNING"):
        installation = connector.parse_installation(account_data)

    assert installation.account_label is None
    assert "NUL byte" in caplog.text


# ---- refresh -----------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_returns_none_when_there_is_no_refresh_token():
    assert await _connector().refresh(_credential(), {"access_token": "tok"}) is None


@pytest.mark.asyncio
async def test_refresh_posts_the_refresh_token_grant():
    http = FakeHttpClient(
        {("POST", _TOKEN_URL): (200, {"access_token": "new", "expires_in": 3600})}
    )
    settings = FakeSettings(client_id="cid", client_secret="csec")

    refreshed = await _connector(http=http, settings=settings).refresh(
        _credential(), {"access_token": "old", "refresh_token": "r1"}
    )

    assert http.last_call.data == {
        "client_id": "cid",
        "client_secret": "csec",
        "grant_type": "refresh_token",
        "refresh_token": "r1",
    }
    assert isinstance(refreshed, RefreshedTokens)
    assert refreshed.token_payload == {"access_token": "new", "refresh_token": "r1"}
    assert refreshed.token_expires_at is not None
    assert refreshed.token_expires_at > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_refresh_keeps_the_old_refresh_token_when_the_provider_does_not_rotate():
    http = FakeHttpClient([(200, {"access_token": "new"})])

    refreshed = await _connector(http=http).refresh(
        _credential(), {"access_token": "old", "refresh_token": "r1"}
    )

    assert refreshed.token_payload["refresh_token"] == "r1"
    assert refreshed.token_expires_at is None


@pytest.mark.asyncio
async def test_refresh_stores_a_rotated_refresh_token():
    http = FakeHttpClient([(200, {"access_token": "new", "refresh_token": "r2"})])

    refreshed = await _connector(http=http).refresh(
        _credential(), {"access_token": "old", "refresh_token": "r1"}
    )

    assert refreshed.token_payload == {"access_token": "new", "refresh_token": "r2"}


@pytest.mark.asyncio
async def test_refresh_carries_over_keys_it_does_not_know():
    """A field another path stored next to the tokens is not lost by a rotation."""
    http = FakeHttpClient([(200, {"access_token": "new"})])

    refreshed = await _connector(http=http).refresh(
        _credential(), {"access_token": "old", "refresh_token": "r1", "token_uri": "x"}
    )

    assert refreshed.token_payload["token_uri"] == "x"


@pytest.mark.asyncio
async def test_refresh_reads_the_scopes_path():
    http = FakeHttpClient([(200, {"access_token": "new", "scope": "read"})])

    refreshed = await _connector(http=http, scopes_path="scope").refresh(
        _credential(), {"refresh_token": "r1"}
    )

    assert refreshed.scopes == "read"


@pytest.mark.asyncio
async def test_refresh_refuses_an_answer_without_an_access_token():
    http = FakeHttpClient([(200, {"token_type": "bearer"})])

    with pytest.raises(RuntimeError, match="token refresh returned no value"):
        await _connector(http=http).refresh(_credential(), {"refresh_token": "r1"})


@pytest.mark.asyncio
async def test_refresh_vets_the_answer_like_an_exchange():
    http = FakeHttpClient([(200, {"error": "invalid_grant"})])

    with pytest.raises(RuntimeError, match="invalid_grant"):
        await _connector(http=http).refresh(_credential(), {"refresh_token": "r1"})


@pytest.mark.asyncio
async def test_refresh_reports_a_non_json_answer():
    http = FakeHttpClient([(503, "<html>")])

    with pytest.raises(RuntimeError, match=r"token refresh returned a non-JSON body \(HTTP 503\)"):
        await _connector(http=http).refresh(_credential(), {"refresh_token": "r1"})


# ---- handle_webhook ----------------------------------------------------


@pytest.mark.asyncio
async def test_handle_webhook_extracts_the_account_and_resources():
    connector = _connector(webhook_resource_path="repository.full_name")
    body = json.dumps({"account": {"id": "acct-1"}, "repository": {"full_name": "acme/api"}})

    dispatch = await connector.handle_webhook(body.encode(), {"x-event": "push"})

    assert dispatch.provider_account_id == "acct-1"
    assert dispatch.resources == ["acme/api"]
    assert dispatch.revoke is False
    assert dispatch.suspend is False
    assert dispatch.delivery_id is None


@pytest.mark.asyncio
async def test_handle_webhook_accepts_a_list_of_resources():
    """A provider may report one changed resource or many at the same path."""
    connector = _connector(webhook_resource_path="repositories")
    body = json.dumps({"account": {"id": "a"}, "repositories": ["acme/api", "acme/web"]})

    dispatch = await connector.handle_webhook(body.encode(), {})

    assert dispatch.resources == ["acme/api", "acme/web"]


@pytest.mark.asyncio
async def test_handle_webhook_stringifies_a_numeric_resource_id():
    """A JSON int at the resource path (a numeric repository id) must come
    through as a string: selections and sync items are always strings, so an
    int would silently no-op the filter or be rejected by the runner.
    """
    connector = _connector(webhook_resource_path="repository.id")
    body = json.dumps({"account": {"id": "a"}, "repository": {"id": 12345}})

    dispatch = await connector.handle_webhook(body.encode(), {})

    assert dispatch.resources == ["12345"]


@pytest.mark.asyncio
async def test_handle_webhook_stringifies_numeric_resource_ids_in_a_list():
    connector = _connector(webhook_resource_path="repository_ids")
    body = json.dumps({"account": {"id": "a"}, "repository_ids": [111, 222]})

    dispatch = await connector.handle_webhook(body.encode(), {})

    assert dispatch.resources == ["111", "222"]


@pytest.mark.asyncio
async def test_handle_webhook_flags_a_revoke_event():
    connector = _connector(webhook_revoke_events=("uninstalled", "suspended"))
    body = json.dumps({"account": {"id": "acct-1"}})

    dispatch = await connector.handle_webhook(body.encode(), {"x-event": "uninstalled"})

    assert dispatch.revoke is True


@pytest.mark.asyncio
async def test_revoke_is_detected_before_any_resource_lookup():
    """A revoke must land even when the payload carries nothing else usable."""
    connector = _connector(
        webhook_revoke_events=("uninstalled",), webhook_resource_path="repository.full_name"
    )
    body = json.dumps({"account": {"id": "acct-1"}})

    dispatch = await connector.handle_webhook(body.encode(), {"x-event": "uninstalled"})

    assert dispatch.revoke is True
    assert dispatch.resources == []


@pytest.mark.asyncio
async def test_handle_webhook_ignores_events_outside_the_sync_list():
    connector = _connector(
        webhook_sync_events=("push",), webhook_resource_path="repository.full_name"
    )
    body = json.dumps({"account": {"id": "a"}, "repository": {"full_name": "acme/api"}})

    dispatch = await connector.handle_webhook(body.encode(), {"x-event": "star"})

    # Still names the account — the delivery was understood, it just implies no
    # work. Distinct from a dropped delivery, which resolves to no account.
    assert dispatch.provider_account_id == "a"
    assert dispatch.resources == []


@pytest.mark.asyncio
async def test_handle_webhook_matches_an_event_header_written_in_canonical_case():
    """The host lower-cases ``headers``, but nothing forces a descriptor to
    write ``webhook_event_header`` that way — a descriptor using a provider's
    own spelling ("X-GitHub-Event") must still match.
    """
    connector = _connector(
        webhook_event_header="X-Event-Type", webhook_revoke_events=("uninstalled",)
    )
    body = json.dumps({"account": {"id": "a"}})

    dispatch = await connector.handle_webhook(body.encode(), {"x-event-type": "uninstalled"})

    assert dispatch.revoke is True


@pytest.mark.asyncio
async def test_handle_webhook_matches_headers_a_host_did_not_lower_case():
    connector = _connector(webhook_revoke_events=("uninstalled",))
    body = json.dumps({"account": {"id": "a"}})

    dispatch = await connector.handle_webhook(body.encode(), {"X-Event": "uninstalled"})

    assert dispatch.revoke is True


@pytest.mark.asyncio
async def test_handle_webhook_syncs_every_event_when_no_list_is_declared():
    """Over-syncing costs compute; under-syncing silently serves stale data."""
    connector = _connector(webhook_resource_path="repository.full_name")
    body = json.dumps({"account": {"id": "a"}, "repository": {"full_name": "acme/api"}})

    dispatch = await connector.handle_webhook(body.encode(), {"x-event": "anything"})

    assert dispatch.resources == ["acme/api"]


@pytest.mark.parametrize(
    "body",
    [
        b"not json at all",
        b"",
        b'{"truncated":',
        # Not valid UTF-8, which json.loads hits while DECODING the bytes
        # rather than while parsing them — UnicodeDecodeError, a ValueError
        # but not a JSONDecodeError.
        b'{"account": {"id": "A1"}, "repo": "\xff\xfe"}',
    ],
    ids=["garbage", "empty", "truncated", "invalid_utf8"],
)
@pytest.mark.asyncio
async def test_handle_webhook_drops_an_unparseable_body(body):
    """Malformed provider input is a no-op, not a retryable dispatch failure."""
    dispatch = await _connector().handle_webhook(body, {})

    assert dispatch.provider_account_id is None


@pytest.mark.asyncio
async def test_handle_webhook_drops_a_body_with_no_account_id():
    body = json.dumps({"something": "else"})

    dispatch = await _connector().handle_webhook(body.encode(), {})

    assert dispatch.provider_account_id is None


@pytest.mark.asyncio
async def test_handle_webhook_survives_a_resource_path_of_the_wrong_shape():
    connector = _connector(webhook_resource_path="repository.full_name")
    body = json.dumps({"account": {"id": "a"}, "repository": "a-string-not-an-object"})

    dispatch = await connector.handle_webhook(body.encode(), {})

    assert dispatch.resources == []


@pytest.mark.parametrize(
    "account_id_value",
    [{"nested": "object"}, ["a", "list"]],
    ids=["dict", "list"],
)
@pytest.mark.asyncio
async def test_handle_webhook_drops_a_non_scalar_account_id(account_id_value, caplog):
    """Mirrors parse_installation's own non-scalar guard, but on untrusted
    webhook traffic it must degrade rather than raise.
    """
    body = json.dumps({"account": {"id": account_id_value}})

    with caplog.at_level("WARNING"):
        dispatch = await _connector().handle_webhook(body.encode(), {})

    assert dispatch.provider_account_id is None
    assert "non-scalar" in caplog.text


@pytest.mark.asyncio
async def test_handle_webhook_drops_a_non_scalar_resource(caplog):
    connector = _connector(webhook_resource_path="repositories")
    body = json.dumps(
        {"account": {"id": "a"}, "repositories": ["acme/api", {"not": "a resource id"}]}
    )

    with caplog.at_level("WARNING"):
        dispatch = await connector.handle_webhook(body.encode(), {})

    assert dispatch.resources == ["acme/api"]
    assert "non-scalar" in caplog.text


@pytest.mark.asyncio
async def test_handle_webhook_drops_a_non_scalar_scalar_resource_path():
    """The non-list branch of _webhook_resources gets the same guard."""
    connector = _connector(webhook_resource_path="repository")
    body = json.dumps({"account": {"id": "a"}, "repository": {"not": "a scalar"}})

    dispatch = await connector.handle_webhook(body.encode(), {})

    assert dispatch.resources == []


@pytest.mark.asyncio
async def test_handle_webhook_drops_an_empty_account_id():
    """The webhook half of parse_installation's empty-id refusal."""
    body = json.dumps({"account": {"id": ""}}).encode()

    dispatch = await _connector().handle_webhook(body, {"x-event": "push"})

    assert dispatch.provider_account_id is None


@pytest.mark.asyncio
async def test_handle_webhook_drops_a_boolean_account_id():
    body = json.dumps({"account": {"id": True}}).encode()

    dispatch = await _connector().handle_webhook(body, {"x-event": "push"})

    assert dispatch.provider_account_id is None


# ---- handle_webhook: the delivery id ------------------------------------


@pytest.mark.asyncio
async def test_handle_webhook_reports_the_delivery_id_for_the_host_to_claim():
    """Dedupe is the host's; the connector reads the id out of the signed body."""
    connector = _connector(webhook_delivery_id_path="delivery.id")
    body = json.dumps({"account": {"id": "acct-1"}, "delivery": {"id": "dlv-1"}}).encode()

    dispatch = await connector.handle_webhook(body, {})

    assert dispatch.provider_account_id == "acct-1"
    assert dispatch.delivery_id == "dlv-1"


@pytest.mark.asyncio
async def test_handle_webhook_leaves_replays_to_the_host():
    """A byte-identical replay is processed again here — it carries the same
    delivery id, which is what the host claims on.
    """
    connector = _connector(webhook_delivery_id_path="delivery.id")
    body = json.dumps({"account": {"id": "acct-1"}, "delivery": {"id": "dlv-1"}}).encode()

    first = await connector.handle_webhook(body, {})
    replay = await connector.handle_webhook(body, {})

    assert (first.provider_account_id, first.delivery_id) == ("acct-1", "dlv-1")
    assert (replay.provider_account_id, replay.delivery_id) == ("acct-1", "dlv-1")


@pytest.mark.asyncio
async def test_handle_webhook_reads_the_delivery_id_from_the_body_not_a_header():
    """``sha256_prefix``/``bare_hex`` HMAC ``raw_body`` only, so a header
    carries no guarantee it was not changed after signing.
    """
    connector = _connector(webhook_delivery_id_path="delivery.id")
    body = json.dumps({"account": {"id": "A1"}, "delivery": {"id": "D1"}}).encode()

    dispatch = await connector.handle_webhook(body, {"x-event": "push", "x-delivery": "H1"})

    assert dispatch.delivery_id == "D1"


@pytest.mark.asyncio
async def test_handle_webhook_has_no_delivery_id_without_a_path():
    body = json.dumps({"account": {"id": "acct-1"}, "delivery": {"id": "dlv-1"}}).encode()

    dispatch = await _connector().handle_webhook(body, {})

    assert dispatch.delivery_id is None


@pytest.mark.asyncio
async def test_handle_webhook_warns_when_a_declared_delivery_id_is_missing(caplog):
    """Declaring the path is a claim that this connector has replay protection.

    A provider renaming the field turns it off while every delivery still
    verifies. Nothing can be done about it in flight — dropping real
    deliveries over a missing id would be worse — so the log is the whole
    defense.
    """
    connector = _connector(webhook_delivery_id_path="delivery.id")
    body = json.dumps({"account": {"id": "A1"}}).encode()

    with caplog.at_level("WARNING"):
        dispatch = await connector.handle_webhook(body, {"x-event": "push"})

    assert dispatch.provider_account_id == "A1"
    assert dispatch.delivery_id is None
    assert "replay dedupe is inactive" in caplog.text


@pytest.mark.asyncio
async def test_handle_webhook_drops_a_boolean_delivery_id():
    """A JSON ``true`` would stringify to a CONSTANT dedupe key."""
    connector = _connector(webhook_delivery_id_path="delivery.id")
    body = json.dumps({"account": {"id": "A1"}, "delivery": {"id": True}}).encode()

    dispatch = await connector.handle_webhook(body, {})

    assert dispatch.provider_account_id == "A1"
    assert dispatch.delivery_id is None


# ---- webhook_delivery_key ------------------------------------------------


def test_webhook_delivery_key_reads_the_id_from_the_raw_body():
    connector = _connector(webhook_delivery_id_path="delivery.id")
    body = json.dumps({"account": {"id": "A1"}, "delivery": {"id": 77}}).encode()

    assert connector.webhook_delivery_key(body, {"x-delivery": "H1"}) == "77"


def test_webhook_delivery_key_is_none_without_a_path():
    body = json.dumps({"delivery": {"id": "D1"}}).encode()

    assert _connector().webhook_delivery_key(body, {}) is None


@pytest.mark.parametrize("body", [b"not json", b"", b'{"account": {"id": "A1"}}'])
def test_webhook_delivery_key_is_none_when_the_body_does_not_carry_one(body):
    connector = _connector(webhook_delivery_id_path="delivery.id")

    assert connector.webhook_delivery_key(body, {}) is None


@pytest.mark.asyncio
async def test_webhook_delivery_key_agrees_with_handle_webhook():
    connector = _connector(webhook_delivery_id_path="delivery.id")
    body = json.dumps({"account": {"id": "A1"}, "delivery": {"id": "D1"}}).encode()

    dispatch = await connector.handle_webhook(body, {})

    assert connector.webhook_delivery_key(body, {}) == dispatch.delivery_id == "D1"


# ---- webhook_verifier -------------------------------------------------


def test_webhook_verifier_is_absent_for_a_polling_provider():
    """``None`` makes the events route 404 — the same answer an unknown provider gets."""
    assert _connector(**_polling_overrides()).webhook_verifier() is None


def test_webhook_verifier_is_present_for_a_webhook_provider():
    assert _connector().webhook_verifier() is not None


@pytest.mark.asyncio
async def test_webhook_verifier_verifies_a_delivery_signed_with_the_settings_secret():
    connector = _connector()
    body = b'{"account": {"id": "A1"}}'

    verified = await connector.webhook_verifier().verify(
        body, webhook_headers(connector.descriptor, _WEBHOOK_SECRET, body)
    )

    assert verified == body


@pytest.mark.asyncio
async def test_webhook_verifier_rejects_a_delivery_signed_with_another_secret():
    connector = _connector()
    body = b'{"account": {"id": "A1"}}'

    with pytest.raises(WebhookVerificationError):
        await connector.webhook_verifier().verify(
            body, webhook_headers(connector.descriptor, "other", body)
        )


def test_webhook_verifier_resolves_the_secret_per_request():
    """A rotated secret must take effect without re-registering the connector."""
    settings = FakeSettings(webhook_secret="first")
    verifier = _connector(settings=settings).webhook_verifier()

    assert verifier._secret_provider() == "first"
    settings.webhook_secret = "rotated"
    assert verifier._secret_provider() == "rotated"


def test_webhook_verifier_falls_back_to_a_signing_secret():
    """Slack's settings call the webhook key ``signing_secret``."""
    verifier = _connector(settings=FakeSettings(signing_secret="slack")).webhook_verifier()

    assert verifier._secret_provider() == "slack"


@pytest.mark.asyncio
async def test_webhook_verifier_without_a_secret_rejects_every_delivery():
    connector = _connector(settings=FakeSettings())
    body = b"{}"

    with pytest.raises(WebhookVerificationError):
        await connector.webhook_verifier().verify(
            body, webhook_headers(connector.descriptor, "", body)
        )


@pytest.mark.asyncio
async def test_webhook_verifier_without_settings_rejects_every_delivery():
    connector = DescriptorConnector(_descriptor(), None, FakeHttpClient())
    body = b"{}"

    with pytest.raises(WebhookVerificationError):
        await connector.webhook_verifier().verify(
            body, webhook_headers(connector.descriptor, "", body)
        )


@pytest.mark.asyncio
async def test_slack_shaped_descriptor_verifies_end_to_end():
    connector = _connector(
        signature_scheme=SignatureScheme.SLACK_V0,
        signature_header="X-Slack-Signature",
        signature_timestamp_header="X-Slack-Request-Timestamp",
        settings=FakeSettings(signing_secret="slack-secret"),
    )
    body = b'{"account": {"id": "T1"}}'

    verified = await connector.webhook_verifier().verify(
        body, webhook_headers(connector.descriptor, "slack-secret", body)
    )

    assert verified == body


# ---- sync_scope -------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_scope_names_the_dataset_from_the_provider_account_id():
    scope = await _connector().sync_scope(_credential(), _TOKENS, resources=["repo-1"])

    assert scope.dataset_name == "testprov_acct_1"
    assert scope.credential == "tok"
    assert scope.items == ["repo-1"]


@pytest.mark.asyncio
async def test_sync_scope_dataset_name_is_unaffected_by_the_account_label():
    """A renamed org or workspace must not move the dataset."""
    with_label = await _connector().sync_scope(
        _credential(account_label="Acme Corp"), _TOKENS, resources=["repo-1"]
    )
    renamed = await _connector().sync_scope(
        _credential(account_label="Acme Corp International"), _TOKENS, resources=["repo-1"]
    )
    without_label = await _connector().sync_scope(
        _credential(account_label=None), _TOKENS, resources=["repo-1"]
    )

    assert with_label.dataset_name == renamed.dataset_name == without_label.dataset_name
    assert with_label.dataset_name == "testprov_acct_1"


@pytest.mark.asyncio
async def test_sync_scope_refuses_a_full_sync_it_cannot_enumerate():
    """A descriptor must not turn unknown scope into an accepted empty sync."""
    with pytest.raises(NotImplementedError, match="cannot enumerate"):
        await _connector().sync_scope(_credential(), _TOKENS)


@pytest.mark.asyncio
async def test_sync_scope_rejects_an_empty_non_none_resources_list():
    """An empty ``resources`` must never be read as "sync everything"."""
    with pytest.raises(ValueError, match="empty \\(non-None\\) resources list"):
        await _connector().sync_scope(_credential(), _TOKENS, resources=[])


@pytest.mark.asyncio
async def test_sync_scope_has_no_credential_when_nothing_is_stored():
    """Normal for an auth model that mints per operation."""
    scope = await _connector().sync_scope(_credential(), {}, resources=["repo-1"])

    assert scope.credential is None


@pytest.mark.asyncio
async def test_sync_scope_prefers_the_access_token_over_the_bot_token():
    tokens = {"access_token": "user", "bot_access_token": "bot"}

    scope = await _connector().sync_scope(_credential(), tokens, resources=["r"])

    assert scope.credential == "user"


@pytest.mark.asyncio
async def test_sync_scope_uses_the_bot_token_when_that_is_all_there_is():
    scope = await _connector().sync_scope(
        _credential(), {"bot_access_token": "bot"}, resources=["r"]
    )

    assert scope.credential == "bot"


@pytest.mark.asyncio
async def test_sync_scope_carries_the_declared_content_type():
    scope = await _connector(content_type="code").sync_scope(
        _credential(), _TOKENS, resources=["repo-1"]
    )

    assert scope.content_type == "code"


@pytest.mark.parametrize(
    "expired",
    [
        datetime.now(timezone.utc) - timedelta(hours=1),
        # Naive, as SQLite hands it back.
        (datetime.now(timezone.utc) - timedelta(hours=1)).replace(tzinfo=None),
    ],
    ids=["aware", "naive"],
)
@pytest.mark.asyncio
async def test_sync_scope_logs_and_still_returns_an_expired_token(expired, caplog):
    """Handing the token out is unavoidable; logging is what makes the eventual
    401 attributable to "token expired, nobody refreshed it".
    """
    with caplog.at_level("WARNING"):
        scope = await _connector().sync_scope(
            _credential(token_expires_at=expired), _TOKENS, resources=["repo-1"]
        )

    assert scope.credential == "tok"
    assert "expired" in caplog.text
    assert "testprov" in caplog.text


@pytest.mark.asyncio
async def test_sync_scope_does_not_warn_for_an_unexpired_token(caplog):
    not_yet_expired = datetime.now(timezone.utc) + timedelta(hours=1)

    with caplog.at_level("WARNING"):
        await _connector().sync_scope(
            _credential(token_expires_at=not_yet_expired), _TOKENS, resources=["repo-1"]
        )

    assert "expired" not in caplog.text


# ---- dataset_name / polling / content_type -----------------------------


def test_dataset_name_matches_the_sync_scope_dataset():
    assert _connector().dataset_name(_credential()) == "testprov_acct_1"


def test_poll_targets_is_none_for_a_webhook_provider():
    """``None`` tells the scheduler to skip this connector entirely."""
    assert _connector().poll_targets(_credential()) is None
    assert _connector().poll_interval_seconds is None


def test_poll_targets_is_empty_for_a_polling_provider():
    connector = _connector(**_polling_overrides())

    assert connector.poll_targets(_credential()) == []
    assert connector.poll_interval_seconds == 900


def test_content_type_comes_from_the_descriptor():
    assert _connector().content_type is None
    assert _connector(content_type="code").content_type == "code"


# ---- api_key / local providers -----------------------------------------


def _api_key_connector(**overrides) -> DescriptorConnector:
    return DescriptorConnector(
        ConnectorDescriptor(provider="apollo", display_name="Apollo", auth="api_key", **overrides),
        None,
        FakeHttpClient(),
    )


def test_an_api_key_connector_describes_itself():
    connector = _api_key_connector()

    assert connector.auth == "api_key"
    assert connector.webhook_verifier() is None
    assert connector.poll_targets(_credential()) is None


def test_an_api_key_connector_has_no_oauth_members():
    connector = _api_key_connector()

    with pytest.raises(NotImplementedError, match="auth='api_key'"):
        connector.authorize_url("s")
    with pytest.raises(NotImplementedError, match="parse_installation"):
        connector.parse_installation({"access_token": "tok"})


@pytest.mark.asyncio
async def test_an_api_key_connector_cannot_exchange_a_code():
    with pytest.raises(NotImplementedError, match="exchange_callback"):
        await _api_key_connector().exchange_callback("code", {})


@pytest.mark.asyncio
async def test_an_api_key_connector_has_nothing_to_refresh():
    assert await _api_key_connector().refresh(_credential(), {"refresh_token": "r"}) is None


@pytest.mark.asyncio
async def test_an_api_key_connector_still_scopes_a_sync():
    scope = await _api_key_connector().sync_scope(
        _credential(provider="apollo", provider_account_id="team-9"),
        {"api_key": "k"},
        resources=["contacts"],
    )

    assert scope.dataset_name == "apollo_team_9"
    assert scope.items == ["contacts"]
    # An API key is not a bearer the contract knows; the source reads it itself.
    assert scope.credential is None


def test_a_local_connector_names_its_dataset():
    connector = DescriptorConnector(
        ConnectorDescriptor(provider="obsidian", display_name="Obsidian", auth="local")
    )

    assert connector.auth == "local"
    assert connector.dataset_name(_credential(provider_account_id="My Vault")) == (
        "obsidian_my_vault"
    )
