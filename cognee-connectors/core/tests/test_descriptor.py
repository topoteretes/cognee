"""Unit tests for cognee_connectors_core.descriptor.

Ported from cloud-backend tests/modules/integrations/test_framework_descriptor.py
(cognee-saas-backend, 2026-10-26). The cloud's framework-owned metadata keys
(``selected_resource_ids``, ``resource_health``) are host concerns and their
guards did not move; ``auth`` is new.

Two things are being protected. First, that a descriptor which cannot work is
rejected at import rather than at runtime — every one of those validations
exists because its runtime symptom shows up far from its cause (a connector
that 404s its own webhook, a connection that never syncs). Second, that URL
building and success-detection handle the shapes real providers actually send,
including the ones that report failure with HTTP 200.
"""

from urllib.parse import parse_qs, urlparse

import pytest

from cognee_connectors_core import descriptor as descriptor_module
from cognee_connectors_core.descriptor import ConnectorDescriptor, resolve_path
from cognee_connectors_core.signatures import SignatureScheme


def _descriptor(**overrides) -> ConnectorDescriptor:
    """A minimal valid webhook-capable descriptor, overridable per test."""
    base = {
        "provider": "testprov",
        "display_name": "Test Provider",
        "authorize_url": "https://testprov.example/oauth/authorize",
        "token_url": "https://testprov.example/oauth/token",
        "account_id_path": "account.id",
        "signature_scheme": SignatureScheme.SHA256_PREFIX,
        "signature_header": "x-signature",
        "webhook_account_id_path": "account.id",
    }
    return ConnectorDescriptor(**{**base, **overrides})


# ---- resolve_path ------------------------------------------------------


def test_resolve_path_reads_nested_value():
    assert resolve_path({"team": {"id": "T1"}}, "team.id") == "T1"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"team": {}},
        {"team": {"id": None}},
        # A non-mapping mid-path: provider bodies change shape without notice,
        # and one surprise should drop one delivery rather than raise out of a
        # verified webhook handler.
        {"team": "not-a-dict"},
        {"team": [{"id": "T1"}]},
    ],
    ids=["empty", "missing_leaf", "null_leaf", "scalar_midpath", "list_midpath"],
)
def test_resolve_path_returns_none_for_unreachable(payload):
    assert resolve_path(payload, "team.id") is None


def test_resolve_path_preserves_non_string_types():
    """Ids arrive as JSON numbers from several providers; coercion is the caller's."""
    assert resolve_path({"installation": {"id": 42}}, "installation.id") == 42


# ---- build_authorize_url ----------------------------------------------


def test_build_authorize_url_passes_every_parameter_through():
    """Every parameter reaches the provider with its value intact.

    ``:``, ``,`` and ``://`` are all legal unencoded in a query value, so these
    assertions hold whether urlencode ran or not. The encoding itself is
    covered by the test below, on inputs that actually diverge.
    """
    descriptor = _descriptor(scopes="read:content,write:comment")
    url = descriptor.build_authorize_url(
        client_id="client-123",
        redirect_uri="https://api.example.com/api/v1/integrations/testprov/callback",
        state="signed:state:value",
    )

    query = parse_qs(urlparse(url).query)
    assert query["client_id"] == ["client-123"]
    assert query["scope"] == ["read:content,write:comment"]
    assert query["redirect_uri"] == [
        "https://api.example.com/api/v1/integrations/testprov/callback"
    ]
    assert query["state"] == ["signed:state:value"]


def test_build_authorize_url_escapes_values_that_would_corrupt_the_query():
    """The characters that actually break raw interpolation, on the raw string.

    A space and an ``&`` are the two that diverge. Space-separated scopes are
    what Atlassian, Microsoft and HubSpot use, and a literal space is illegal
    in a URL.

    The ``&`` is the security half, not a formatting nit: ``authorize_params``
    is guarded against declaring a reserved KEY, but under raw interpolation a
    reserved parameter can be smuggled through a VALUE — ``prompt`` set to
    ``consent&scope=admin:org`` parses as a second ``scope``, i.e. the guard
    the reserved-key check exists to provide is bypassed. Asserted on the raw
    URL because ``parse_qs`` round-trips a literal space either way.
    """
    descriptor = _descriptor(
        scopes="read:jira-work manage:jira-project offline_access",
        authorize_params=(("prompt", "consent&scope=admin:org"),),
    )
    url = descriptor.build_authorize_url(
        client_id="client-123",
        redirect_uri="https://api.example.com/cb",
        state="signed:state:value",
    )

    assert " " not in url
    assert "scope=read%3Ajira-work+manage%3Ajira-project+offline_access" in url
    assert "prompt=consent%26scope%3Dadmin%3Aorg" in url

    query = parse_qs(urlparse(url).query)
    # One scope, the descriptor's own — the smuggled one did not become a
    # parameter.
    assert query["scope"] == ["read:jira-work manage:jira-project offline_access"]
    assert query["prompt"] == ["consent&scope=admin:org"]


def test_build_authorize_url_omits_scope_when_unset():
    """An empty ``scope=`` is not the same as no scope to every provider."""
    url = _descriptor(scopes="").build_authorize_url(
        client_id="c", redirect_uri="https://r", state="s"
    )
    assert "scope=" not in url


def test_build_authorize_url_includes_fixed_extra_params():
    descriptor = _descriptor(authorize_params=(("actor", "app"), ("response_type", "code")))
    query = parse_qs(
        urlparse(
            descriptor.build_authorize_url(client_id="c", redirect_uri="https://r", state="s")
        ).query
    )

    assert query["actor"] == ["app"]
    assert query["response_type"] == ["code"]


# ---- check_exchange ---------------------------------------------------


def test_check_exchange_accepts_a_clean_response():
    _descriptor().check_exchange({"access_token": "tok"})


def test_check_exchange_rejects_http_200_with_error_field():
    """The GitHub shape: HTTP 200, failure reported in the body."""
    with pytest.raises(RuntimeError, match="bad_verification_code"):
        _descriptor().check_exchange({"error": "bad_verification_code"})


def test_check_exchange_rejects_falsey_ok_field():
    """The Slack shape: HTTP 200 with ``ok: false``."""
    descriptor = _descriptor(ok_path="ok")
    with pytest.raises(RuntimeError, match="invalid_code"):
        descriptor.check_exchange({"ok": False, "error": "invalid_code"})


def test_check_exchange_names_the_provider_and_reason():
    """The message has to be actionable on its own — it is what reaches the logs."""
    with pytest.raises(RuntimeError) as excinfo:
        _descriptor().check_exchange({"error": "redirect_uri_mismatch"})

    message = str(excinfo.value)
    assert "testprov" in message
    assert "redirect_uri_mismatch" in message


def test_check_exchange_reports_unknown_when_no_reason_given():
    descriptor = _descriptor(ok_path="ok", error_path=None)
    with pytest.raises(RuntimeError, match="unknown"):
        descriptor.check_exchange({"ok": False})


# ---- Validation at construction ---------------------------------------


def test_missing_account_id_path_is_refused():
    """Without it a credential stores fine and then never receives a webhook.

    "Connected, silently never syncs" is the worst available failure shape, so
    it is refused at import instead.
    """
    with pytest.raises(ValueError, match="account_id_path"):
        _descriptor(account_id_path="")


def test_webhook_scheme_without_header_is_refused():
    with pytest.raises(ValueError, match="signature_header"):
        _descriptor(signature_header=None)


def test_webhook_scheme_without_routing_path_is_refused():
    """A verifiable delivery that cannot be routed is still a dropped delivery."""
    with pytest.raises(ValueError, match="webhook_account_id_path"):
        _descriptor(webhook_account_id_path=None)


def test_no_webhook_and_no_poll_interval_is_refused():
    """Such a connector would sync once at connect and then never again."""
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        _descriptor(
            signature_scheme=None,
            signature_header=None,
            webhook_account_id_path=None,
        )


def test_polling_connector_without_webhook_is_valid():
    descriptor = _descriptor(
        signature_scheme=None,
        signature_header=None,
        webhook_account_id_path=None,
        poll_interval_seconds=900,
    )
    assert descriptor.poll_interval_seconds == 900


def test_revoke_events_without_event_header_is_refused():
    with pytest.raises(ValueError, match="webhook_event_header"):
        _descriptor(webhook_revoke_events=("uninstalled",))


def test_sync_events_without_event_header_is_refused():
    """The same gate as revoke events, and the more dangerous one to miss.

    Both lists are matched against ``headers[webhook_event_header]``, so
    without that header every event name resolves to ``""`` and matches
    nothing — leaving resources empty on every delivery. A connection that
    looks healthy and never syncs, with no error and no log.
    """
    with pytest.raises(ValueError, match="webhook_sync_events"):
        _descriptor(webhook_sync_events=("push",))


def test_slack_v0_without_timestamp_header_is_refused_at_descriptor_level():
    """HmacVerifier refuses this too, but only when it is constructed.

    For a descriptor-driven connector that is the first call to
    ``webhook_verifier()`` — inside a live request, or inside the catalog.
    Checking at construction is what makes the import-time promise true.
    """
    with pytest.raises(ValueError, match="signature_timestamp_header"):
        _descriptor(
            signature_scheme=SignatureScheme.SLACK_V0,
            signature_header="x-slack-signature",
        )


def test_slack_v0_with_timestamp_header_is_valid():
    descriptor = _descriptor(
        signature_scheme=SignatureScheme.SLACK_V0,
        signature_header="x-slack-signature",
        signature_timestamp_header="x-slack-request-timestamp",
    )
    assert descriptor.signature_timestamp_header == "x-slack-request-timestamp"


def test_descriptor_is_immutable():
    """Descriptors are registry singletons read on every request."""
    with pytest.raises(AttributeError):
        _descriptor().provider = "hijacked"


@pytest.mark.parametrize("key", ["client_id", "redirect_uri", "state", "scope"], ids=lambda k: k)
def test_authorize_params_colliding_with_a_host_param_is_refused(key):
    """build_authorize_url sets these itself; a descriptor also declaring one
    doesn't error at the provider, it silently wins (most providers honor the
    last occurrence of a repeated query parameter) — replacing state, the
    signed CSRF binding, with something unsigned in the worst case.
    """
    with pytest.raises(ValueError, match=key):
        _descriptor(authorize_params=((key, "whatever"),))


def test_authorize_params_with_no_collision_is_valid():
    descriptor = _descriptor(authorize_params=(("actor", "app"),))
    assert descriptor.authorize_params == (("actor", "app"),)


def test_empty_authorize_url_is_refused():
    with pytest.raises(ValueError, match="authorize_url"):
        _descriptor(authorize_url="")


def test_empty_token_url_is_refused():
    with pytest.raises(ValueError, match="token_url"):
        _descriptor(token_url="")


def test_empty_provider_is_refused():
    with pytest.raises(ValueError, match="requires a provider"):
        _descriptor(provider="")


def test_signature_scheme_as_a_raw_string_is_refused():
    """SignatureScheme is a str enum, so a plain string like "sha256_prefix"
    imports and equality-compares as if it were fine — but HmacVerifier's
    dispatch is `is`-based, which a raw string never matches. Every delivery
    would silently fall through to "unsupported scheme" and 401, discovered
    only in production.
    """
    with pytest.raises(ValueError, match="signature_scheme"):
        _descriptor(signature_scheme="sha256_prefix")


def test_signature_scheme_as_the_enum_member_is_valid():
    descriptor = _descriptor(signature_scheme=SignatureScheme.SHA256_PREFIX)
    assert descriptor.signature_scheme is SignatureScheme.SHA256_PREFIX


@pytest.mark.parametrize("field", ["authorize_url", "token_url"])
def test_a_url_with_an_existing_query_string_is_refused(field):
    """build_authorize_url does an unconditional f"{authorize_url}?{urlencode(...)}"
    — a URL that already carries a query string produces a malformed
    double-query URL, and whichever parameter the provider parses first wins,
    silently discarding the rest (client_id has been observed to vanish this
    way).
    """
    with pytest.raises(ValueError, match=field):
        _descriptor(**{field: "https://testprov.example/oauth?already=here"})


def test_authorize_url_without_a_query_string_is_valid():
    descriptor = _descriptor(authorize_url="https://testprov.example/oauth/authorize")
    assert descriptor.authorize_url == "https://testprov.example/oauth/authorize"


def test_webhook_delivery_id_path_is_optional():
    """Not every provider sends a delivery id — dedupe stays opt-in."""
    descriptor = _descriptor()
    assert descriptor.webhook_delivery_id_path is None


def test_webhook_delivery_id_path_can_be_set():
    descriptor = _descriptor(webhook_delivery_id_path="delivery.id")
    assert descriptor.webhook_delivery_id_path == "delivery.id"


@pytest.mark.parametrize("source_path", ["access_token", "bot_access_token", "refresh_token"])
def test_metadata_paths_reading_token_material_is_refused(source_path):
    """``metadata_paths`` copies straight into the cleartext ``provider_metadata``
    column — a path that reads a live bearer or refresh token writes a secret
    into a column that is never encrypted.
    """
    with pytest.raises(ValueError, match="credential material"):
        # Matched on the guard's own wording, not on ``source_path``: the
        # message interpolates the path, so matching the path would be
        # satisfied by it echoing back out of any of the other ValueErrors
        # __post_init__ raises.
        _descriptor(metadata_paths=(("tok", source_path),))


def test_metadata_paths_with_no_collision_is_valid():
    descriptor = _descriptor(metadata_paths=(("account_type", "account.type"),))
    assert descriptor.metadata_paths == (("account_type", "account.type"),)


@pytest.mark.parametrize(
    "source_path",
    ["authed_user.access_token", "data.bot_access_token", "x.y.refresh_token"],
)
def test_metadata_paths_reading_nested_token_material_is_refused(source_path):
    """The whole-string match this guard used to do let every nested path through.

    ``resolve_path`` traverses dotted keys, so checking ``path in
    _TOKEN_MATERIAL_FIELDS`` only ever caught a top-level field name.
    ``authed_user.access_token`` is not a hypothetical shape: it is where
    Slack puts the user token in its own ``oauth.v2.access`` response.
    """
    with pytest.raises(ValueError, match="credential material"):
        _descriptor(metadata_paths=(("tok", source_path),))


def test_the_credential_denylist_is_a_superset_of_what_oauth2_stores():
    """A superset, deliberately not an equality.

    What has to hold is one-directional — every field oauth2 hands the host
    to encrypt must also be refused as a descriptor path — while the denylist
    stays free to refuse more.
    """
    from cognee_connectors_core import oauth2

    stored = set(oauth2._ACCESS_TOKEN_FIELDS) | {oauth2._REFRESH_TOKEN_FIELD}
    assert stored <= descriptor_module._TOKEN_MATERIAL_SEGMENTS


@pytest.mark.parametrize(
    "path_field",
    ["account_id_path", "account_label_path", "scopes_path", "webhook_account_id_path"],
)
def test_every_descriptor_path_field_refuses_credential_material(path_field):
    """The guard used to live inside the metadata_paths loop only.

    ``account_label_path`` and ``scopes_path`` write straight into their own
    cleartext columns on the host's credential row, so a descriptor pointing
    either at Slack's ``authed_user.access_token`` stored a live user token in
    the clear — the same finding as the metadata_paths one, at a field nobody
    had enumerated.
    """
    with pytest.raises(ValueError, match="credential material"):
        _descriptor(**{path_field: "authed_user.access_token"})


@pytest.mark.parametrize(
    "path",
    [
        "id_token",
        "token",
        "access_token_secret",
        "incoming_webhook.url",
        "authed_user.token",
        "app.client_secret",
        "data.session_password",
    ],
)
def test_credential_material_beyond_the_three_stored_field_names_is_refused(path):
    """Real provider responses carry credentials under more than three names.

    ``incoming_webhook.url`` is in Slack's own ``oauth.v2.access`` response
    and is a post-anything-to-this-channel credential; ``id_token`` is OIDC's
    bearer identity assertion; ``token``/``access_token_secret`` are the
    OAuth1-era spellings Twitter and Jira still emit.
    """
    with pytest.raises(ValueError, match="credential material"):
        _descriptor(metadata_paths=(("x", path),))


@pytest.mark.parametrize("path", ["account.page_token", "account.nextSyncToken", "sync_token"])
def test_a_pagination_cursor_is_not_treated_as_credential_material(path):
    """The over-broad half of the same rule.

    ``sync_token``/``page_token`` contain "token" and are what Google
    Calendar, Gmail and every cursor-paginated API hand you — a delta cursor
    belongs in ``provider_metadata``, and refusing it would tell the first
    author who needs one that their pagination cursor is a credential.
    """
    descriptor = _descriptor(metadata_paths=(("cursor", path),))
    assert descriptor.metadata_paths == (("cursor", path),)


def test_every_path_resolved_in_the_package_source_is_a_marked_field():
    """The completeness check the credential guard's coverage rests on.

    Deliberately NOT "every field whose name ends in _path is enumerated":
    that version filtered the declared set by the same predicate the
    derivation selected by, so both sides moved together and a plural
    path-shaped field was invisible to both. This reads the package's own
    source for descriptor attributes handed to ``resolve_path`` and asserts
    each one carries the ``payload_path`` marker, which is an axis the
    derivation does not control.
    """
    import dataclasses
    import re as _re
    from pathlib import Path as _Path

    marked = {
        f.name for f in dataclasses.fields(ConnectorDescriptor) if f.metadata.get("payload_path")
    }

    package = _Path(descriptor_module.__file__).parent
    resolved: set[str] = set()
    for source_file in package.glob("*.py"):
        source = source_file.read_text()
        for call in _re.finditer(r"resolve_path\(\s*[^,)]+,\s*([^)]+?)\)", source, _re.DOTALL):
            argument = call.group(1)
            for attribute in _re.findall(r"(?:descriptor|self)\.(\w+)", argument):
                resolved.add(attribute)

    assert resolved, "found no resolve_path call sites — the regex has rotted"
    assert resolved <= marked, f"resolved as a path but not marked: {sorted(resolved - marked)}"


def test_dropping_the_marker_from_a_path_field_is_caught():
    """Proves the marker is what the guard reads, not the field's name.

    A field renamed away from the ``*_path`` convention keeps its guard; a
    field that loses the marker loses it, which is the trade this design
    makes deliberately — intent is declared, not inferred.
    """
    descriptor = _descriptor(scopes_path="account.scopes")
    assert ("scopes_path", "account.scopes") in descriptor._path_fields()


@pytest.mark.parametrize("path", ["access_token.value", "incoming_webhook.url"])
def test_credential_material_in_a_non_final_segment_is_refused(path):
    """Pins the split itself, not the suffix rule standing in for it.

    Most nested cases (``authed_user.access_token``) would still be caught by
    the ``_token`` suffix even if the guard compared the whole path string, so
    they do not prove the decomposition. These two do: neither whole path ends
    in a blocked suffix nor equals a blocked name, so only splitting on "."
    finds the credential segment.
    """
    with pytest.raises(ValueError, match="credential material"):
        _descriptor(metadata_paths=(("x", path),))


def test_payload_paths_enumerates_a_marked_field_not_named_path():
    """What makes the marker mechanism different from the convention it replaced.

    Every field on ``ConnectorDescriptor`` happens to end in ``_path``, so on
    that class alone a metadata read and an ``endswith("_path")`` check are
    indistinguishable — a mutation swapping one for the other left the whole
    suite green. This exercises the enumeration against a field named
    ``cursor``, which only the marker can find.
    """
    from dataclasses import dataclass, field

    from cognee_connectors_core.descriptor import (
        NAMED_PAYLOAD_PATHS,
        PAYLOAD_PATH,
        payload_paths,
    )

    @dataclass
    class Odd:
        cursor: str = field(default="account.cursor", metadata=PAYLOAD_PATH)
        extras: tuple[tuple[str, str], ...] = field(
            default=(("a", "x.y"),), metadata=NAMED_PAYLOAD_PATHS
        )
        unmarked: str = "account.ignored"

    assert payload_paths(Odd()) == [("cursor", "account.cursor"), ("extras['a']", "x.y")]


# ---- auth: API-key and local providers --------------------------------


def test_an_api_key_descriptor_needs_only_its_identity():
    """Apollo, Asana, OneNote: a key the user pastes, no authorize URL, no token
    endpoint, no account id in any token response. The descriptor still names
    the provider for hosts and the catalog.
    """
    descriptor = ConnectorDescriptor(provider="apollo", display_name="Apollo", auth="api_key")

    assert descriptor.auth == "api_key"
    assert descriptor.authorize_url == ""
    assert descriptor.account_id_path == ""
    assert descriptor.poll_interval_seconds is None


def test_a_local_descriptor_needs_only_its_identity():
    descriptor = ConnectorDescriptor(
        provider="obsidian", display_name="Obsidian", auth="local", content_type=None
    )

    assert descriptor.auth == "local"


def test_auth_defaults_to_oauth2():
    assert _descriptor().auth == "oauth2"


def test_an_unknown_auth_is_refused():
    with pytest.raises(ValueError, match="auth must be one of"):
        ConnectorDescriptor(provider="x", display_name="X", auth="magic")


def test_an_api_key_descriptor_still_validates_its_webhook_wiring():
    """Relaxing the OAuth fields must not relax the webhook ones."""
    with pytest.raises(ValueError, match="signature_header"):
        ConnectorDescriptor(
            provider="apollo",
            display_name="Apollo",
            auth="api_key",
            signature_scheme=SignatureScheme.SHA256_PREFIX,
        )


def test_an_api_key_descriptor_still_refuses_a_query_string_in_a_url_it_does_declare():
    with pytest.raises(ValueError, match="token_url"):
        ConnectorDescriptor(
            provider="apollo",
            display_name="Apollo",
            auth="api_key",
            token_url="https://apollo.example/token?x=1",
        )


def test_an_api_key_descriptor_still_refuses_credential_material_in_a_path():
    with pytest.raises(ValueError, match="credential material"):
        ConnectorDescriptor(
            provider="apollo",
            display_name="Apollo",
            auth="api_key",
            metadata_paths=(("key", "api_key"),),
        )
