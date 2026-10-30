"""A connector expressed as data.

Moved from cloud-backend ``modules.integrations.framework.descriptor``
(cognee-saas-backend, 2026-10-26). The cloud's own metadata keys (``selection``,
``resource_health``) are host concerns and are no longer guarded here; ``auth``
is new, so API-key and local providers can describe themselves on the same
dataclass.

The bet this module makes: most of the roadmap's 50+ providers differ only in
strings — an authorize URL, a token endpoint, scopes, and where in the
response their account identity sits. If that holds, those providers need no
Python at all, and adding one is a descriptor plus a secret. If it does not
hold, the tier model in :mod:`~.base` is wrong and it is much cheaper to find
that out here than after 20 connectors are built on it.

So the descriptor covers standard OAuth2 exhaustively and refuses to grow
hooks. When a provider does not fit, it graduates to Tier 2 (a descriptor plus
an overridden :meth:`~.oauth2.DescriptorConnector.resolve_account`) or Tier 3
(its own :class:`~.base.Connector`) — rather than this dataclass acquiring a
field for every provider's quirk, which is how declarative config decays back
into code with worse ergonomics.

**Why JSON paths.** The only genuinely per-provider logic in a standard OAuth2
flow is "which field holds the account id" — ``team.id`` for Slack,
``organization.id`` for Linear. Expressed as a path string, that is config;
expressed as a lambda, it is code, and code cannot live in a YAML catalog or
be reviewed by someone who does not write Python. Paths are deliberately
minimal — dotted key traversal, no wildcards, no filters, no indexing. A
provider needing more than that needs a hook, and the tier model already says
where hooks go.
"""

from dataclasses import dataclass, field, fields
from typing import Any
from urllib.parse import urlencode

from cognee_connectors_core.base import AUTH_KINDS, AuthKind
from cognee_connectors_core.oauth_state import DEFAULT_STATE_TTL_SECONDS
from cognee_connectors_core.signatures import SignatureScheme

# The field names a standard OAuth2 token response carries bearer/refresh
# material under, i.e. the ones oauth2.py hands the host to encrypt. Defined
# HERE rather than in oauth2.py, which is where they are actually consumed,
# for one reason: oauth2.py already imports this module, so the dependency
# only runs one way and oauth2.py can import these — where the reverse would
# be a cycle. They used to be duplicated in both files with a comment asking
# the reader to keep them in sync, which is not a mechanism.
ACCESS_TOKEN_FIELDS = ("access_token", "bot_access_token")
REFRESH_TOKEN_FIELD = "refresh_token"

# Substrings that make a path segment credential-bearing. Matched by
# CONTAINMENT on a case-folded, alphanumeric-only form of the segment, which
# is what makes this a rule rather than a list: "access_token",
# "accessToken", "ACCESS-TOKEN" and "authed_user.access_token" all reduce to
# something containing "token".
#
# Three earlier versions of this guard were enumerations that each closed the
# one example a report had named — a set of three leaf names, then the same
# three matched per segment, then those plus hand-picked extras and the
# suffixes "_token"/"_secret"/"_password". Each still let the next real field
# through: "config.secret" is GitHub's hook signing secret, "auth.api_key"
# and "app.privateKey" are ordinary, and none of them ends in one of those
# suffixes. So the direction is inverted here — anything that looks like a
# credential is refused, and an author who needs an exception argues for it
# in _CREDENTIAL_EXEMPT below.
_CREDENTIAL_SUBSTRINGS = (
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "apikey",
    "privatekey",
    "publickey",
    "signingkey",
    "accesskey",
    "cookie",
    "signature",
    "bearer",
    "jwt",
    "pem",
)

# Segments that contain one of the above and are provably not credentials.
# Every entry is a field a real provider sends and a descriptor has a
# legitimate reason to store: pagination and delta-sync cursors, which belong
# in provider_metadata and which Google Calendar, Gmail, Notion and every
# cursor-paginated API hand you under exactly these names. Without this the
# rule refuses them and tells the author their pagination cursor is
# credential material, which is the over-broad half of the same problem.
_CREDENTIAL_EXEMPT = frozenset(
    {
        "pagetoken",
        "nextpagetoken",
        "synctoken",
        "nextsynctoken",
        "deltatoken",
        "continuationtoken",
    }
)

# Whole segments refused by name, for credentials whose field name carries no
# hint of what it is. The containment rule cannot catch these and no rule
# can: "incoming_webhook" is Slack's oauth.v2.access subtree whose "url"
# field is a post-anything-to-this-channel credential, and "url" is not a
# word that can go in a denylist. The whole subtree is refused rather than
# just its url — nothing under it is worth the risk of enumerating leaves.
_CREDENTIAL_SUBTREES = frozenset({"incomingwebhook"})

# Kept as a named set because oauth2.py's stored-token fields must be a
# SUBSET of what this guard refuses, and a test asserts that direction. It is
# deliberately one-directional: the denylist is free to refuse more than
# oauth2 stores, and an earlier test asserting the two were EQUAL made the
# denylist unable to grow at all.
_TOKEN_MATERIAL_SEGMENTS = frozenset({*ACCESS_TOKEN_FIELDS, REFRESH_TOKEN_FIELD})


def _normalize_segment(segment: str) -> str:
    """Case-fold a path segment down to its letters and digits.

    So camelCase, snake_case, kebab-case and any punctuation a provider
    invents all reduce to one comparable form. Without this the rule below is
    a snake_case-only rule, which is how "accessToken" got past three
    previous versions of it.
    """
    return "".join(char for char in segment.lower() if char.isalnum())


def _credential_segment(path: str) -> str | None:
    """The first segment of ``path`` that names credential material, if any.

    Split the same way :func:`resolve_path` splits, because a guard that
    decomposes a path differently from the code that traverses it is a guard
    with a gap — that is exactly how ``authed_user.access_token`` got past an
    earlier version of this check, which compared the whole path string
    against a set of leaf names.
    """
    for segment in path.split("."):
        normalized = _normalize_segment(segment)
        if not normalized:
            continue
        if normalized in _CREDENTIAL_SUBTREES:
            return segment.strip()
        if normalized in _CREDENTIAL_EXEMPT:
            continue
        if any(marker in normalized for marker in _CREDENTIAL_SUBSTRINGS):
            return segment.strip()
    return None


# Query parameters build_authorize_url always sets itself, in this order —
# see the ``params = [...]`` list there. A descriptor's authorize_params
# extends this list, so any of these names in it collides and (per how every
# provider treats a repeated query parameter) silently replaces the host's
# own value. state is the one that actually matters: it is the signed CSRF
# binding the whole install to a user, and a descriptor overwriting it
# — even by accident, declaring `("state", "x")` for some unrelated reason —
# would replace it with something unsigned.
_RESERVED_AUTHORIZE_PARAMS = frozenset({"client_id", "redirect_uri", "state", "scope"})


def resolve_path(payload: Any, path: str) -> Any | None:
    """Read a dotted path out of a decoded JSON body, or ``None``.

    ``resolve_path({"team": {"id": "T1"}}, "team.id") == "T1"``

    Absent and null are both ``None`` on purpose: for every use here — is
    there an account id? a label? — "the key was missing" and "the key was
    null" call for the same handling, and distinguishing them would only add a
    branch every caller collapses again.

    A non-mapping encountered mid-path is ``None`` rather than an error.
    Webhook and token bodies are provider-controlled and change without
    notice; a shape surprise should drop one delivery, not raise out of a
    verified webhook handler.
    """
    current = payload
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def payload_paths(instance: Any) -> list[tuple[str, str]]:
    """Enumerate the payload paths any dataclass instance declares.

    Reads the fields' own ``payload_path`` metadata, so a path field joins the
    credential guard by declaring what it is rather than by being named a
    particular way. Written as a free function over any dataclass, not a
    method, so it can be exercised against a field whose name does NOT end in
    ``_path`` — otherwise the marker mechanism is indistinguishable from the
    naming convention it replaced, and a test cannot tell which one the guard
    is really reading.
    """
    collected: list[tuple[str, str]] = []
    for declared in fields(instance):
        kind = declared.metadata.get("payload_path")
        if kind is None:
            continue
        value = getattr(instance, declared.name)
        if not value:
            continue
        if kind == "scalar":
            collected.append((declared.name, value))
        else:
            collected.extend((f"{declared.name}[{name!r}]", path) for name, path in value if path)
    return collected


# Field markers, so the credential guard enumerates paths by DECLARED INTENT
# rather than by a naming convention. The convention version keyed on
# `name.endswith("_path")` and then special-cased metadata_paths by name,
# which is the tell that the convention was never load-bearing: a second
# plural path field would have been invisible to it, and invisible to the
# test meant to catch exactly that (the test filtered the declared set by the
# same predicate the derivation selected by, so both sides moved together).
#: A single dotted path resolved against a provider payload.
PAYLOAD_PATH = {"payload_path": "scalar"}
#: A ``((name, path), ...)`` tuple of them.
NAMED_PAYLOAD_PATHS = {"payload_path": "named_tuple"}


@dataclass(frozen=True, slots=True)
class ConnectorDescriptor:
    """Everything needed to serve a standard OAuth2 connector.

    Frozen: descriptors are registry singletons read on every request, and a
    mutable one would be a cross-request side channel.

    For an API-key or local provider (``auth="api_key"`` / ``auth="local"``)
    only ``provider``, ``display_name`` and ``auth`` are needed: the OAuth
    fields may stay empty, and the descriptor still names the provider, its
    dataset naming, its content type and its polling for hosts and the
    catalog.
    """

    # ---- Identity ------------------------------------------------------

    #: URL segment and credential discriminator. Stable forever — it is
    #: written into every credential row.
    provider: str

    #: Name shown in the connector catalog and UI card.
    display_name: str

    #: How a connection is established: ``oauth2`` (the fields below), or
    #: ``api_key`` / ``local`` for a provider with no OAuth flow — their
    #: OAuth members on :class:`~.oauth2.DescriptorConnector` raise
    #: ``NotImplementedError`` and the OAuth fields are not validated.
    auth: AuthKind = "oauth2"

    # ---- OAuth2 --------------------------------------------------------

    #: Provider's authorize endpoint, without query string. The host
    #: appends ``client_id``, ``scope``, ``redirect_uri``, ``state``.
    #: Required for ``auth="oauth2"``.
    authorize_url: str = ""

    #: Endpoint that exchanges the code for tokens. Required for
    #: ``auth="oauth2"``.
    token_url: str = ""

    #: Space- or comma-separated scopes, exactly as the provider expects them.
    #: Most providers will not grant new scopes to an already-authorized
    #: account, so widening this generally forces every existing connection to
    #: reconnect — batch scope changes rather than shipping them singly.
    scopes: str = ""

    #: Extra fixed query parameters on the authorize URL, for providers that
    #: require one (``actor=app``, ``response_type=code``, ``prompt=consent``).
    authorize_params: tuple[tuple[str, str], ...] = ()

    # ---- Reading the token response ------------------------------------

    #: Path to the external account id — the value webhooks route by. Its
    #: stability across connect and webhook payloads is what makes routing
    #: work; see :attr:`~.base.ConnectorInstallation.provider_account_id`.
    #: Required for ``auth="oauth2"``.
    account_id_path: str = field(default="", metadata=PAYLOAD_PATH)

    #: Path to a human-readable account name, for the UI only.
    account_label_path: str | None = field(default=None, metadata=PAYLOAD_PATH)

    #: Path to the granted-scopes string, when the provider reports it.
    scopes_path: str | None = field(default=None, metadata=PAYLOAD_PATH)

    #: Paths copied into ``provider_metadata`` (cleartext), keyed by the name
    #: to store them under. Not verbatim: a path resolving to an object or an
    #: array is dropped with a warning rather than stringified, so only JSON
    #: scalars land in the column. Must never point at credential material —
    #: this dict is not encrypted, and ``__post_init__`` refuses paths that
    #: look like it.
    metadata_paths: tuple[tuple[str, str], ...] = field(default=(), metadata=NAMED_PAYLOAD_PATHS)

    #: Path that must be truthy for the exchange to count as successful.
    #: Slack returns HTTP 200 with ``ok: false``, so status alone cannot be
    #: trusted; ``None`` means do not check.
    ok_path: str | None = field(default=None, metadata=PAYLOAD_PATH)

    #: Path that must be *absent* for success. Several providers return HTTP
    #: 200 with an ``error`` field. ``None`` means do not check.
    error_path: str | None = field(default="error", metadata=PAYLOAD_PATH)

    # ---- Webhooks ------------------------------------------------------

    #: ``None`` for a provider with no webhooks — ``POST /{provider}/events``
    #: then 404s, and the connector is polled instead.
    signature_scheme: SignatureScheme | None = None

    #: Header carrying the signature (``x-hub-signature-256``).
    signature_header: str | None = None

    #: Header carrying the timestamp, for the schemes that have one.
    signature_timestamp_header: str | None = None

    #: Path to the account id in a webhook body. Often differs from
    #: :attr:`account_id_path` even though it must resolve to the same value —
    #: the token response and the event envelope rarely share a shape.
    webhook_account_id_path: str | None = field(default=None, metadata=PAYLOAD_PATH)

    #: Path to the changed resource in a webhook body.
    webhook_resource_path: str | None = field(default=None, metadata=PAYLOAD_PATH)

    #: Path to a per-delivery id in a webhook body, handed to the host on
    #: :attr:`~.base.WebhookDispatch.delivery_id` for replay dedupe.
    #: Deliberately a body path, not a header: for the
    #: ``sha256_prefix``/``bare_hex`` signature schemes the HMAC covers only
    #: ``raw_body``, so a header carries no guarantee it was not changed after
    #: signing — a delivery id sourced from a header would let a replay dodge
    #: dedupe just by editing that header, signature intact. ``None`` (the
    #: default) means this provider sends nothing reliable to dedupe on, and
    #: the connector must rely on its handler being idempotent instead — not
    #: every provider needs this field.
    webhook_delivery_id_path: str | None = field(default=None, metadata=PAYLOAD_PATH)

    #: Header naming the event type (``x-github-event``).
    webhook_event_header: str | None = None

    #: Event names meaning "this install is gone" — uninstall, suspend, token
    #: revoked. The host revokes the credential and skips syncing.
    webhook_revoke_events: tuple[str, ...] = ()

    #: Event names that imply a re-sync. Empty means every event does, which
    #: is the safe default: over-syncing costs compute, under-syncing silently
    #: serves stale data.
    webhook_sync_events: tuple[str, ...] = ()

    # ---- Ingestion -----------------------------------------------------

    #: cognee ingestion mode — ``"code"`` for repositories, ``None`` for
    #: ordinary documents.
    content_type: str | None = None

    #: Poll interval for a provider with no webhooks. Required for an
    #: ``oauth2`` connector when :attr:`signature_scheme` is ``None``,
    #: meaningless otherwise.
    poll_interval_seconds: int | None = None

    #: How long a minted OAuth state stays valid.
    state_ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS

    def _path_fields(self) -> list[tuple[str, str]]:
        """Every ``(field_name, path)`` this descriptor will resolve at runtime."""
        return payload_paths(self)

    def __post_init__(self) -> None:
        """Reject a descriptor that cannot work, at import rather than at runtime.

        These are all wiring mistakes whose runtime symptom is far from its
        cause — a connector that 404s its own webhook, or authenticates
        against an empty key. Catching them here means a bad descriptor fails
        the test suite instead of failing in production on the first delivery.
        """
        if not self.provider:
            raise ValueError("descriptor requires a provider")
        if self.auth not in AUTH_KINDS:
            raise ValueError(
                f"{self.provider}: auth must be one of {', '.join(AUTH_KINDS)}, got {self.auth!r}"
            )
        oauth2 = self.auth == "oauth2"
        if oauth2 and not self.authorize_url:
            raise ValueError(f"{self.provider}: authorize_url is required")
        if oauth2 and not self.token_url:
            raise ValueError(f"{self.provider}: token_url is required")
        for field_name in ("authorize_url", "token_url"):
            value = getattr(self, field_name)
            if value and ("?" in value or "#" in value):
                # build_authorize_url does `f"{authorize_url}?{urlencode(params)}"`
                # unconditionally — a URL that already carries a query string
                # (or a fragment, which a query string appended after it would
                # silently become part of) produces a malformed double-query
                # URL, and whichever parameter the provider parses first wins,
                # discarding the rest (client_id has been observed to vanish
                # this way). A provider needing fixed query params on either
                # URL should use authorize_params instead.
                raise ValueError(
                    f"{self.provider}: {field_name} must not contain a query string or "
                    "fragment — use authorize_params for fixed query parameters"
                )
        for key, _value in self.authorize_params:
            if key in _RESERVED_AUTHORIZE_PARAMS:
                # Most providers honor the last occurrence of a repeated query
                # parameter, so this would not error at the provider — it would
                # silently replace client_id/redirect_uri/scope, or worse,
                # state (the signed CSRF binding) with whatever the descriptor
                # declared, discovered only when installs mysteriously fail to
                # come back to the right user.
                raise ValueError(
                    f"{self.provider}: authorize_params cannot set {key!r} — it collides "
                    "with a parameter the host already sets on the authorize URL"
                )
        # Every path field, not just metadata_paths. This check has now been
        # fixed twice at whatever field the last report happened to name: once
        # for a whole-string match that missed nested paths, once for
        # metadata_paths while account_label_path and scopes_path wrote a live
        # token into their own cleartext String columns unchecked. So it is
        # written as an enumeration of the descriptor's path fields instead of
        # a check at one of them, and _path_fields() is asserted against the
        # dataclass's real field list in the tests — a path field added later
        # fails that test rather than silently skipping this guard.
        for field_name, path in self._path_fields():
            leaking = _credential_segment(path)
            if leaking:
                raise ValueError(
                    f"{self.provider}: {field_name} cannot read {path!r} — its "
                    f"{leaking!r} segment is credential material, and every value a "
                    "descriptor path resolves is stored or logged in the clear"
                )
        if oauth2 and not self.account_id_path:
            raise ValueError(
                f"{self.provider}: account_id_path is required — without it a webhook "
                "cannot be routed back to the connections that own the account"
            )
        if self.signature_scheme is not None and not isinstance(
            self.signature_scheme, SignatureScheme
        ):
            # SignatureScheme is a str enum, so a plain string like
            # "sha256_prefix" imports cleanly and looks right at a glance —
            # but HmacVerifier.verify's dispatch is `is`-based
            # (`self.scheme is SignatureScheme.SHA256_PREFIX`), which a raw
            # string never matches. Every delivery would fall through to the
            # "unsupported scheme" branch and 401, discovered only once a
            # provider's webhooks silently never verify in production.
            raise ValueError(
                f"{self.provider}: signature_scheme must be a SignatureScheme member, "
                f"got {self.signature_scheme!r}"
            )
        if self.signature_scheme and not self.signature_header:
            raise ValueError(f"{self.provider}: signature_scheme requires signature_header")
        if self.signature_scheme and not self.webhook_account_id_path:
            raise ValueError(
                f"{self.provider}: a webhook-capable connector needs "
                "webhook_account_id_path to route deliveries"
            )
        if oauth2 and self.signature_scheme is None and not self.poll_interval_seconds:
            raise ValueError(
                f"{self.provider}: a connector with no webhook signature must set "
                "poll_interval_seconds, or it would never sync after the initial connect"
            )
        if self.signature_scheme is SignatureScheme.SLACK_V0 and not (
            self.signature_timestamp_header
        ):
            # HmacVerifier refuses this too, but only when it is constructed —
            # which for a descriptor-driven connector is the first call to
            # webhook_verifier(), i.e. inside a live request or inside the
            # catalog. Checking here keeps the promise this method makes: a
            # broken descriptor fails at import, not in production.
            raise ValueError(
                f"{self.provider}: {SignatureScheme.SLACK_V0.value} requires "
                "signature_timestamp_header, or it would verify the HMAC and "
                "silently skip replay protection"
            )
        # Both event lists are read via headers[webhook_event_header], so
        # either one without that header resolves every event name to "" and
        # matches nothing. For revoke that means an uninstall is never
        # processed; for sync it means resources is empty on every delivery —
        # the "connected, silently never syncs" shape, with no error and no log.
        for field_name in ("webhook_revoke_events", "webhook_sync_events"):
            if getattr(self, field_name) and not self.webhook_event_header:
                raise ValueError(
                    f"{self.provider}: {field_name} needs webhook_event_header "
                    "to know which event arrived"
                )

    def build_authorize_url(self, *, client_id: str, redirect_uri: str, state: str) -> str:
        """The provider URL the browser is sent to.

        Every parameter is URL-encoded — scopes routinely contain ``:`` and
        ``,``, and redirect URIs contain ``://``, all of which corrupt the
        query string when interpolated raw.
        """
        params = [
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
            ("state", state),
        ]
        if self.scopes:
            params.append(("scope", self.scopes))
        params.extend(self.authorize_params)
        return f"{self.authorize_url}?{urlencode(params)}"

    def check_exchange(self, payload: dict[str, Any]) -> None:
        """Raise ``RuntimeError`` if the token response reports failure.

        Named in the exception so the error says which provider rejected what,
        rather than surfacing as a ``KeyError`` three frames later.
        """
        if self.ok_path is not None and not resolve_path(payload, self.ok_path):
            raise RuntimeError(
                f"{self.provider} token exchange failed: "
                f"{resolve_path(payload, self.error_path or 'error') or 'unknown'}"
            )
        if self.error_path is not None:
            error = resolve_path(payload, self.error_path)
            if error:
                raise RuntimeError(f"{self.provider} token exchange failed: {error}")


__all__ = [
    "ACCESS_TOKEN_FIELDS",
    "DEFAULT_STATE_TTL_SECONDS",
    "NAMED_PAYLOAD_PATHS",
    "PAYLOAD_PATH",
    "REFRESH_TOKEN_FIELD",
    "ConnectorDescriptor",
    "payload_paths",
    "resolve_path",
]
