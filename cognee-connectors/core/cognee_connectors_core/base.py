"""The connector contract every cognee connector implements.

Moved from cloud-backend ``modules.integrations.framework.base``
(cognee-saas-backend, 2026-10-26) and re-expressed without a host: no FastAPI
request, no credential model, no delivery dedupe. The host — cognee core's
``cognee.modules.integrations``, cloud-backend — owns credential storage, token
decryption, the OAuth routes and webhook dedupe; a connector supplies only what
is genuinely provider-specific, and this module is where that line is drawn.

Slack was built as a hand-written module: thousands of lines across ~21 files,
its own router block, its own sync router, its own secret keys. That is
affordable once. The connectors roadmap is 50+ providers, where it is not: it
would mean 50 copies of OAuth state signing, credential handling, and webhook
HMAC verification — logic that has nothing to do with any individual provider,
replicated 50 times, each copy an independent chance to get a security boundary
subtly wrong. So the contract owns that logic exactly once.

**Three tiers, by how much code a provider actually needs.** The tier is an
observation about a provider, not a setting — it falls out of which methods the
provider has to override:

* **Tier 1 — pure config.** Standard OAuth2: an authorize URL, a token
  endpoint, scopes, and an account identity readable straight out of the token
  response. No Python at all; see :class:`~.descriptor.ConnectorDescriptor`,
  which :class:`~.oauth2.DescriptorConnector` turns into a working connector.
* **Tier 2 — config plus one hook.** OAuth2, but the token response does not
  say *which* account authorized, so identity needs one extra API call. Linear
  is the type case (its token response omits the organization; webhooks route
  by it). A descriptor plus an overridden
  :meth:`~.oauth2.DescriptorConnector.resolve_account`.
* **Tier 3 — full connector.** An auth model that is not OAuth2-with-a-bearer-
  token at all. GitHub is the type case: an org-level RSA key signs a short
  JWT, which mints hour-lived per-installation tokens, and nothing durable is
  stored. Subclass :class:`Connector` directly.

API-key and local providers (Apollo, Asana, OneNote, Obsidian) describe
themselves on the same contract with :attr:`Connector.auth` set to
``"api_key"`` or ``"local"``: their OAuth members raise ``NotImplementedError``
and the rest — dataset naming, the catalog entry, polling — works as for any
other provider.

**Multi-tenancy is the host's job, not the connector's.** A connector never
sees a user id or a tenant id. It maps provider data to and from an *external
account identity* (:attr:`ConnectorInstallation.provider_account_id`) and the
host resolves that to users or tenants — one authorize, or many on a fan-out
path, where one external workspace legitimately feeds several tenants. A
connector that reasons about tenants itself is a connector that will
eventually leak across them.

**Token material is passed in, never looked up.** Every method that needs a
stored token takes ``tokens``, the decrypted token payload, as a plain mapping.
Nothing in this package decrypts or persists; the host does both, at the
moment of use. Provider app secrets (a GitHub App's private key) stay with the
connector's own settings and never go on a :class:`SyncScope`, which is shipped
to whatever runs the sync.
"""

import logging
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal, Protocol
from uuid import UUID

logger = logging.getLogger(__name__)

#: How a connection is established. See :attr:`Connector.auth`.
AuthKind = Literal["oauth2", "api_key", "local"]
AUTH_KINDS: tuple[str, ...] = ("oauth2", "api_key", "local")


@dataclass(slots=True)
class ConnectorInstallation:
    """What a completed connect yields, normalized for the credential store.

    This is the whole surface between a provider's OAuth response and the
    host's credential table. :meth:`Connector.parse_installation` produces it;
    the host encrypts :attr:`token_payload` and writes the rest, adding the
    user or tenant scoping the connector never sees.
    """

    # The external account this install belongs to — a Slack team id, a GitHub
    # App installation id, a Linear organization id. Load-bearing well beyond
    # display: it is the *only* handle an inbound webhook carries, so it is how
    # a delivery is routed back to the connections that own it. It must
    # therefore be stable for the life of the install and identical on both
    # paths (the value parsed at connect time and the one read off a webhook
    # payload). A provider whose webhooks carry a different id than its token
    # response is a provider whose deliveries will resolve to nothing.
    provider_account_id: str

    # Token material. Encrypted at rest by the host; never logged, never
    # returned by an API route. May be empty for an auth model with nothing
    # durable to keep — GitHub App installs store ``{}`` and mint a fresh token
    # per operation.
    token_payload: dict[str, Any] = field(default_factory=dict)

    # Non-secret provider facts worth keeping in the clear, so support and the
    # UI can answer "connected to what?" without decrypting anything.
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    # Human-readable account name for the UI ("acme-corp", "Acme Workspace").
    # Never used for identity or lookups: it is provider-controlled, mutable,
    # and not unique.
    account_label: str | None = None

    # Granted scopes as the provider reported them, for support and for
    # detecting an install that predates a scope addition.
    scopes: str | None = None

    # When the access token expires, for providers that rotate. ``None`` means
    # either no expiry or no stored token at all — both are "nothing to
    # refresh", which is why they share a representation.
    token_expires_at: datetime | None = None

    # How to interpret ``token_payload``: ``oauth2`` (bearer), ``github_app``
    # (mint on demand), ``api_key`` (user-supplied). Read by refresh logic to
    # decide whether rotation even applies.
    auth_type: str = "oauth2"


@dataclass(slots=True)
class RefreshedTokens:
    """What :meth:`Connector.refresh` hands back for the host to store.

    ``token_payload`` replaces the stored payload wholesale, so a connector
    that rotates the refresh token puts the new one here and one whose
    provider does not rotate puts the old one back. ``token_expires_at`` and
    ``scopes`` replace their stored values; ``None`` means the provider said
    nothing about them.
    """

    token_payload: dict[str, Any]
    token_expires_at: datetime | None = None
    scopes: str | None = None


@dataclass(slots=True)
class SyncScope:
    """What a sync is asked to ingest, and the short-lived credential to do it.

    Returned by :meth:`Connector.sync_scope` and handed to whatever runs the
    sync (cognee's ingestion, a tenant pod). Deliberately not the provider's
    own payload shape: the runner stays provider-agnostic, so adding a
    connector never touches it.
    """

    # cognee dataset the content lands in. Convention is
    # ``{provider}_{account_slug}`` — one dataset per external account rather
    # than per repo/channel, because backend access control isolates per
    # dataset and per-resource datasets would multiply that cost without
    # buying isolation anyone asked for.
    dataset_name: str

    # cognee ingestion mode — e.g. ``"code"`` for repositories. ``None`` for
    # ordinary document ingestion.
    content_type: str | None = None

    # What to ingest: repo clone URLs, channel ids, page ids. Opaque to the
    # host, meaningful to the source for this provider.
    items: list[str] = field(default_factory=list)

    # Short-lived credential the runner uses to fetch the items above — an
    # hour-lived GitHub installation token, a Slack bot token. Scoped to this
    # one external account, never a durable app secret, and never persisted by
    # the runner.
    credential: str | None = None

    # Expiry of ``credential`` when it is deliberately short-lived. The runner
    # uses this only to refresh before expiry; it never persists it. A
    # connector that can mint a replacement marks this scope refreshable, and
    # the host binds that capability to this dispatch rather than exposing a
    # durable provider secret.
    credential_expires_at: datetime | None = None
    credential_refreshable: bool = False

    # Provider-specific hints for the source (branch name, since cursor). Kept
    # a free-form dict so a provider can pass what it needs without widening
    # this dataclass — and therefore never secrets, which belong in
    # :attr:`credential` where the redaction rules apply.
    options: dict[str, Any] = field(default_factory=dict)

    # Which resource each entry of :attr:`items` came from, same length and
    # order. Optional: fill it when ``items`` are *derived* from resource ids
    # rather than being them, so a per-item outcome can be attributed back to
    # the id a picker shows. GitHub needs it — its items are clone URLs built
    # from ``acme/api`` — while a connector whose items already *are* resource
    # ids can leave it ``None`` and be attributed positionally.
    #
    # The host never parses these, and neither does the runner: they travel
    # down and come back as opaque labels, which is what keeps the runner
    # provider-agnostic while still reporting something a UI can key on.
    item_resource_ids: list[str] | None = None


@dataclass(slots=True)
class ConnectorResource:
    """One thing a connection could ingest, for the picker UI.

    A repository, a channel, a space. Display data plus the id a selection is
    recorded against — deliberately not the provider's whole object, so the
    picker stays provider-agnostic and no provider field leaks into it by
    accident.
    """

    # Stable provider id, and what a selection stores. Must be the same value
    # the connector's own sync path filters on, or a picked resource is not the
    # resource that gets ingested.
    id: str

    # What the user sees — a repository full name, a channel name.
    name: str

    # Optional one-line context for the picker (description, visibility).
    description: str | None = None

    # Provider-specific display hints (``private``, ``archived``, member
    # counts). Free-form so a connector can surface what matters for choosing,
    # without this dataclass growing a field per provider. Display only, never
    # read back for logic, and never secrets.
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WebhookDispatch:
    """A verified webhook, reduced to "who does this concern, and what changed?".

    :meth:`Connector.handle_webhook` returns this instead of acting directly,
    which is what keeps fan-out in the host: the connector names the external
    account, and the host resolves it to connections, enforces any fan-out
    cap, and dispatches. A connector that dispatched for itself would have to
    re-implement that resolution — and one of the 50 would get it wrong.
    """

    # Which external account this delivery is about. ``None`` means the payload
    # carried no account identity (a ping, a malformed body) and is dropped
    # without further work.
    provider_account_id: str | None

    # Resources to re-ingest (repo full names, channel ids). Empty means
    # "nothing to sync" — an event that was understood but implies no work,
    # like a push to a non-default branch.
    resources: list[str] = field(default_factory=list)

    # Set when the provider says the install is GONE FOR GOOD (uninstalled,
    # token permanently revoked). The host revokes the credential and skips
    # syncing. Distinct from an empty ``resources``: one is "no work", this is
    # "stop, and forget the token material".
    revoke: bool = False

    # Set when the install is temporarily unusable but reversible — GitHub's
    # ``installation.suspend``, or an equivalent pause another provider
    # exposes. Distinct from ``revoke``: the connection survives, its sync
    # status degrades to "error" instead, and an ``unsuspend``/resume delivery
    # heals it the same way any other successful sync does. Treating a
    # reversible pause as a permanent revoke would force every affected user
    # through a full reconnect for a state the provider itself is going to
    # clear.
    suspend: bool = False

    # Provider delivery id read from inside the signed body, for the host to
    # claim before fan-out so a replayed delivery is not processed twice. The
    # host owns the claim and its window; a connector only reports the id.
    # ``None`` when the provider sends nothing reliable to dedupe on, in which
    # case the handler being idempotent is the whole protection.
    delivery_id: str | None = None


class WebhookVerifier(ABC):
    """Authenticates an inbound webhook, before its body is parsed.

    Verification must happen over the *raw* request bytes. Re-serializing a
    parsed body changes key order and escaping, which breaks every HMAC scheme
    in use — so this returns the verified bytes and handlers parse those,
    rather than declaring parsed body parameters.
    """

    # Declared empty so a subclass's own __slots__ actually takes effect:
    # __slots__ only suppresses __dict__ when *every* class in the MRO declares
    # it, so omitting it here would silently give every verifier a __dict__ and
    # make the subclass declaration decorative.
    __slots__ = ()

    @abstractmethod
    async def verify(self, body: bytes, headers: Mapping[str, str]) -> bytes:
        """Return the verified raw body, or raise :class:`~.errors.WebhookVerificationError`.

        ``body`` is the request body exactly as received and ``headers`` the
        request headers; header names are matched case-insensitively, so a
        host may pass them in whatever case its framework keeps.
        """
        raise NotImplementedError


class Credential(Protocol):
    """The stored connection a connector may read, as the host hands it over.

    A ``typing.Protocol``, not a model: cognee's ``IntegrationCredential`` row,
    cloud-backend's, and a ``SimpleNamespace`` in a test all satisfy it by
    having these attributes. It lists every field a connector may read and
    deliberately nothing else — no status, no user, no ciphertext. Token
    material is not here either: the host decrypts it and passes it as
    ``tokens`` to the methods that need it.
    """

    provider: str
    provider_account_id: str
    account_label: str | None
    provider_metadata: dict[str, Any] | None
    scopes: str | None
    token_expires_at: datetime | None
    auth_type: str


class Connector(ABC):
    """Base class for a connector.

    Subclass directly only for Tier 3 (an auth model that is not
    OAuth2-with-a-bearer-token). For Tier 1 and 2, write a
    :class:`~.descriptor.ConnectorDescriptor` instead and let
    :class:`~.oauth2.DescriptorConnector` implement these methods.

    Instances are singletons held in the registry and shared across requests,
    so they must be stateless: no per-request or per-user attributes. Config
    is read through the provider's settings object, which is process-wide.
    """

    #: URL path segment and credential discriminator — ``"github"`` serves
    #: ``/api/v1/integrations/github/*``. Must be stable forever: it is written
    #: into every credential row, so renaming it orphans every existing
    #: connection.
    provider: ClassVar[str]

    #: Shown in the connector catalog and the UI card.
    display_name: ClassVar[str]

    #: How a connection is established. ``oauth2`` is the full contract;
    #: ``api_key`` (the user pastes a key) and ``local`` (a vault on disk, no
    #: account at all) make the OAuth members raise ``NotImplementedError``
    #: and tell a host which connect flow to offer.
    auth: ClassVar[AuthKind] = "oauth2"

    @abstractmethod
    def authorize_url(self, state: str) -> str:
        """The provider URL the browser is sent to, carrying the signed ``state``.

        ``state`` is minted by the host (:mod:`~.oauth_state`) and binds this
        install to the user who started it. It must be passed through
        verbatim: the callback is necessarily unauthenticated (the browser
        arrives from the provider with no session), so a valid state is the
        *only* thing tying the returned code to a user.
        """
        raise NotImplementedError

    @abstractmethod
    async def exchange_callback(self, code: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Turn the callback's query parameters into raw account data.

        ``params`` is the full query string, for providers that send more than a
        code — GitHub's ``installation_id`` is the type case. Nothing in it is
        trustworthy: it arrives on an unauthenticated endpoint. A value that
        determines *what* gets connected must be verified against the provider
        (that the caller actually has access to that installation) before it is
        used, or the endpoint is a confused deputy that connects an account its
        caller has no claim to. Raise :class:`PermissionError` specifically
        for that check failing — every connector authenticating an installation
        this way should raise the same type, so a caller can tell "this account
        is not yours" apart from "the provider rejected the exchange" without
        each connector inventing its own exception for the same failure.

        Raise on failure generally. Several providers report OAuth errors as
        HTTP 200 with an error field in the body, so status alone cannot be
        trusted.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_installation(self, account_data: dict[str, Any]) -> ConnectorInstallation:
        """Normalize :meth:`exchange_callback`'s output for the credential store.

        The one place a provider's response shape is interpreted. Everything
        downstream — storage, fan-out, the UI — reads
        :class:`ConnectorInstallation` and never the provider's own JSON.

        Split secret material (access/refresh tokens — anything that grants
        access) into ``token_payload``, which the host encrypts at rest;
        anything else (display/routing data such as a bot user id) goes in
        ``provider_metadata``, which is stored in the clear. Never put secret
        material in ``provider_metadata``.
        """
        raise NotImplementedError

    @abstractmethod
    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """What to ingest for this connection, and the short-lived token to do it.

        ``tokens`` is the decrypted token payload the host stored from
        :meth:`parse_installation` (or :meth:`refresh`); the host decrypts, the
        connector reads.

        Async because building a scope legitimately means calling the provider:
        GitHub mints an installation token and lists the installation's
        repositories here. That enumeration belongs on this side, where the
        provider credentials are.

        ``resources`` is the subset a webhook named, or ``None`` on the connect
        path meaning "everything". It is passed *in* rather than substituted
        afterwards so a connector can skip enumeration it does not need: for
        GitHub, a push to one repository must not list the whole installation
        and discard the result, which would make every commit cost a full
        re-enumeration.

        Never call this with an empty-but-not-``None`` ``resources`` meaning
        "nothing to sync" — :attr:`WebhookDispatch.resources` already uses
        empty for exactly that, but :attr:`SyncScope.items` cannot also mean
        both "nothing" and "everything" with the same empty list, so this
        method rejects that call instead of silently full-syncing. A webhook
        that implies no work must be skipped by the caller before it gets here.

        Called on connect and on every webhook that implies work. May mint a
        fresh provider token; must not return a durable app secret — the
        result is shipped to whatever runs the sync.
        """
        raise NotImplementedError

    # ---- Optional hooks. Defaults are correct for a provider that does not
    # ---- support the capability, so a Tier 1 connector overrides none of them.

    def webhook_verifier(self) -> WebhookVerifier | None:
        """The verifier for this provider's webhooks, or ``None`` if it has none.

        ``None`` makes ``POST /{provider}/events`` answer 404 — the same answer
        an unknown provider gets, so the route reveals nothing about which
        providers are configured. Polling connectors legitimately return
        ``None``.
        """
        return None

    def dataset_name(self, credential: Credential) -> str | None:
        """The dataset this connection's content lands in, for an opted-in delete.

        ``None`` means the connector has no dataset of its own to name — the
        host then leaves data alone on disconnect.
        """
        return None

    def webhook_delivery_key(self, raw_body: bytes, headers: Mapping[str, str]) -> str | None:
        """A delivery key computable with no provider call, if there is one.

        ``None`` by default: most of what :meth:`handle_webhook` does is cheap
        (JSON decode, dict lookups), but it is not required to be — the same
        method that computes the delivery id can also mint a provider token and
        make a listing call for one event type (a GitHub unsuspend enumerates
        what came back into scope). Claiming the delivery only happens *after*
        the whole method returns, which means a replayed delivery — and
        GitHub's signature scheme has no timestamp, so a captured one replays
        forever — pays for that provider call again on every single replay
        before dedupe ever sees it.

        Override this when the id can be read from the raw request alone, the
        same restriction :meth:`handle_webhook`'s own docstring places on where
        a safe delivery id may come from (inside the signed body, never a
        header). A host claims on this key *before* calling
        :meth:`handle_webhook` at all, so a replay is rejected without ever
        reaching the provider call the id does not need.
        """
        return None

    async def handle_webhook(self, raw_body: bytes, headers: Mapping[str, str]) -> WebhookDispatch:
        """Reduce a verified delivery to an account and the resources to re-sync.

        Runs after signature validation and before the webhook is acknowledged,
        so a failure remains visible to the provider and can be retried.
        ``headers`` is lower-cased by the host.

        Must be idempotent. Every provider redelivers, none of them guarantee
        ordering, and for ``sha256_prefix``/``bare_hex`` schemes (no timestamp)
        nothing in the signature carries a replay window either — a captured
        delivery re-verifies forever, and for those two schemes the HMAC covers
        ``raw_body`` only, so nothing outside the body (a header, in
        particular) is covered by the signature either. Return a delivery id
        read from *inside the signed body* on :attr:`WebhookDispatch.delivery_id`
        so the host can claim it before fan-out — never use a header, even one
        that looks like it carries a delivery id. A header-sourced key can be
        changed by an attacker on a captured, still-validly-signed delivery
        without invalidating the signature, which defeats the dedupe entirely
        rather than merely weakening it. For a Tier 1/2 descriptor connector,
        :attr:`~.descriptor.ConnectorDescriptor.webhook_delivery_id_path` is
        the wired-up version of this. It is not mandatory — some providers send
        no id at all — but where one exists in the body, relying on it is
        materially cheaper and more precise than "handler happens to be
        idempotent".
        """
        raise NotImplementedError(f"{self.provider} declares no webhook handling")

    @classmethod
    def supports_resource_selection(cls) -> bool:
        """Whether this connector can list resources for a picker.

        Derived from whether :meth:`list_resources` is overridden, rather than
        declared as a separate flag. A flag could disagree with the
        implementation in both directions — claiming a picker that 404s, or
        hiding one that works — and the catalog would then be advertising
        something untrue. There is nothing to keep in sync here.

        Answerable without a credential and without calling the provider, which
        is what lets the catalog carry it: the catalog is not user-scoped, so it
        has no credential to call anything with.
        """
        return cls.list_resources is not Connector.list_resources

    async def revoke_remote(self, credential: Credential, tokens: Mapping[str, Any]) -> None:
        """Best-effort revocation at the provider. Default: do nothing.

        Deliberately a no-op by default, and left that way for some providers
        on purpose. For a GitHub App the only remote equivalent is deleting the
        installation — uninstalling from the entire org — which is far too
        destructive for one user clicking Disconnect. Local revocation stops
        token use, which is the cut-off that actually matters.

        Never raise: the local revoke must succeed regardless of whether the
        provider is reachable.
        """
        return

    async def refresh(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> RefreshedTokens | None:
        """Refresh an expiring token. Default: nothing to refresh.

        Only meaningful for providers that issue refresh tokens. Auth models
        that mint per-operation (GitHub App) have nothing to refresh and return
        ``None``; so does a connector whose stored ``tokens`` carry no refresh
        token. A returned :class:`RefreshedTokens` is what the host stores in
        place of ``tokens``.
        """
        return None

    async def list_resources(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> list[ConnectorResource]:
        """What this connection could ingest, for the picker UI.

        The default returns nothing and leaves :meth:`supports_resource_selection`
        false, which is the honest answer for a descriptor: it cannot enumerate
        a provider's resources, because doing so needs a provider-shaped API
        call. A host consults :meth:`supports_resource_selection` before
        offering a picker and never calls this for a connector that says no.

        May call the provider. It is a read: it must not change the selection,
        only report what is available. The *current* selection is read from
        the credential by the host, not from here.
        """
        return []

    @property
    def poll_interval_seconds(self) -> int | None:
        """Periodic sync cadence; ``None`` leaves this connector webhook-driven."""
        return None

    def poll_targets(self, credential: Credential) -> list[str] | None:
        """Resources to re-check on a polling schedule, for providers without webhooks.

        Not every provider can push. ``None`` means this connector is
        webhook-driven and the scheduler skips it; a list means poll these. An
        empty list asks :meth:`sync_scope` to discover the saved scope.
        Returning a value requires a positive :attr:`poll_interval_seconds`.
        """
        return None

    @property
    def content_type(self) -> str | None:
        """cognee ingestion mode for this provider's content — ``"code"`` for
        repositories, ``None`` for ordinary documents."""
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} provider={self.provider!r}>"


def account_slug(raw: str, *, fallback: str = "account") -> str:
    """Slugify an external account name for use in a dataset name.

    Shared so every connector slugifies identically — a dataset name is
    effectively an identifier, and two connectors disagreeing on how to
    normalize one would split a single account's data across two datasets.

    Lossy on purpose, in two ways: punctuation collapses (``My-Org`` and
    ``my.org`` both become ``my_org``) and case folds (``ACME`` and ``acme``
    both become ``acme``). Within one user either collision merges two
    accounts' data into one dataset, which is the lesser evil against dataset
    names carrying arbitrary provider-controlled text. Nothing here can safely
    refuse it — the account name is provider-controlled and there is no
    non-lossy name to fall back to — so a log for support is the whole
    defense.

    This function does no logging: the collision log lives in
    :func:`dataset_name_for`, which is the caller that knows the provider.
    That matters because the collision worth reporting is "two ids, one
    DATASET", and a dataset name is ``{provider}_{slug}`` — two providers
    whose accounts normalize alike share no dataset, so warning from here
    reported collisions that were not collisions. Earlier versions also
    warned on normalization rather than on an observed collision, which fired
    on every sync and every delivery for the two shapes the lossiness exists
    for (case-sensitive Slack team ids and GitHub node ids normalize every
    single call while two differing only in case do not exist).

    Not logged for a canonical UUID, even though it has dashes and therefore
    also gets collapsed: a UUID's dashes sit at fixed positions and two
    different UUIDs never collapse to the same slug, so there is no realistic
    collision to warn about. Warning anyway would make this fire on every
    sync for any provider whose ids are UUIDs (Linear's org ids, for one) —
    100% noise for the most common id shape on the roadmap, drowning out the
    actual signal this log exists for (an arbitrary human-typed name like
    ``My-Org``/``my.org``, or an all-punctuation id collapsing to the shared
    ``fallback``).
    """
    # NFC first: "café-org" typed on one platform and normalized to NFD on
    # another is two different byte sequences for the same visible text, and
    # both are alphanumeric-per-character to isalnum() either way — without
    # normalizing to one form first, they slugify to two different strings and
    # split one logical account across two datasets, exactly the collision
    # this helper exists to prevent.
    raw = unicodedata.normalize("NFC", raw)
    cleaned = "".join(char if char.isalnum() else "_" for char in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    slug = cleaned.strip("_").lower() or fallback

    return slug


# (provider, slug) -> the first raw id seen for it, and the pairs already
# warned about. Keyed on the PROVIDER too, because the thing being collided
# is a dataset name and dataset_name_for puts the provider in it: slack_acme
# and github_acme are two different datasets, and an earlier version keyed on
# the bare slug reported them as one.
_slug_origins: dict[tuple[str, str], str] = {}
_slug_collisions: set[tuple[str, str]] = set()

# Bounded because both are keyed on provider-controlled ids and nothing
# evicts them. Cleared wholesale rather than LRU-evicted: losing the history
# costs at most a missed duplicate warning, and this is a diagnostic, not a
# guard.
_SLUG_ORIGIN_LIMIT = 10_000


def _note_slug(provider: str, raw: str, slug: str) -> None:
    """Warn ONCE when two different raw ids have collapsed to one dataset name.

    Three properties, each of which a previous version got wrong:

    Keyed on ``(provider, slug)``, so two providers' accounts reaching the
    same slug are not reported as sharing a dataset — they do not.

    Recorded after warning, so a collision that recurs on every sync and
    every webhook delivery is logged once rather than forever. The un-recorded
    version was the same unactionable-log-per-delivery problem that made the
    original normalization warning useless.

    Only tracked when the slug actually differs from the raw id, or when it
    would collide with something already tracked. Storing every account burnt
    the bound on entries that can never produce a warning, and the wholesale
    clear then dropped the history a colliding pair has to straddle — so
    detection degraded toward zero exactly as the account count grew.
    """
    key = (provider, slug)
    previous = _slug_origins.get(key)

    if previous is not None and previous != raw:
        if key not in _slug_collisions:
            _slug_collisions.add(key)
            logger.warning(
                "account_slug() collapsed two different account ids onto one slug: "
                "%r and %r both -> %r for provider %r — their datasets are now the "
                "same dataset",
                previous,
                raw,
                slug,
                provider,
            )
        return

    if previous is None and slug != raw:
        if len(_slug_origins) >= _SLUG_ORIGIN_LIMIT:
            _slug_origins.clear()
            _slug_collisions.clear()
        _slug_origins[key] = raw
        # DEBUG, not WARNING: normalizing happened, but nothing has collided
        # with it and for most id shapes nothing ever will.
        logger.debug("account_slug() normalized %r -> %r for %r", raw, slug, provider)


def reset_slug_origins() -> None:
    """Empty the slug-collision history. For tests; mirrors the registry's reset."""
    _slug_origins.clear()
    _slug_collisions.clear()


def _is_uuid(raw: str) -> bool:
    """Whether ``raw`` is a canonical UUID string — see :func:`account_slug`.

    Deliberately stricter than ``UUID(raw)`` succeeding: the constructor
    strips dashes from anywhere in the string and also accepts
    ``urn:uuid:...`` and braced forms, so an arbitrary punctuated string like
    ``"1234-5678-1234-5678-1234-5678-1234-5678"`` parses cleanly even though
    its dashes are not at fixed positions — exactly the collision shape this
    function exists to distinguish from a real UUID. Round-tripping through
    ``str()`` and comparing case-insensitively rejects all of those while
    still accepting every real canonical UUID.
    """
    if len(raw) != 36:
        return False
    try:
        return str(UUID(raw)).lower() == raw.lower()
    except (ValueError, AttributeError, TypeError):
        return False


def dataset_name_for(provider: str, provider_account_id: str) -> str:
    """The dataset an account's content lands in: ``{provider}_{slug}``.

    Keyed on ``provider_account_id``, the one field on a credential that is
    stable for the life of the connection — never on ``account_label``, which
    is provider-controlled, mutable, and rewritten on every reconnect. Passing
    a label here would move an account's dataset out from under it the moment
    an org or workspace gets renamed.
    """
    slug = account_slug(provider_account_id)
    if not _is_uuid(provider_account_id):
        # The bookkeeping lives here, not in account_slug, because the
        # collision that matters is "two ids, one DATASET" and the dataset
        # name is what this function builds — account_slug alone does not
        # know the provider and so cannot tell a real collision from two
        # providers coincidentally normalizing alike.
        #
        # UUIDs are skipped entirely rather than just silenced: their dashes
        # sit at fixed positions so two different UUIDs can never collapse to
        # one slug, and they are the most common id shape on the roadmap
        # (Linear's org ids), so tracking them would spend the bound above on
        # entries that can never warn.
        _note_slug(provider, provider_account_id, slug)
    return f"{provider}_{slug}"


__all__ = [
    "AUTH_KINDS",
    "AuthKind",
    "Connector",
    "ConnectorInstallation",
    "ConnectorResource",
    "Credential",
    "RefreshedTokens",
    "SyncScope",
    "WebhookDispatch",
    "WebhookVerifier",
    "account_slug",
    "dataset_name_for",
    "reset_slug_origins",
]
