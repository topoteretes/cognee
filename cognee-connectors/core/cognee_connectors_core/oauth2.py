"""A working connector built from a descriptor alone.

Moved from cloud-backend ``modules.integrations.framework.oauth2``
(cognee-saas-backend, 2026-10-26) and re-expressed on the host-agnostic
contract: app credentials come from a :class:`~.settings.ConnectorSettings`-
shaped object instead of the cloud's secret blobs, HTTP goes through an
injectable :class:`~.http.HttpClient` so tests never touch the network, token
material is passed in by the host, and delivery dedupe is the host's — the
delivery id is still read out of the signed body and handed back on the
dispatch for it to claim.

This is the class that makes Tier 1 real: given a
:class:`~.descriptor.ConnectorDescriptor`, it implements every abstract method
of :class:`~.base.Connector`, so a standard-OAuth2 provider ships with no
Python of its own.

Tier 2 subclasses it and overrides exactly one method — nearly always
:meth:`DescriptorConnector.resolve_account`, for a provider whose token
response does not say which account authorized. Overriding more than one or
two is the signal that a provider is really Tier 3 and should subclass
:class:`~.base.Connector` directly, where it is not fighting assumptions that
do not apply to it.

**Known limits, deliberately not solved with descriptor fields.** Client
credentials go in the request body (``client_id``/``client_secret`` form
fields), which is what the roadmap's providers use; a provider requiring HTTP
Basic instead is Tier 2 territory until a real one turns up, at which point
``token_auth_style`` is the field to add.
"""

import json
import logging
import math
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from cognee_connectors_core.base import (
    Connector,
    ConnectorInstallation,
    Credential,
    RefreshedTokens,
    SyncScope,
    WebhookDispatch,
    WebhookVerifier,
    dataset_name_for,
)
from cognee_connectors_core.descriptor import (
    ACCESS_TOKEN_FIELDS as _ACCESS_TOKEN_FIELDS,
)
from cognee_connectors_core.descriptor import (
    REFRESH_TOKEN_FIELD as _REFRESH_TOKEN_FIELD,
)
from cognee_connectors_core.descriptor import ConnectorDescriptor, resolve_path
from cognee_connectors_core.http import AiohttpClient, HttpClient
from cognee_connectors_core.settings import require
from cognee_connectors_core.signatures import HmacVerifier

logger = logging.getLogger(__name__)

# Field names looked for in a token response are imported from descriptor.py
# above rather than declared here: descriptor.py's metadata_paths guard has to
# refuse those same names, and two copies drifting apart is how a path reading
# token material gets past it. Standard OAuth2 names, which the roadmap's
# providers follow; a provider that does not is Tier 2.
_EXPIRES_IN_FIELD = "expires_in"

# Real tokens are tens to a few hundred bytes; even a fat JWT stays under a few
# KB. This is generous headroom, not a realistic ceiling — its job is to stop
# a compromised or misconfigured token endpoint from having whatever it
# returns encrypted and stored verbatim (the host's ciphertext column is
# unbounded, so nothing downstream would otherwise reject it).
_MAX_TOKEN_FIELD_BYTES = 65536

# The settings attributes a webhook secret is read from, in order. GitHub-shaped
# providers call it ``webhook_secret``; Slack calls the same thing its
# ``signing_secret`` and cognee's SlackSettings kept that name.
_WEBHOOK_SECRET_FIELDS = ("webhook_secret", "signing_secret")


def _scalar_or_none(
    value: Any, *, provider: str, path: str, context: str, source: str = "webhook"
) -> Any | None:
    """Drop a non-scalar resolved value with a log, or return it unchanged.

    A dict/list at a descriptor path is a wiring mistake, not a real value,
    and ``str()``-ing it silently produces a Python repr that becomes whatever
    everything downstream reads or routes by. This is the *shape* half of that
    check only — it deliberately does not stringify, so a caller storing
    arbitrary provider JSON (``_metadata``) keeps a number as a number.

    Degrades rather than raising, which is right for both callers now that
    ``source`` distinguishes them: on a webhook the body is untrusted traffic
    the provider changes without notice, and on the connect path these are the
    *optional* fields, where dropping a mis-shaped label is strictly better
    than failing an otherwise-good install. The one connect-path value that
    does raise is the account id, in :meth:`parse_installation` — without it
    the credential can never be routed to at all.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        logger.warning(
            "%s %s has a non-scalar value at '%s' (%s) while resolving %s; dropping",
            provider,
            source,
            path,
            type(value).__name__,
            context,
        )
        return None
    if isinstance(value, str) and "\x00" in value:
        # Postgres rejects a NUL byte in a text column outright, so this
        # would abort the whole connect transaction from account_label or
        # scopes. provider_metadata is JSON (not JSONB) and would accept the
        # \u0000 escape, but there is no value in letting one through on the
        # column where it happens to survive.
        logger.warning(
            "%s %s has a NUL byte at '%s' while resolving %s; dropping",
            provider,
            source,
            path,
            context,
        )
        return None
    if isinstance(value, float) and not math.isfinite(value):
        # json.loads accepts the non-standard NaN/Infinity/-Infinity literals
        # by default, and provider_metadata is a JSON column: a non-finite
        # float round-trips through here fine and then fails the INSERT
        # ("Out of range float values are not JSON compliant"), taking the
        # whole connect transaction with it. Only reachable since this guard
        # stopped stringifying — str() used to flatten it to "nan" — so it is
        # this function's own regression to catch.
        logger.warning(
            "%s %s has a non-finite float at '%s' while resolving %s; dropping",
            provider,
            source,
            path,
            context,
        )
        return None
    return value


def _scalar_str_or_none(value: Any, **kwargs: Any) -> str | None:
    """:func:`_scalar_or_none` for the fields stored as text, refusing booleans.

    Here rather than at each call site, because guarding the account id alone
    was the shape of the bug this closes: a JSON ``true`` at
    ``webhook_delivery_id_path`` became the literal ``"True"``, i.e. a
    CONSTANT dedupe key, so every delivery in the window after the first was
    silently discarded — and the value is provider-controlled. The same
    ``str(bool)`` also produced ``account_label="True"`` and
    ``scopes="False"``, which are only cosmetic by comparison.

    Booleans stay legal in :func:`_scalar_or_none`, which feeds the JSON
    ``provider_metadata`` column: there a ``true`` is stored as a boolean and
    means something. It is the stringification that destroys the distinction,
    so the guard belongs on the stringifying path only.
    """
    if isinstance(value, bool):
        logger.warning(
            "%s %s has a boolean at '%s' while resolving %s; dropping "
            "(a stringified boolean would be a constant, not an identifier)",
            kwargs.get("provider"),
            kwargs.get("source", "webhook"),
            kwargs.get("path"),
            kwargs.get("context"),
        )
        return None
    scalar = _scalar_or_none(value, **kwargs)
    return None if scalar is None else str(scalar)


class DescriptorConnector(Connector):
    """Serves any standard OAuth2 provider from its descriptor.

    ``settings`` is the provider's settings object — a
    :class:`~.settings.ConnectorSettings` subclass instance, or anything with
    ``client_id``, ``client_secret`` and ``redirect_uri`` attributes, plus
    ``webhook_secret`` or ``signing_secret`` when the descriptor verifies
    webhooks. It is read at call time, never captured, so a value the host
    rotates on the settings object takes effect on the next request. ``None``
    is allowed for a descriptor with no OAuth flow and no webhooks.

    ``http`` is the :class:`~.http.HttpClient` every provider call goes
    through; tests pass :class:`~.testing.FakeHttpClient`, hosts take the
    aiohttp default.
    """

    def __init__(
        self,
        descriptor: ConnectorDescriptor,
        settings: Any = None,
        http: HttpClient | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.settings = settings
        self.http: HttpClient = http if http is not None else AiohttpClient()
        # Set as instance attributes rather than class attributes because every
        # descriptor-driven connector shares this one class — a ClassVar here
        # would have each registration overwrite the last.
        self.provider = descriptor.provider
        self.display_name = descriptor.display_name
        self.auth = descriptor.auth

    # ---- Settings ------------------------------------------------------

    def _setting(self, field_name: str) -> str:
        """A required settings value, named by its environment variable when unset."""
        if self.settings is None:
            raise RuntimeError(
                f"{self.provider}: DescriptorConnector was built without settings, so "
                f"{field_name!r} cannot be read"
            )
        return require(self.settings, field_name)

    def _require_oauth2(self, member: str) -> None:
        if self.auth != "oauth2":
            raise NotImplementedError(
                f"{self.provider} declares auth={self.auth!r}; {member} is only meaningful "
                "for an oauth2 connector"
            )

    # ---- Connect -------------------------------------------------------

    def authorize_url(self, state: str) -> str:
        self._require_oauth2("authorize_url()")
        return self.descriptor.build_authorize_url(
            client_id=self._setting("client_id"),
            redirect_uri=self._setting("redirect_uri"),
            state=state,
        )

    async def exchange_callback(self, code: str, params: Mapping[str, str]) -> dict[str, Any]:
        """Exchange the code for tokens, then let :meth:`resolve_account` enrich them.

        ``params`` is ignored here: a standard OAuth2 callback carries nothing
        but ``code`` and ``state``, and the state has already been verified by
        the host. A provider that sends something load-bearing in the query
        string (GitHub's ``installation_id``) must not be served by this class —
        that value arrives unauthenticated and needs verification against the
        provider, which is exactly the Tier 3 boundary.
        """
        self._require_oauth2("exchange_callback()")
        payload = {
            "client_id": self._setting("client_id"),
            "client_secret": self._setting("client_secret"),
            "code": code,
            "redirect_uri": self._setting("redirect_uri"),
            "grant_type": "authorization_code",
        }
        body = await self._post_token_request(payload, what="token exchange")

        extra = await self.resolve_account(body)
        if extra:
            body = {**body, **extra}
        return body

    async def resolve_account(self, token_response: dict[str, Any]) -> dict[str, Any] | None:
        """Tier 2 hook: look up the account identity the token response omits.

        Called once by :meth:`exchange_callback`, after the exchange was
        checked and before :meth:`parse_installation` reads the result. It
        receives the raw token response — bearer token included, so it can
        call the provider through ``self.http`` — and returns extra keys that
        are merged over it (``{"organization": {"id": ..., "name": ...}}``),
        which the descriptor's paths then read like any other field; keys it
        returns win over the response's own. ``None``, the default, leaves the
        response untouched.

        Linear is the type case: its token response omits the organization and
        its webhooks route by it, so a subclass queries the viewer and returns
        it here, and ``account_id_path`` points at the merged key.
        """
        return None

    async def _post_token_request(self, payload: dict[str, str], *, what: str) -> dict[str, Any]:
        """POST to the token endpoint and vet the answer the way providers require.

        The body is checked before the HTTP status: providers disagree on
        whether a rejected exchange is a 4xx or a 200 with an error field, and
        the body is the only signal common to all of them.
        """
        status, body = await self.http.post_form(
            self.descriptor.token_url,
            payload,
            # Without this, several providers (GitHub notably) answer
            # form-encoded and the client cannot decode a response that is
            # otherwise fine.
            {"Accept": "application/json"},
        )
        if isinstance(body, (str, bytes)):
            # A provider mid-outage answers with an HTML error page; the client
            # hands that back as text. Status is included because it is the
            # only useful thing left to say about an unparseable body.
            raise RuntimeError(f"{self.provider} {what} returned a non-JSON body (HTTP {status})")
        if not isinstance(body, dict):
            raise RuntimeError(f"{self.provider} {what} returned a non-object response")

        self.descriptor.check_exchange(body)

        if not 200 <= status < 300:
            # A 3xx too: the client does not follow redirects (see
            # cognee_connectors_core.http), so a redirect is a failed exchange,
            # never a token response to accept.
            raise RuntimeError(f"{self.provider} {what} returned HTTP {status}")
        return body

    def parse_installation(self, account_data: dict[str, Any]) -> ConnectorInstallation:
        self._require_oauth2("parse_installation()")
        descriptor = self.descriptor

        account_id = resolve_path(account_data, descriptor.account_id_path)
        if account_id is None:
            # Without an account id the credential could be stored but never
            # reached by a webhook, which presents as "connected, silently
            # never syncs" — the worst failure shape available. Refuse instead.
            raise RuntimeError(
                f"{self.provider} token response has no value at "
                f"'{descriptor.account_id_path}'; cannot identify the connected account"
            )

        if isinstance(account_id, (dict, list)):
            # resolve_path found something at the path, just not an id — a
            # descriptor pointed at an object or array instead of a scalar
            # leaf. That is a wiring mistake, not a real account identity, and
            # str()-ing a dict produces a "Python repr" string that silently
            # becomes the provider_account_id every webhook is routed against
            # — a foreign key to nothing real.
            raise RuntimeError(
                f"{self.provider} token response has a non-scalar value at "
                f"'{descriptor.account_id_path}' ({type(account_id).__name__}); "
                "cannot identify the connected account"
            )

        if isinstance(account_id, bool):
            # A JSON boolean is a scalar, so neither the shape check above nor
            # the emptiness check below sees it — and str() turns it into the
            # literal "True"/"False", which becomes the provider_account_id
            # every webhook is routed against. Same "foreign key to nothing
            # real" as the dict repr, from a value that looks fine. Checked
            # before int, because bool IS an int in Python: a plain 0 is
            # arguably a legitimate account id and is deliberately allowed.
            raise RuntimeError(
                f"{self.provider} token response has a boolean at "
                f"'{descriptor.account_id_path}'; cannot identify the connected account"
            )

        if not str(account_id).strip():
            # An empty or whitespace-only id resolves, so neither the None
            # check above nor the shape check catches it — but account_slug()
            # falls back to the literal "account" for it, so every connection
            # whose id came back blank lands in one shared
            # "{provider}_account" dataset, and each new one is routed to
            # whichever credential row was found first. Same refusal as a
            # missing id, for the same reason.
            raise RuntimeError(
                f"{self.provider} token response has an empty value at "
                f"'{descriptor.account_id_path}'; cannot identify the connected account"
            )

        token_payload = self._token_payload(account_data)
        if not token_payload:
            # Every field _token_payload looked for came back empty. Unlike a
            # GitHub App install (Tier 3, stores {} on purpose because it mints
            # per operation), a standard OAuth2 exchange always returns a
            # bearer token — an empty result here means the provider used a
            # field name _ACCESS_TOKEN_FIELDS does not know about (camelCase,
            # a nested wrapper), not that there is nothing to store. Refuse
            # rather than persist a credential with no way to authenticate.
            raise RuntimeError(
                f"{self.provider} token response has no value at any of "
                f"{_ACCESS_TOKEN_FIELDS}; cannot find token material to store"
            )

        return ConnectorInstallation(
            provider_account_id=str(account_id),
            token_payload=token_payload,
            provider_metadata=self._metadata(account_data),
            account_label=self._optional_str(account_data, descriptor.account_label_path),
            scopes=self._optional_str(account_data, descriptor.scopes_path),
            token_expires_at=self._expiry(account_data),
        )

    def _token_payload(self, account_data: Mapping[str, Any]) -> dict[str, str]:
        """Collect the token material for the host to encrypt.

        Only known token fields are copied — never the whole response. Provider
        responses carry plenty of non-secret noise, and sweeping it into the
        encrypted blob makes it invisible to support while protecting nothing.
        """
        payload = {}
        for field_name in _ACCESS_TOKEN_FIELDS:
            value = account_data.get(field_name)
            if value:
                payload[field_name] = self._bounded(field_name, value)
        refresh = account_data.get(_REFRESH_TOKEN_FIELD)
        if refresh:
            # Stored so refresh() can rotate the access token later without
            # forcing a reconnect.
            payload[_REFRESH_TOKEN_FIELD] = self._bounded(_REFRESH_TOKEN_FIELD, refresh)
        return payload

    def _bounded(self, field_name: str, value: Any) -> str:
        text = str(value)
        if len(text.encode()) > _MAX_TOKEN_FIELD_BYTES:
            # The limit itself is left out, even though this never reaches an
            # unauthenticated caller (the host's callback route turns it into
            # a generic redirect before logging it) — the constant is one line
            # above for whoever reads the log, and this message describes the
            # refusal without needing to repeat it.
            raise RuntimeError(
                f"{self.provider} token response's {field_name!r} is too large to store"
            )
        return text

    def _metadata(self, account_data: dict[str, Any]) -> dict[str, Any]:
        """Copy the declared metadata paths, dropping any that is not a scalar.

        Routed through the same shape guard the webhook path uses. Without it
        these two paths disagreed about what a descriptor pointing at an
        object means: the webhook side dropped it with a warning, this side
        wrote ``"{'deep': {'x': 1}}"`` into provider_metadata as a string. A
        descriptor is one object shared by both paths, so a wiring mistake in
        it has to read the same way on both.
        """
        metadata = {}
        for name, path in self.descriptor.metadata_paths:
            value = _scalar_or_none(
                resolve_path(account_data, path),
                provider=self.provider,
                path=path,
                context=f"metadata key {name!r}",
                source="token response",
            )
            if value is not None:
                metadata[name] = value
        return metadata

    def _optional_str(self, account_data: dict[str, Any], path: str | None) -> str | None:
        """A declared optional text field, or ``None`` — see :meth:`_metadata`."""
        if not path:
            return None
        return _scalar_str_or_none(
            resolve_path(account_data, path),
            provider=self.provider,
            path=path,
            context="an optional account field",
            source="token response",
        )

    @staticmethod
    def _expiry(account_data: Mapping[str, Any]) -> datetime | None:
        """Absolute expiry from a relative ``expires_in``, or ``None``.

        Stored absolute because relative is meaningless once persisted. Computed
        from local time, so it inherits whatever clock skew exists between here
        and the provider — fine for deciding when to refresh, not a security
        boundary.
        """
        expires_in = account_data.get(_EXPIRES_IN_FIELD)
        if expires_in is None:
            # Not `not expires_in`: a provider sending 0 means "already
            # expired, right now" — real information, and falsy only by
            # coincidence. Collapsing it onto "no expiry info at all" would
            # hand out a token this connector already knows is dead without
            # ever logging the warning below.
            return None
        try:
            return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
        except (TypeError, ValueError, OverflowError):
            # OverflowError is real here, not defensive noise: int() is
            # arbitrary precision, so a provider sending something absurd like
            # 10**12 parses cleanly and only blows up in timedelta's C float
            # conversion — the same trap signatures.py's _parse_timestamp
            # documents for the same reason (an unhandled 500 from a value no
            # real provider would ever send).
            logger.warning("Ignoring unparseable %s: %r", _EXPIRES_IN_FIELD, expires_in)
            return None

    # ---- Refresh -------------------------------------------------------

    async def refresh(
        self, credential: Credential, tokens: Mapping[str, Any]
    ) -> RefreshedTokens | None:
        """Rotate the access token through the ``refresh_token`` grant.

        ``None`` when there is nothing to do: a connector that is not
        ``oauth2``, or stored ``tokens`` with no refresh token (a provider that
        issues non-expiring tokens). Otherwise the provider's answer is vetted
        like a code exchange and merged over ``tokens``: fields the provider
        returned replace the old ones, a refresh token it did not return is
        kept (most providers do not rotate it), and any key this class does
        not know about is carried over untouched. A refresh that comes back
        without an access token is an error, not a silent downgrade.
        """
        if self.auth != "oauth2":
            return None
        refresh_token = tokens.get(_REFRESH_TOKEN_FIELD)
        if not refresh_token:
            return None

        payload = {
            "client_id": self._setting("client_id"),
            "client_secret": self._setting("client_secret"),
            "grant_type": "refresh_token",
            "refresh_token": str(refresh_token),
        }
        body = await self._post_token_request(payload, what="token refresh")

        rotated = self._token_payload(body)
        if not any(rotated.get(field_name) for field_name in _ACCESS_TOKEN_FIELDS):
            raise RuntimeError(
                f"{self.provider} token refresh returned no value at any of "
                f"{_ACCESS_TOKEN_FIELDS}; keeping the stored tokens"
            )
        token_payload = {**tokens, **rotated}
        return RefreshedTokens(
            token_payload=token_payload,
            token_expires_at=self._expiry(body),
            scopes=self._optional_str(body, self.descriptor.scopes_path),
        )

    # ---- Sync ----------------------------------------------------------

    async def sync_scope(
        self,
        credential: Credential,
        tokens: Mapping[str, Any],
        *,
        resources: list[str] | None = None,
    ) -> SyncScope:
        """What the sync should ingest for this connection.

        A descriptor can pass through resource ids named by a webhook, but it
        cannot enumerate an account on a full sync. Enumeration needs a
        provider-shaped API call and therefore a Tier 2/3 override. Refuse that
        case explicitly: an empty ``items`` list cannot safely mean both
        "everything" and "nothing", and a generic runner has no provider app
        credentials with which to discover the difference.

        ``resources=None`` (the connect path) and ``resources=[]`` are NOT
        interchangeable, even though :attr:`~.base.SyncScope.items` cannot
        represent both: ``None`` means "everything", so it collapses to empty
        items on purpose. An empty-but-not-None ``resources`` is a webhook
        that named accounts but implied no work (:attr:`~.base.WebhookDispatch.resources`'s
        own "empty means nothing to sync") — silently reusing the same empty
        ``items`` for that would turn a no-op event into a full re-sync. The
        caller must never reach this method for that case; it should check
        ``WebhookDispatch.resources`` and skip the sync entirely. This guard
        makes that contract impossible to violate silently.

        The dataset is keyed on ``provider_account_id``, never ``account_label``
        — see :func:`~.base.dataset_name_for`. ``account_label`` is rewritten
        on every connect and reconnect, so an org rename, a workspace rename,
        or a reconnect that first resolves a label that was previously missing
        would otherwise move an account's content to a brand-new dataset
        mid-connection, silently orphaning everything ingested before it.
        """
        if resources is None:
            raise NotImplementedError(
                "A descriptor-only connector cannot enumerate a full account; "
                "override sync_scope() for this provider"
            )
        if not resources:
            raise ValueError(
                "sync_scope() called with an empty (non-None) resources list — this "
                "would sync everything, not nothing. A webhook that implies no work "
                "must be skipped before calling sync_scope, not passed through here."
            )
        return SyncScope(
            dataset_name=dataset_name_for(self.provider, credential.provider_account_id),
            content_type=self.descriptor.content_type,
            items=list(resources),
            credential=self._access_token(credential, tokens),
        )

    def _access_token(self, credential: Credential, tokens: Mapping[str, Any]) -> str | None:
        """The stored bearer token, read from the payload the host decrypted.

        ``None`` when there is nothing stored — normal for an auth model that
        mints per operation, and the source is what decides whether that is
        workable for its provider.
        """
        expires_at = credential.token_expires_at
        if expires_at is not None:
            if expires_at.tzinfo is None:
                # SQLite hands naive datetimes back; the host stored UTC.
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < datetime.now(timezone.utc):
                # The host is expected to call refresh() before a sync. Handing
                # the token out anyway isn't wrong exactly (there is nothing
                # better to return), but doing it silently turns every
                # subsequent failure into an untraceable 401 loop. Logging here
                # is what makes that loop attributable to "token expired,
                # nobody refreshed it" instead of a mystery on the sync side.
                logger.warning(
                    "%s access token expired at %s and was not refreshed; handing it out anyway",
                    self.provider,
                    expires_at,
                )

        for field_name in _ACCESS_TOKEN_FIELDS:
            token = tokens.get(field_name)
            if token:
                return str(token)
        return None

    def dataset_name(self, credential: Credential) -> str | None:
        return dataset_name_for(self.provider, credential.provider_account_id)

    # ---- Webhooks ------------------------------------------------------

    def _webhook_secret(self) -> str | None:
        """The webhook secret off the settings object, resolved per request."""
        if self.settings is None:
            return None
        for field_name in _WEBHOOK_SECRET_FIELDS:
            value = getattr(self.settings, field_name, None)
            if value:
                return str(value)
        return None

    def webhook_verifier(self) -> WebhookVerifier | None:
        descriptor = self.descriptor
        if descriptor.signature_scheme is None:
            return None
        return HmacVerifier(
            provider=self.provider,
            scheme=descriptor.signature_scheme,
            header=descriptor.signature_header or "",
            timestamp_header=descriptor.signature_timestamp_header,
            # Resolved per request rather than captured here, so a rotated
            # secret is picked up without re-registering the connector.
            secret_provider=self._webhook_secret,
        )

    def _delivery_id(self, payload: Any) -> str | None:
        """The per-delivery id out of a decoded body, when the descriptor names one.

        Sourced from the body, not a header — for sha256_prefix/bare_hex the
        HMAC covers raw_body only, so a header carries no guarantee it wasn't
        changed after signing, and keying dedupe on one would let a replay
        dodge it by editing that header alone. See
        :attr:`~.descriptor.ConnectorDescriptor.webhook_delivery_id_path`.
        """
        path = self.descriptor.webhook_delivery_id_path
        if not path:
            return None
        delivery_id = _scalar_str_or_none(
            resolve_path(payload, path),
            provider=self.provider,
            path=path,
            context="the delivery id",
        )
        if not delivery_id:
            # Logged, not silent: the descriptor declaring this path is a
            # statement that this connector HAS replay protection, and a
            # provider renaming the field turns that off while every
            # delivery still verifies and every test stays green. There is
            # nothing to do about it in-flight (dropping real deliveries
            # over a missing id would be worse), so the only defense is
            # that it is visible.
            logger.warning(
                "%s webhook has no delivery id at '%s'; replay dedupe is "
                "inactive for this delivery",
                self.provider,
                path,
            )
            return None
        return delivery_id

    def webhook_delivery_key(self, raw_body: bytes, headers: Mapping[str, str]) -> str | None:
        """The delivery id from the signed body alone, for the host to claim early.

        The wired-up version of :meth:`~.base.Connector.webhook_delivery_key`:
        ``None`` unless the descriptor names a
        :attr:`~.descriptor.ConnectorDescriptor.webhook_delivery_id_path`, and
        ``None`` for any body that does not decode or carry it.
        """
        if not self.descriptor.webhook_delivery_id_path:
            return None
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return self._delivery_id(payload)

    async def handle_webhook(self, raw_body: bytes, headers: Mapping[str, str]) -> WebhookDispatch:
        """Reduce a verified delivery to an account and the resources to re-sync.

        Every unrecognized shape resolves to "nothing to do" rather than an
        error. The provider controls this body and may change it without
        notice; malformed or irrelevant shapes are acknowledged as no-ops,
        while actual dispatch failures remain retryable by the provider.
        """
        descriptor = self.descriptor
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # UnicodeDecodeError as well: json.loads decodes bytes before it
            # parses them, so a body that is not valid UTF-8 raises that
            # instead — a ValueError, but not a JSONDecodeError. This method
            # promises above to resolve every unrecognized shape to "nothing
            # to do". Treating malformed input as a no-op avoids pointless
            # provider retries.
            logger.warning("%s webhook body is not JSON; dropping", self.provider)
            return WebhookDispatch(provider_account_id=None)

        # No bool check here: _scalar_str_or_none refuses booleans for every
        # text-stored field at once, which is the point — this used to be
        # guarded per call site and the delivery id was missed.
        account_id = _scalar_str_or_none(
            resolve_path(payload, descriptor.webhook_account_id_path or ""),
            provider=self.provider,
            path=descriptor.webhook_account_id_path or "",
            context="the account id",
        )
        if account_id is None or not account_id.strip():
            # Empty as well as absent: an id of "" routes to no credential at
            # all, and letting it through would have handle_webhook return a
            # dispatch whose account id is the blank string — the same shared
            # bucket parse_installation refuses on the connect path.
            logger.info(
                "%s webhook carries no account id at '%s'; dropping",
                self.provider,
                descriptor.webhook_account_id_path,
            )
            return WebhookDispatch(provider_account_id=None)

        # Reported to the host, which claims it before fan-out. The claim and
        # its window are the host's: a connector only says which id to key on.
        delivery_id = self._delivery_id(payload)

        # headers is lower-cased by the host (see base.py's handle_webhook
        # docstring), but a descriptor is free to write the provider's own
        # canonical header spelling ("X-GitHub-Event") — lower-case both here,
        # the same way HmacVerifier lower-cases what it matches, or every
        # descriptor written that way never matches anything.
        lowered = {str(name).lower(): value for name, value in headers.items()}
        event = lowered.get((descriptor.webhook_event_header or "").lower(), "")

        if event and event in descriptor.webhook_revoke_events:
            # Returned before any resource lookup: a revoke must land even when
            # the payload carries nothing else usable, and it is the one event
            # that must not be dropped for being otherwise empty.
            return WebhookDispatch(
                provider_account_id=account_id,
                revoke=True,
                delivery_id=delivery_id,
            )

        if descriptor.webhook_sync_events and event not in descriptor.webhook_sync_events:
            return WebhookDispatch(
                provider_account_id=account_id,
                delivery_id=delivery_id,
            )

        return WebhookDispatch(
            provider_account_id=account_id,
            resources=self._webhook_resources(payload),
            delivery_id=delivery_id,
        )

    def _webhook_resources(self, payload: Any) -> list[str]:
        """Resources named by a webhook body, normalized to a list of strings.

        A provider may report one changed resource or many at the same path, so
        both shapes are accepted; anything else resolves to empty rather than
        raising, for the reason given in :meth:`handle_webhook`.
        """
        path = self.descriptor.webhook_resource_path
        if not path:
            return []
        value = resolve_path(payload, path)
        if value is None:
            return []
        if isinstance(value, list):
            resources = []
            for item in value:
                scalar = _scalar_str_or_none(
                    item, provider=self.provider, path=path, context="a resource"
                )
                if scalar is not None:
                    resources.append(scalar)
            return resources
        scalar = _scalar_str_or_none(value, provider=self.provider, path=path, context="a resource")
        return [scalar] if scalar is not None else []

    # ---- Polling and ingestion ----------------------------------------

    @property
    def poll_interval_seconds(self) -> int | None:
        return self.descriptor.poll_interval_seconds

    def poll_targets(self, credential: Credential) -> list[str] | None:
        """Empty list for a polling provider, ``None`` for a webhook-driven one.

        Empty means "poll this connection, the source decides what to fetch" —
        the same division of labor as :meth:`sync_scope`.
        """
        if self.descriptor.poll_interval_seconds is None:
            return None
        return []

    @property
    def content_type(self) -> str | None:
        return self.descriptor.content_type


__all__ = ["DescriptorConnector"]
