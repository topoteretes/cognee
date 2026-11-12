"""Base contracts for third-party OAuth integrations.

Every provider (Slack today; Notion, GitHub, Google Drive later) needs the
same connect story — mint an authorize URL, exchange a code for tokens, and
turn the resulting token response into exactly what the generic credential
store (:mod:`cognee.modules.integrations.credentials`) needs to persist an
:class:`~cognee.modules.integrations.models.IntegrationCredential.IntegrationCredential`
row. That shared story is what :class:`OAuthIntegration` captures; everything
provider-specific (endpoints, scopes, response shapes) lives in the concrete
adapter, not here.

This is a public extension seam: a provider registers an ``OAuthIntegration``
instance via :func:`cognee.modules.integrations.registry.use_integration`,
the same way a loader registers via ``use_loader`` or a vector adapter via
``use_vector_adapter``. Treat method signatures here as a contract third
parties build against (e.g. a ``cognee-community-integration-<x>`` package)
— changing one is a breaking change for anyone who implemented it.

``revoke_remote``/``refresh`` default to no-ops rather than being abstract:
a first cut of a provider (see the Slack adapter) is allowed to not support
token rotation or remote revocation yet — the generic OAuth flow and router
work either way. Override them once the provider actually needs the
behavior; don't stub them out speculatively for providers that don't exist.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Optional

from fastapi import Request
from pydantic_settings import BaseSettings

from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.users.models import User


class IntegrationSettings(BaseSettings):
    """Common OAuth2 app credentials every provider settings class shares.

    A concrete provider subclasses this with its own ``env_prefix`` (e.g.
    ``NOTION_``) and adds provider-specific fields (Slack's
    ``signing_secret``, for one) — see
    :class:`cognee.modules.integrations.slack.slack_settings.SlackSettings`
    for the reference shape. Fields default to empty rather than failing at
    import so deployments without the integration configured still boot;
    fail loudly at use time instead (see that module's ``require()``).
    """

    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    frontend_base_url: str = ""


@dataclass(slots=True)
class OAuthInstallation:
    """Everything :func:`cognee.modules.integrations.credentials.upsert_credential`
    needs, extracted from one provider's raw OAuth token response.

    A single dataclass rather than several fine-grained extractor methods —
    Slack's own response needs the ``team``/``enterprise`` sub-dicts to
    compute both ``provider_account_id`` and ``account_label``, so splitting
    that extraction across multiple methods would mean parsing the same
    response twice for no benefit.
    """

    provider_account_id: str
    token_payload: dict[str, Any]
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    account_label: str | None = None
    scopes: str | None = None
    token_expires_at: datetime | None = None
    auth_type: str = "oauth2"


class ResourceParentError(ValueError):
    """A ``parent`` the resource picker cannot list under; the route answers 422."""


class ResourceParentNotSupportedError(ResourceParentError):
    """The provider lists all its resources at once and takes no ``parent``."""


class ResourceParentNotFoundError(ResourceParentError):
    """The ``parent`` names no listable resource at the provider; the route answers 404."""


def refuse_parent(provider: str, parent: str | None) -> None:
    """For a provider that lists everything at once: a ``parent`` is a client error."""
    if parent is not None:
        raise ResourceParentNotSupportedError(
            f"{provider} lists all its resources at once and takes no parent"
        )


class OAuthIntegration(ABC):
    """One third-party OAuth2 connector, registered under ``provider``.

    ``provider`` is the discriminator stored on every
    :class:`IntegrationCredential` row for this connector, and the path
    segment the generic router dispatches on (``/{provider}/authorize`` etc).
    It must be unique across every registered integration.
    """

    provider: ClassVar[str]
    settings_cls: ClassVar[type[IntegrationSettings]]
    # Optional provider metadata key used by the generic resource picker.
    # ``None`` means this provider has no selectable sub-resources.
    resource_selection_key: ClassVar[str | None] = None
    # Whether a stored ``null`` selection may mean "all resources" for this provider.
    resource_selection_allows_all: ClassVar[bool] = True

    @abstractmethod
    def authorize_url(self, state: str) -> str:
        """The provider's consent-screen URL the frontend redirects to."""

    @abstractmethod
    async def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange an OAuth ``code`` for the provider's raw token response."""

    @abstractmethod
    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        """Turn a raw token response into everything the credential store needs.

        Split secret material (access/refresh tokens — anything that grants
        access) into ``token_payload``, which is encrypted at rest; anything
        else (display/routing data such as a bot user id) goes in
        ``provider_metadata``, which is stored in the clear. Never put secret
        material in ``provider_metadata``.

        ``provider_metadata`` is **merged** into whatever the row already
        holds, not swapped for it, so that settings written outside the OAuth
        flow (Slack's channel allowlist) survive a reconnect. Two rules follow
        for an adapter: build the dict unconditionally, writing a key with a
        null value rather than leaving it out, since an omitted key keeps its
        stored value; and do not expect to clear the metadata from here.
        """

    @abstractmethod
    def frontend_base_url(self) -> str:
        """Where the browser lands once the OAuth round-trip completes."""

    async def exchange_callback(self, code: str, params: dict[str, str]) -> dict[str, Any]:
        """Exchange one provider callback into the raw token response.

        Default delegates to ``exchange_code`` — the plain OAuth2 shape where
        ``code`` is the only thing the callback carries. Override when the
        provider sends more than a code (GitHub App installs carry an
        ``installation_id`` alongside it); ``params`` is the callback's full
        query string, minus nothing — the adapter picks what it trusts.
        """
        return await self.exchange_code(code)

    def webhook_verifier(self) -> Optional["WebhookVerifier"]:
        """The verifier for this provider's inbound webhooks, if it has any.

        Default ``None``: the generic ``POST /{provider}/events`` route
        rejects deliveries for a provider that doesn't opt in. Returning a
        verifier turns that route on for this provider; pair it with a
        ``handle_webhook`` override that actually does something.
        """
        return None

    async def handle_webhook(self, raw_body: bytes, headers: dict[str, str]) -> None:
        """Process one verified webhook delivery.

        Called off the request path (the route acks the provider first), so
        implementations may do slow work but must not assume a live request.
        ``headers`` are lower-cased — providers put the event name in a
        header (GitHub's ``x-github-event``), not the body.
        """
        return

    async def on_installed(self, credential: IntegrationCredential) -> None:
        """Post-install hook, fired in the background after a successful connect.

        Default no-op. Override for providers whose connect should kick off
        work beyond storing the credential (an initial content sync, say).
        Runs detached from the callback request — failures log, they never
        break the install redirect.
        """
        return

    async def sync_now(self, credential: IntegrationCredential) -> None:
        """Run a user-requested sync, when the provider supports one.

        The default is deliberately a no-op: not every OAuth provider has a
        sync implementation yet, and a generic route must not invent one.
        Providers with a durable or manually-triggered sync override this
        method and the same hook can later be called by a scheduler.
        """
        return

    async def list_resources(
        self, credential: IntegrationCredential, *, parent: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Return provider resources for a generic picker, or ``None``.

        ``parent`` asks for one level of a tree: the resources directly inside it (``None``,
        the top level). A provider that lists everything at once refuses a parent with
        :class:`ResourceParentNotSupportedError` (:func:`refuse_parent`), before any request.
        """
        refuse_parent(self.provider, parent)
        return None

    def invalid_resource_ids(self, resource_ids: list[str]) -> list[str]:
        """The ids of a saved selection this provider can never sync, refused with a 422.

        A shape check, made before the selection is stored: an id that can never be valid
        would fail every later sync. Whether the resource exists is the sync's to find out.
        """
        return []

    async def validate_resource_selection(
        self, credential: IntegrationCredential, resource_ids: list[str]
    ) -> None:
        """Refuse a selection the provider cannot sync, before it is stored.

        Default: accept. Raise :class:`ValueError` with a message safe to show the user
        (the resource picker answers 422 with it); the sync must still check, since a
        stored selection can predate a change at the provider.
        """
        return

    def dataset_name(self, credential: IntegrationCredential) -> str | None:
        """Return the provider dataset to delete on an opted-in disconnect."""
        return None

    def data_deleted_after(self, credential: IntegrationCredential) -> datetime | None:
        """When the data of a connection the provider ended is deleted, if that is scheduled.

        Default ``None``: an ended connection keeps its data. Slack schedules a deletion 14
        days after an uninstall; the connection status shows it so the page can warn the
        owner before it happens.
        """
        return None

    def data_deleted_at(self, credential: IntegrationCredential) -> datetime | None:
        """When the scheduled deletion of an ended connection's data ran, if it did.

        Default ``None``. With :meth:`data_deleted_after` it tells the owner whether the
        data is about to go or is gone.
        """
        return None

    async def forget_related_data(self, credential: IntegrationCredential, user: User) -> None:
        """Delete what the provider's app stored for ``user`` outside ``dataset_name``.

        Called on an opted-in disconnect, after the dataset is deleted. Default
        no-op. Override where the app writes memory of its own (Slack's remembered
        messages); delete only what ``user`` owns.
        """
        return

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort remote token revoke, called on disconnect.

        Default no-op. Override where the provider exposes a revoke
        endpoint — until then, disconnect only marks the local credential
        revoked and the remote token stays live until it expires or the
        user removes the app from their side.
        """
        return

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Refresh an expiring token in place.

        Default no-op. Override for providers whose tokens actually expire
        and support a refresh grant; callers should not assume this rotates
        anything unless the concrete integration documents that it does.
        """
        return


class WebhookVerifier(ABC):
    """Authenticates one inbound webhook request for a provider.

    Verification MUST run over the raw request bytes, before any parsing —
    re-serializing a parsed body changes key order/escaping and breaks most
    providers' HMAC schemes (see the Slack adapter for why). Implementations
    return the verified raw body so handlers parse it themselves instead of
    declaring a parsed body parameter.
    """

    @abstractmethod
    async def verify(self, request: Request) -> bytes:
        """Return the raw, verified request body, or raise on bad signature."""
