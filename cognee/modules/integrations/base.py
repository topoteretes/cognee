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
    account_label: Optional[str] = None
    scopes: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    auth_type: str = "oauth2"


class OAuthIntegration(ABC):
    """One third-party OAuth2 connector, registered under ``provider``.

    ``provider`` is the discriminator stored on every
    :class:`IntegrationCredential` row for this connector, and the path
    segment the generic router dispatches on (``/{provider}/authorize`` etc).
    It must be unique across every registered integration.
    """

    provider: ClassVar[str]
    settings_cls: ClassVar[type[IntegrationSettings]]

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
        """

    @abstractmethod
    def state_signing_secret(self) -> str:
        """The secret this provider's OAuth CSRF state is signed/verified with.

        Often the same secret used to verify inbound webhooks (as with
        Slack) — but doesn't have to be; a provider with no inbound webhooks
        still needs a secret here.
        """

    @abstractmethod
    def frontend_base_url(self) -> str:
        """Where the browser lands once the OAuth round-trip completes."""

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort remote token revoke, called on disconnect.

        Default no-op. Override where the provider exposes a revoke
        endpoint — until then, disconnect only marks the local credential
        revoked and the remote token stays live until it expires or the
        user removes the app from their side.
        """
        return None

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Refresh an expiring token in place.

        Default no-op. Override for providers whose tokens actually expire
        and support a refresh grant; callers should not assume this rotates
        anything unless the concrete integration documents that it does.
        """
        return None


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
