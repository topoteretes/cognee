"""The cognee host adapter for __DISPLAY_NAME__.

cognee's generic ``/api/v1/integrations/{provider}/...`` routes drive an
``OAuthIntegration``; this one delegates the OAuth contract to ``CONNECTOR`` and
keeps only what the host needs (its settings class, the state-signing secret, the
frontend URL). Host-bound sync code moved from cognee lives next to this module.
"""

from typing import Any

from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration

from __MODULE__.connector import CONNECTOR, DESCRIPTOR
from __MODULE__.settings import __CLASS__Settings, __PROVIDER___settings
from cognee_connectors_core.settings import require


class __CLASS__Integration(OAuthIntegration):
    """__DISPLAY_NAME__ on cognee's integration seam."""

    provider = DESCRIPTOR.provider
    settings_cls = __CLASS__Settings

    def __init__(self, settings: __CLASS__Settings | None = None) -> None:
        self.settings = settings or __PROVIDER___settings

    def authorize_url(self, state: str) -> str:
        """The provider's consent-screen URL the frontend redirects to."""
        return CONNECTOR.authorize_url(state)

    async def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange an OAuth ``code`` for the provider's raw token response."""
        return await CONNECTOR.exchange_callback(code, {})

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        """Turn a raw token response into everything the credential store needs."""
        installation = CONNECTOR.parse_installation(token_response)
        return OAuthInstallation(
            provider_account_id=installation.provider_account_id,
            token_payload=installation.token_payload,
            provider_metadata=installation.provider_metadata,
            account_label=installation.account_label,
            scopes=installation.scopes,
            token_expires_at=installation.token_expires_at,
            auth_type=installation.auth_type,
        )

    def state_signing_secret(self) -> str:
        """The secret this provider's OAuth CSRF state is signed/verified with."""
        return require(self.settings, "client_secret")

    def frontend_base_url(self) -> str:
        """Where the browser lands once the OAuth round-trip completes."""
        return require(self.settings, "frontend_base_url")
