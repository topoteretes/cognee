"""Integrations router: OAuth install flow and connection state for every
registered provider, dispatched generically on ``{provider}``.

Route roles differ sharply in their auth model, which is the point:

* ``POST /{provider}/authorize`` — authenticated. Minting the signed OAuth
  state is the permission gate for the whole install; the callback trusts
  the state alone.
* ``GET /{provider}/callback`` — necessarily unauthenticated (the browser
  arrives from the provider's site without a session header). A valid,
  unexpired state is the only credential, and it was only ever issued to the
  connecting user.
* ``GET/DELETE /{provider}/connection`` — authenticated; a user only ever
  sees or disconnects their own connection (credentials are user-scoped, not
  shared across a tenant/org).

Adding a second provider (Notion, GitHub, ...) needs none of these endpoints
touched — only a new ``OAuthIntegration`` registered via
:func:`cognee.modules.integrations.registry.use_integration`. An unknown
``{provider}`` 404s rather than 500ing; every other failure mode redirects
back to the frontend with a coarse outcome slug instead of surfacing a raw
error (the query string ends up in browser history and access logs).
"""

import logging
from datetime import datetime
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from cognee.api.DTO import OutDTO
from cognee.modules.integrations.base import OAuthIntegration
from cognee.modules.integrations.connect import complete_installation
from cognee.modules.integrations.credentials import (
    CrossUserConflictError,
    get_active_credential_for_user,
    revoke_credential_by_account,
)
from cognee.modules.integrations.oauth_flow import make_state, validate_state
from cognee.modules.integrations.registry import get_integration
from cognee.modules.observability import new_span
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User

logger = logging.getLogger(__name__)


class AuthorizeUrlDTO(OutDTO):
    authorize_url: str


class ConnectionStatusDTO(OutDTO):
    connected: bool
    account_label: Optional[str] = None
    provider_account_id: Optional[str] = None
    connected_at: Optional[datetime] = None


class DisconnectResultDTO(OutDTO):
    disconnected: bool


def _integration_or_404(provider: str) -> OAuthIntegration:
    try:
        return get_integration(provider)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown integration provider: {provider!r}")


def _frontend_redirect(integration: OAuthIntegration, outcome: str) -> RedirectResponse:
    try:
        base = integration.frontend_base_url().rstrip("/")
    except RuntimeError:
        # Same require()-style config guard as authorize() — but this runs
        # from inside the callback, with the browser already sitting on our
        # domain, so there is nowhere valid left to redirect it. A clear 503
        # beats a raw, unexplained 500.
        logger.exception(
            "%s frontend_base_url is not configured; cannot redirect (outcome=%s)",
            integration.provider,
            outcome,
        )
        raise HTTPException(
            status_code=503,
            detail=f"{integration.provider} integration is not configured on this server.",
        )
    return RedirectResponse(url=f"{base}/integrations?{integration.provider}={quote(outcome)}")


def get_integrations_router():
    integrations_router = APIRouter()

    @integrations_router.post("/{provider}/authorize")
    async def authorize(provider: str, user: User = Depends(get_authenticated_user)) -> AuthorizeUrlDTO:
        """Mint the provider's authorize URL for the requesting user."""
        with new_span("cognee.integrations.authorize") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)
            try:
                state = make_state(
                    user_id=user.id, signing_secret=integration.state_signing_secret()
                )
                return AuthorizeUrlDTO(authorize_url=integration.authorize_url(state))
            except RuntimeError:
                # A provider's require()-style settings guard raises when its
                # client id/secret/signing key aren't configured — a
                # deploy-time gap, not a per-request fault. Surface it as a
                # clear 503 instead of a bare 500 so the frontend doesn't have
                # to guess why "Connect" failed.
                logger.exception("%s authorize requested but is not configured", provider)
                raise HTTPException(
                    status_code=503,
                    detail=f"{provider} integration is not configured on this server.",
                )

    @integrations_router.get("/{provider}/callback", include_in_schema=False)
    async def callback(provider: str, code: str = "", state: str = "", error: str = ""):
        """OAuth redirect target — state-authenticated, browser-facing."""
        with new_span("cognee.integrations.callback") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)

            if error:
                # The user clicked "Cancel" on the provider's consent screen
                # (or the provider rejected the request) — not a fault, just
                # an aborted install.
                span.set_attribute("cognee.integrations.outcome", "cancelled")
                return _frontend_redirect(integration, "cancelled")

            user_id = validate_state(state, signing_secret=integration.state_signing_secret())
            if user_id is None:
                span.set_attribute("cognee.integrations.outcome", "error_invalid_state")
                return _frontend_redirect(integration, "error_invalid_state")

            try:
                credential = await complete_installation(integration, code=code, user_id=user_id)
            except CrossUserConflictError:
                # The account is already connected to another user; refuse
                # rather than silently reassign it (see upsert_credential).
                logger.warning(
                    "%s account already connected elsewhere; user %s", provider, user_id
                )
                span.set_attribute("cognee.integrations.outcome", "error_already_connected")
                return _frontend_redirect(integration, "error_already_connected")
            except Exception:  # noqa: BLE001 - any exchange/parse failure must redirect, not 500
                # Full trace server-side; the browser only learns that it failed.
                logger.exception("%s OAuth exchange failed for user %s", provider, user_id)
                span.set_attribute("cognee.integrations.outcome", "error_exchange_failed")
                return _frontend_redirect(integration, "error_exchange_failed")

            logger.info(
                "%s account %s connected to user %s",
                provider,
                credential.provider_account_id,
                user_id,
            )
            span.set_attribute("cognee.integrations.outcome", "connected")
            return _frontend_redirect(integration, "connected")

    @integrations_router.get("/{provider}/connection", response_model_exclude_none=True)
    async def connection_status(
        provider: str, user: User = Depends(get_authenticated_user)
    ) -> ConnectionStatusDTO:
        """Connection state for the Integrations page."""
        integration = _integration_or_404(provider)
        credential = await get_active_credential_for_user(user.id, integration.provider)
        if credential is None:
            return ConnectionStatusDTO(connected=False)

        # Token material stays server-side; the frontend only needs display state.
        return ConnectionStatusDTO(
            connected=True,
            account_label=credential.account_label,
            provider_account_id=credential.provider_account_id,
            connected_at=credential.created_at,
        )

    @integrations_router.delete("/{provider}/connection")
    async def disconnect(
        provider: str, user: User = Depends(get_authenticated_user)
    ) -> DisconnectResultDTO:
        """Disconnect the account connected by the requesting user.

        Marks the stored installation revoked, and best-effort asks the
        provider to kill the token on its own side via
        ``integration.revoke_remote``. That call is wrapped here too, on top
        of each adapter's own best-effort handling — a third-party
        integration that doesn't honor the "never raise" contract on
        ``revoke_remote`` still must not block the local disconnect.
        """
        with new_span("cognee.integrations.disconnect") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)
            credential = await get_active_credential_for_user(user.id, integration.provider)
            if credential is None or credential.provider_account_id is None:
                return DisconnectResultDTO(disconnected=False)

            try:
                await integration.revoke_remote(credential)
            except Exception:  # noqa: BLE001 - a remote-revoke failure must never block disconnect
                logger.exception(
                    "%s revoke_remote raised for account %s",
                    provider,
                    credential.provider_account_id,
                )

            await revoke_credential_by_account(integration.provider, credential.provider_account_id)
            return DisconnectResultDTO(disconnected=True)

    return integrations_router
