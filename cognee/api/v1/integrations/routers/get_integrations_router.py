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

The ``/plugins/{plugin_key}/…`` routes are a separate concern sharing the
router: agent plugins (claude-code, codex, MCP clients, ...) are not OAuth
providers but clients of cognee itself. Each connected plugin gets its own
agent sub-user + labeled API key, so status, attribution, and revocation
key off that identity instead of a shared tenant key. Declared before the
generic ``{provider}`` routes so ``plugins`` is never captured as a
provider name.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from fastapi_users.exceptions import UserAlreadyExists

from cognee.api.DTO import OutDTO
from cognee.modules.agents.create_agent import create_agent
from cognee.modules.agents.list_agents import list_agents
from cognee.modules.agents.registry import (
    AGENT_CONFIG_NAME,
    build_agent_connection_id,
    deactivate_agent_connection,
    register_agent_connection,
)
from cognee.modules.integrations.base import OAuthIntegration
from cognee.modules.integrations.connect import complete_installation
from cognee.modules.integrations.credentials import (
    CrossUserConflictError,
    get_active_credential_for_user,
    revoke_credential_by_account,
)
from cognee.modules.integrations.oauth_flow import make_state, validate_state
from cognee.modules.integrations.plugins import get_plugin
from cognee.modules.integrations.registry import get_integration
from cognee.modules.observability import new_span
from cognee.modules.users.api_key.create_api_key import create_api_key
from cognee.modules.users.api_key.delete_api_key import delete_api_key
from cognee.modules.users.api_key.get_api_keys import get_api_keys
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.methods.get_principal_configuration import (
    get_principal_all_configuration,
)
from cognee.modules.users.methods.store_principal_configuration import (
    store_principal_configuration,
)
from cognee.modules.users.models import User

logger = logging.getLogger(__name__)

# Plugin keys with a native AgentConnectionType; anything else registers as
# a generic "sdk" connection in the agent-connection registry.
_PLUGIN_CONNECTION_TYPES: dict[str, str] = {
    "claude-code": "claude_code",
    "opencode": "opencode",
    "mcp": "mcp",
    "api": "api",
}


class AuthorizeUrlDTO(OutDTO):
    authorize_url: str


class ConnectionStatusDTO(OutDTO):
    connected: bool
    account_label: Optional[str] = None
    provider_account_id: Optional[str] = None
    connected_at: Optional[datetime] = None


class DisconnectResultDTO(OutDTO):
    disconnected: bool


class PluginProvisionDTO(OutDTO):
    plugin_key: str
    agent_id: UUID
    api_key: str
    created: bool


def _integration_or_404(provider: str) -> OAuthIntegration:
    try:
        return get_integration(provider)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown integration provider: {provider!r}")


def _plugin_or_404(plugin_key: str) -> dict:
    try:
        return get_plugin(plugin_key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown plugin: {plugin_key!r}")


def _plugin_session_name(plugin_key: str) -> str:
    return f"plugin:{plugin_key}"


async def _find_plugin_agent(user: User, plugin_key: str) -> Optional[User]:
    """Resolve the agent sub-user provisioned for ``(user, plugin_key)``.

    ``create_agent`` derives the agent's internal email deterministically
    (``<plugin_key>+<parent_id>@cognee.agent``), so matching on it among the
    user's child agents is the lookup — no extra mapping table needed.
    """
    internal_email = f"{plugin_key}+{user.id}@cognee.agent"
    for agent in await list_agents(user.id):
        if agent.user.email == internal_email:
            return agent.user
    return None


async def _record_plugin_on_agent(agent_user_id: UUID, plugin_key: str) -> None:
    """Mark the agent as a plugin identity in its principal configuration.

    Stored under the same ``AGENT_CONFIG_NAME`` blob the agent-connection
    registry persists to, so no ``User`` schema change is needed. The first
    ``provisioned_at`` sticks across re-provisions (key rotation isn't a new
    install).
    """
    all_configs = await get_principal_all_configuration(agent_user_id)
    existing_config = {}
    for config in all_configs:
        if config.get("name") == AGENT_CONFIG_NAME:
            existing_config = config.get("configuration", {})
            break

    plugin_entry = existing_config.get("plugin", {})
    provisioned_at = plugin_entry.get("provisioned_at") or datetime.now(timezone.utc).isoformat()

    await store_principal_configuration(
        principal_id=agent_user_id,
        name=AGENT_CONFIG_NAME,
        configuration={
            **existing_config,
            "plugin": {"key": plugin_key, "provisioned_at": provisioned_at},
        },
    )


async def _rotate_agent_api_key(agent_user: User, plugin_key: str) -> str:
    """Revoke every key the plugin agent holds and mint a fresh labeled one."""
    for old_key in await get_api_keys(agent_user):
        await delete_api_key(agent_user, old_key.id)
    new_key = await create_api_key(agent_user, plugin_key)
    return new_key.api_key


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

    # ------------------------------------------------------------------ #
    # Agent plugins (fixed /plugins prefix — before the {provider} routes)
    # ------------------------------------------------------------------ #

    @integrations_router.post("/plugins/{plugin_key}/provision")
    async def provision_plugin(
        plugin_key: str, user: User = Depends(get_authenticated_user)
    ) -> PluginProvisionDTO:
        """Provision (or re-key) a dedicated agent identity for a plugin.

        Idempotent get-or-create: the first call creates an agent sub-user
        for ``(user, plugin_key)`` with a labeled API key; every later call
        returns the same agent but rotates the key (old keys are revoked —
        re-provision *is* the rotation flow). The returned key is shown once
        and never retrievable again.
        """
        with new_span("cognee.integrations.plugins.provision") as span:
            span.set_attribute("cognee.integrations.plugin", plugin_key)
            _plugin_or_404(plugin_key)

            created = True
            try:
                agent_user, api_key = await create_agent(plugin_key, user)
            except UserAlreadyExists:
                # Already provisioned — same agent, new key (rotation).
                created = False
                agent_user = await _find_plugin_agent(user, plugin_key)
                if agent_user is None:
                    # The deterministic agent email exists but isn't among
                    # this user's child agents — a state we can't repair
                    # here without risking a key handout to the wrong owner.
                    raise HTTPException(
                        status_code=409,
                        detail=f"Plugin agent for {plugin_key!r} exists but could not be resolved.",
                    )
                api_key = await _rotate_agent_api_key(agent_user, plugin_key)

            await _record_plugin_on_agent(agent_user.id, plugin_key)

            # Register in the agent-connection registry so /agents/connections
            # (and the status endpoint) see the plugin before its first traffic.
            await register_agent_connection(
                agent_session_name=_plugin_session_name(plugin_key),
                connection_type=_PLUGIN_CONNECTION_TYPES.get(plugin_key, "sdk"),
                source="api",
                user_id=agent_user.id,
                tenant_id=getattr(agent_user, "tenant_id", None),
                metadata={"plugin_key": plugin_key},
            )

            span.set_attribute("cognee.integrations.outcome", "created" if created else "rotated")
            return PluginProvisionDTO(
                plugin_key=plugin_key,
                agent_id=agent_user.id,
                api_key=api_key,
                created=created,
            )

    @integrations_router.delete("/plugins/{plugin_key}")
    async def disconnect_plugin(
        plugin_key: str, user: User = Depends(get_authenticated_user)
    ) -> DisconnectResultDTO:
        """Disconnect a plugin: revoke its API keys, keep its data.

        The agent user and everything it wrote stay — deleting data on
        disconnect would be surprising; full removal stays on
        ``DELETE /api/v1/agents/{agent_id}``. Re-provisioning later revives
        the same identity with a fresh key.
        """
        with new_span("cognee.integrations.plugins.disconnect") as span:
            span.set_attribute("cognee.integrations.plugin", plugin_key)
            _plugin_or_404(plugin_key)

            agent_user = await _find_plugin_agent(user, plugin_key)
            if agent_user is None:
                return DisconnectResultDTO(disconnected=False)

            for api_key in await get_api_keys(agent_user):
                await delete_api_key(agent_user, api_key.id)

            connection_id = build_agent_connection_id(
                agent_session_name=_plugin_session_name(plugin_key),
                user_id=str(agent_user.id),
            )
            await deactivate_agent_connection(agent_user.id, connection_id)
            return DisconnectResultDTO(disconnected=True)

    @integrations_router.post("/{provider}/authorize")
    async def authorize(
        provider: str, user: User = Depends(get_authenticated_user)
    ) -> AuthorizeUrlDTO:
        """Mint the provider's authorize URL for the requesting user."""
        with new_span("cognee.integrations.authorize") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)
            try:
                state = make_state(user.id, signing_secret=integration.state_signing_secret())
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
                logger.warning("%s account already connected elsewhere; user %s", provider, user_id)
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
