"""Integrations router: OAuth install flow and connection state for every
registered provider, dispatched generically on ``{provider}``.

Route roles differ sharply in their auth model, which is the point:

* ``POST /{provider}/authorize`` — authenticated. It mints the signed OAuth
  state and sets the install-nonce cookie the callback checks against.
* ``GET /{provider}/callback`` — necessarily unauthenticated (the browser
  arrives from the provider's site without a session header). Two things
  authenticate it, and both are needed: a valid, unexpired state, and the
  nonce cookie from the browser that started the install. The state alone is
  not enough, because it travels in a URL that can be handed to somebody
  else: it says an install was started by some user, not that this browser
  belongs to that user. Completing a relayed authorize URL would otherwise
  attach the consenting person's provider account to whoever started the
  flow. See ``_set_install_nonce``.
* ``GET/DELETE /{provider}/connection`` — authenticated; a user only ever
  sees or disconnects their own connection (credentials are user-scoped, not
  shared across a tenant/org).
* ``POST /{provider}/events`` — unauthenticated webhook receiver; the
  provider's registered ``WebhookVerifier`` (HMAC over raw bytes) is the
  entire auth model. Providers without a verifier 404.

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

import asyncio
import hmac
import logging
import secrets
from datetime import datetime, timezone
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi_users.exceptions import UserAlreadyExists
from pydantic import Field
from sqlalchemy.exc import IntegrityError

from cognee.api.DTO import InDTO, OutDTO
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
    list_active_credentials_for_user,
    revoke_credential_by_account,
    update_provider_metadata,
)
from cognee.modules.integrations.oauth_flow import (
    DEFAULT_STATE_TTL_SECONDS,
    make_state,
    validate_state,
)
from cognee.modules.integrations.plugin_status import (
    PluginStatusRow,
    as_utc,
    coerce_provisioned_at,
    identity_plugin_statuses,
    legacy_plugin_statuses,
    merge_plugin_statuses,
    registry_plugin_statuses,
)
from cognee.modules.integrations.plugins import KNOWN_PLUGINS, get_plugin
from cognee.modules.integrations.registry import get_integration, supported_integrations
from cognee.modules.observability import new_span
from cognee.modules.session_lifecycle.visibility import visible_user_ids
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

# Detached post-install / webhook work (initial syncs, event handling) runs
# off the request path; strong references keep the tasks alive until done —
# same pattern as remember()'s _BACKGROUND_REMEMBER_TASKS.
_BACKGROUND_INTEGRATION_TASKS: set = set()


def _spawn_background(coro, *, description: str) -> None:
    """Run ``coro`` detached, logging (never raising) on failure."""

    async def _guarded():
        try:
            await coro
        except Exception:  # detached work must log, not crash the loop
            logger.exception("%s failed", description)

    task = asyncio.create_task(_guarded())
    _BACKGROUND_INTEGRATION_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_INTEGRATION_TASKS.discard)


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
    account_label: str | None = None
    provider_account_id: str | None = None
    connected_at: datetime | None = None
    # "ok" | "degraded" | None (never synced, or the provider doesn't stamp
    # this yet). Written by connectors that have no webhook to self-heal on —
    # see record_sync_result — so a connection can read as connected while
    # its last sync failed or came back partial.
    sync_status: str | None = None
    last_synced_at: datetime | None = None
    sync_counts: dict[str, int] | None = None


class DisconnectResultDTO(OutDTO):
    disconnected: bool


class IntegrationResourceDTO(OutDTO):
    id: str
    name: str
    description: str | None = None
    attributes: dict = Field(default_factory=dict)
    selected: bool = False


class IntegrationResourceListDTO(OutDTO):
    resources: list[IntegrationResourceDTO]
    selected: list[str] | None = None


class IntegrationResourceSelectionPayload(InDTO):
    resource_ids: list[str] | None = None


class IntegrationResourceSelectionResultDTO(OutDTO):
    selected: list[str] | None = None


class IntegrationSyncResultDTO(OutDTO):
    accepted: bool


class PluginProvisionDTO(OutDTO):
    plugin_key: str
    agent_id: UUID
    api_key: str
    created: bool


class IntegrationStatusItemDTO(OutDTO):
    provider: str
    connected: bool
    account_label: str | None = None
    provider_account_id: str | None = None
    connected_at: datetime | None = None
    sync_status: str | None = None
    last_synced_at: datetime | None = None
    sync_counts: dict[str, int] | None = None


class PluginStatusItemDTO(OutDTO):
    key: str
    connected: bool
    agent_id: UUID | None = None
    provisioned_at: datetime | None = None
    last_active_at: datetime | None = None
    session_count: int = 0
    source: str | None = None


class IntegrationsStatusDTO(OutDTO):
    integrations: list[IntegrationStatusItemDTO]
    plugins: list[PluginStatusItemDTO]


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


async def _find_plugin_agent(user: User, plugin_key: str) -> User | None:
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
    install) — but only if it still parses as a datetime: the blob is
    agent-writable, so a tampered value is repaired here rather than
    preserved.
    """
    all_configs = await get_principal_all_configuration(agent_user_id)
    existing_config = {}
    for config in all_configs:
        if config.get("name") == AGENT_CONFIG_NAME:
            existing_config = config.get("configuration", {})
            break

    plugin_entry = existing_config.get("plugin")
    if not isinstance(plugin_entry, dict):
        plugin_entry = {}
    existing_provisioned_at = coerce_provisioned_at(plugin_entry.get("provisioned_at"))
    provisioned_at = (existing_provisioned_at or datetime.now(timezone.utc)).isoformat()

    await store_principal_configuration(
        principal_id=agent_user_id,
        name=AGENT_CONFIG_NAME,
        configuration={
            **existing_config,
            "plugin": {"key": plugin_key, "provisioned_at": provisioned_at},
        },
    )


async def _rotate_agent_api_key(agent_user: User, plugin_key: str) -> str:
    """Mint a fresh labeled key, then revoke every other key the agent holds.

    Order matters: each key operation commits its own transaction, so minting
    first means a failure anywhere in the rotation never leaves the plugin
    keyless — at worst the old key briefly outlives the new one.
    """
    old_keys = await get_api_keys(agent_user)
    new_key = await create_api_key(agent_user, plugin_key)
    for old_key in old_keys:
        await delete_api_key(agent_user, old_key.id)
    return new_key.api_key


# The state proves an install was started by some cognee user; it does not
# prove the browser completing it belongs to that user, because the state
# travels in a URL anyone can be handed. This cookie is what binds the two:
# it is set on the authenticated authorize call and compared at the callback,
# so a relayed authorize link completed by somebody else is refused instead of
# attaching their provider account to the account that started the flow.
#
# One cookie per *install*, not one shared cookie per provider: two tabs on
# the same provider (a double-click, a second tab) each call authorize once
# and each gets their own cookie in a single Set-Cookie, so there is no
# shared value for a second write to race and no read-modify-write for two
# concurrent authorize calls to lose one nonce to the other. The nonce itself
# is part of the cookie name, so setting one is one atomic operation with
# nothing to merge.
#
# SameSite=Lax is what a top-level GET navigation from the provider needs,
# and requires the API and the app to share a registrable domain; on a
# deployment where they do not, this needs SameSite=None and Secure instead.
#
# Each cookie expires on its own after DEFAULT_STATE_TTL_SECONDS, the same
# window the state itself is valid for, so an abandoned install's cookie
# cleans itself up without anything having to track or cap how many are
# outstanding — the previous single-cookie design needed a cap because one
# cookie's value could grow without bound; this design has no shared value
# to grow.
_INSTALL_NONCE_COOKIE_PREFIX = "cognee_oauth_nonce_"
_INSTALL_NONCE_PATH = "/api/v1/integrations"


def _install_nonce_cookie_name(provider: str, nonce: str) -> str:
    return f"{_INSTALL_NONCE_COOKIE_PREFIX}{provider}_{nonce}"


def _install_nonce_cookie_prefix(provider: str) -> str:
    return f"{_INSTALL_NONCE_COOKIE_PREFIX}{provider}_"


def _pending_install_nonces(request: Request, provider: str) -> list[str]:
    """Every nonce this browser currently holds a pending install for."""
    prefix = _install_nonce_cookie_prefix(provider)
    return [
        name[len(prefix) :]
        for name in request.cookies
        if name.startswith(prefix)
        # isascii() before any of these ever reaches hmac.compare_digest,
        # which raises TypeError on a non-ASCII str rather than returning
        # False. A cookie name is attacker-influenceable (a sibling
        # subdomain, a network position on plain http), and filtering here
        # means every caller inherits the guard instead of each needing
        # its own.
        and name[len(prefix) :].isascii()
    ]


def _set_install_nonce(response: Response, request: Request, provider: str, nonce: str) -> None:
    response.set_cookie(
        _install_nonce_cookie_name(provider, nonce),
        "1",
        max_age=DEFAULT_STATE_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        # Not hardcoded: a local http deployment would silently never receive
        # the cookie back, turning every connect into an invalid-state error.
        secure=request.url.scheme == "https",
        path=_INSTALL_NONCE_PATH,
    )


def _install_nonce_matches(request: Request, provider: str, nonce: str) -> bool:
    if not nonce:
        return False
    return any(
        hmac.compare_digest(presented, nonce)
        for presented in _pending_install_nonces(request, provider)
    )


def _retire_install_nonce(
    response: Response, request: Request, provider: str, nonce: str
) -> Response:
    """Drop one finished install, leaving the tabs still waiting untouched.

    Only ever called for a nonce that matched. A callback that did not match
    leaves every cookie alone: clearing one there would let one forged or
    stale redirect cancel an install the user still has open.
    """
    response.delete_cookie(_install_nonce_cookie_name(provider, nonce), path=_INSTALL_NONCE_PATH)
    return response


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
    # Aggregate status (fixed /status path — before the {provider} routes)
    # ------------------------------------------------------------------ #

    @integrations_router.get("/status")
    async def integrations_status(
        user: User = Depends(get_authenticated_user),
    ) -> IntegrationsStatusDTO:
        """Aggregate connection status: every OAuth provider + every known plugin.

        One call powers the whole integrations page. Every registered
        provider and every known plugin appears, connected or not, with
        display fields only — never token material. Each status source
        (credentials, identity plugins, legacy prefixes, agent registry) is
        fetched independently and degrades to its empty default on failure:
        a broken source logs server-side and blanks its section rather than
        500ing the page (same posture as the sessions list).
        """
        with new_span("cognee.integrations.status"):
            credentials = {}
            try:
                credentials = await list_active_credentials_for_user(user.id)
            except Exception:
                logger.exception(
                    "integrations status: credential lookup failed for user %s", user.id
                )

            integrations = []
            for provider in sorted(supported_integrations):
                credential = credentials.get(provider)
                integrations.append(
                    IntegrationStatusItemDTO(
                        provider=provider,
                        connected=credential is not None,
                        account_label=credential.account_label if credential else None,
                        provider_account_id=credential.provider_account_id if credential else None,
                        connected_at=as_utc(credential.created_at) if credential else None,
                        sync_status=credential.sync_status if credential else None,
                        last_synced_at=as_utc(credential.last_synced_at) if credential else None,
                        sync_counts=(getattr(credential, "provider_metadata", None) or {}).get(
                            "last_sync_counts"
                        )
                        if credential
                        else None,
                    )
                )

            plugin_rows: dict[str, PluginStatusRow] = {}
            try:
                plugin_rows = await identity_plugin_statuses(user.id)
            except Exception:
                logger.exception(
                    "integrations status: identity plugin lookup failed for user %s", user.id
                )

            visible_ids = [user.id]
            try:
                visible_ids = await visible_user_ids(user)
            except Exception:
                logger.exception(
                    "integrations status: visible-user lookup failed for user %s", user.id
                )

            try:
                legacy_rows = await legacy_plugin_statuses(
                    visible_ids, exclude_keys=set(plugin_rows)
                )
                plugin_rows = merge_plugin_statuses(plugin_rows, legacy_rows)
            except Exception:
                logger.exception(
                    "integrations status: legacy session lookup failed for user %s", user.id
                )

            try:
                # Takes the caller's id, not visible_ids: the registry source
                # resolves child agents itself so it can distrust their
                # agent-writable connection blobs (see registry_plugin_statuses).
                registry_rows = await registry_plugin_statuses(user.id)
                plugin_rows = merge_plugin_statuses(plugin_rows, registry_rows)
            except Exception:
                logger.exception(
                    "integrations status: agent registry lookup failed for user %s", user.id
                )

            plugins = []
            for plugin_key in KNOWN_PLUGINS:
                row = plugin_rows.get(plugin_key) or PluginStatusRow(key=plugin_key)
                plugins.append(
                    PluginStatusItemDTO(
                        key=plugin_key,
                        connected=row.connected,
                        agent_id=row.agent_id,
                        provisioned_at=row.provisioned_at,
                        last_active_at=row.last_active_at,
                        session_count=row.session_count,
                        source=row.source,
                    )
                )

            return IntegrationsStatusDTO(integrations=integrations, plugins=plugins)

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

        ## Path Parameters
        - **plugin_key** (str): Key of a known plugin (see GET /api/v1/integrations/status).
        """
        with new_span("cognee.integrations.plugins.provision") as span:
            span.set_attribute("cognee.integrations.plugin", plugin_key)
            _plugin_or_404(plugin_key)

            created = True
            try:
                agent_user, api_key = await create_agent(plugin_key, user)
            except (UserAlreadyExists, IntegrityError):
                # Already provisioned — same agent, new key (rotation).
                # IntegrityError covers the concurrent-first-provision race:
                # fastapi-users' create is check-then-insert, so two
                # simultaneous calls can both pass the email check and the
                # loser hits the unique-email constraint instead of
                # UserAlreadyExists. Same recovery: resolve and rotate.
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
                # The parent's tenant, not the agent object's: create_agent
                # sets the agent's tenant_id via a raw UPDATE without
                # refreshing the returned ORM instance, so on first provision
                # agent_user.tenant_id still reads the stale pre-update None.
                # The agent shares the parent's tenant by construction.
                tenant_id=getattr(user, "tenant_id", None),
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

        ## Path Parameters
        - **plugin_key** (str): Key of a known plugin (see GET /api/v1/integrations/status).
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
        provider: str,
        request: Request,
        response: Response,
        user: User = Depends(get_authenticated_user),
    ) -> AuthorizeUrlDTO:
        """Mint the provider's authorize URL for the requesting user.

        Also sets the nonce cookie the callback checks. The state alone says
        which user *started* an install; the cookie is what says the browser
        finishing it is the same one. See :func:`_set_install_nonce`.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).
        """
        with new_span("cognee.integrations.authorize") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)
            try:
                nonce = secrets.token_urlsafe(32)
                state = make_state(
                    user.id, nonce, signing_secret=integration.state_signing_secret()
                )
                authorize_url = integration.authorize_url(state)
                _set_install_nonce(response, request, provider, nonce)
                return AuthorizeUrlDTO(authorize_url=authorize_url)
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
    async def callback(
        provider: str, request: Request, code: str = "", state: str = "", error: str = ""
    ):
        """OAuth redirect target — state-authenticated, browser-facing."""
        with new_span("cognee.integrations.callback") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)

            if error:
                # The user clicked "Cancel" on the provider's consent screen
                # (or the provider rejected the request) — not a fault, just
                # an aborted install.
                span.set_attribute("cognee.integrations.outcome", "cancelled")
                # The cookie is left as it is: this path never learned which
                # nonce the aborted install held, and the user's other tabs
                # are still waiting on theirs. Unused nonces expire with the
                # cookie's own max-age.
                return _frontend_redirect(integration, "cancelled")

            fields = validate_state(
                state, signing_secret=integration.state_signing_secret(), field_count=2
            )
            if not isinstance(fields, tuple):
                span.set_attribute("cognee.integrations.outcome", "error_invalid_state")
                return _frontend_redirect(integration, "error_invalid_state")

            user_id_field, nonce = fields
            if not _install_nonce_matches(request, provider, nonce):
                # The state is authentic but this is not the browser that
                # asked for it. Without this check anyone with a cognee
                # account could mint an authorize URL, hand it to someone
                # else, and have that person's provider account land on the
                # attacker's cognee user. See _set_install_nonce.
                logger.warning(
                    "%s callback arrived without a matching install nonce; refusing", provider
                )
                span.set_attribute("cognee.integrations.outcome", "error_invalid_state")
                # Nothing is retired here. This callback never proved it owns
                # any pending install, so letting it empty the cookie would
                # hand anyone a way to cancel every install the user has open.
                return _frontend_redirect(integration, "error_invalid_state")

            try:
                user_id = UUID(user_id_field)
            except ValueError:
                span.set_attribute("cognee.integrations.outcome", "error_invalid_state")
                return _retire_install_nonce(
                    _frontend_redirect(integration, "error_invalid_state"), request, provider, nonce
                )

            try:
                credential = await complete_installation(
                    integration,
                    code=code,
                    user_id=user_id,
                    # The full query string, for providers whose redirect
                    # carries more than a code (GitHub's installation_id).
                    # The adapter decides what in here it trusts.
                    callback_params=dict(request.query_params),
                )
            except CrossUserConflictError:
                # The account is already connected to another user; refuse
                # rather than silently reassign it (see upsert_credential).
                logger.warning("%s account already connected elsewhere; user %s", provider, user_id)
                span.set_attribute("cognee.integrations.outcome", "error_already_connected")
                return _retire_install_nonce(
                    _frontend_redirect(integration, "error_already_connected"),
                    request,
                    provider,
                    nonce,
                )
            except Exception:  # any exchange/parse failure must redirect, not 500
                # Full trace server-side; the browser only learns that it failed.
                logger.exception("%s OAuth exchange failed for user %s", provider, user_id)
                span.set_attribute("cognee.integrations.outcome", "error_exchange_failed")
                return _retire_install_nonce(
                    _frontend_redirect(integration, "error_exchange_failed"),
                    request,
                    provider,
                    nonce,
                )

            logger.info(
                "%s account %s connected to user %s",
                provider,
                credential.provider_account_id,
                user_id,
            )
            # Post-install work (e.g. GitHub's initial repo sync) runs
            # detached — the browser gets its redirect now, not after.
            _spawn_background(
                integration.on_installed(credential),
                description=f"{provider} on_installed hook",
            )
            span.set_attribute("cognee.integrations.outcome", "connected")
            return _retire_install_nonce(
                _frontend_redirect(integration, "connected"), request, provider, nonce
            )

    @integrations_router.post("/{provider}/events", include_in_schema=False)
    async def provider_events(provider: str, request: Request):
        """Generic webhook receiver, one URL per provider.

        Unauthenticated by design — providers can't send a bearer token, so
        the provider's own ``WebhookVerifier`` (HMAC over the raw bytes) is
        the entire auth model, exactly like the Slack routes. A provider
        that registers no verifier simply doesn't accept webhooks: 404, the
        same answer an unknown provider gets, so the route leaks nothing
        about which providers are configured.

        The delivery is acked as soon as the signature checks out; the
        actual handling (which may clone repositories or run pipelines) runs
        detached so the provider's delivery timeout is never in play.
        """
        with new_span("cognee.integrations.events") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)
            verifier = integration.webhook_verifier()
            if verifier is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"{provider} does not accept webhook deliveries.",
                )

            raw_body = await verifier.verify(request)
            headers = {key.lower(): value for key, value in request.headers.items()}
            _spawn_background(
                integration.handle_webhook(raw_body, headers),
                description=f"{provider} webhook handling",
            )
            return {"ok": True}

    @integrations_router.get("/{provider}/connection", response_model_exclude_none=True)
    async def connection_status(
        provider: str, user: User = Depends(get_authenticated_user)
    ) -> ConnectionStatusDTO:
        """Connection state for the Integrations page.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).
        """
        integration = _integration_or_404(provider)
        credential = await get_active_credential_for_user(user.id, integration.provider)
        if credential is None:
            return ConnectionStatusDTO(connected=False)

        # Token material stays server-side; the frontend only needs display state.
        return ConnectionStatusDTO(
            connected=True,
            account_label=credential.account_label,
            provider_account_id=credential.provider_account_id,
            connected_at=as_utc(credential.created_at),
            sync_status=credential.sync_status,
            last_synced_at=as_utc(credential.last_synced_at),
            sync_counts=(getattr(credential, "provider_metadata", None) or {}).get(
                "last_sync_counts"
            ),
        )

    @integrations_router.get("/{provider}/resources")
    @integrations_router.get("/{provider}/folders")
    @integrations_router.get("/{provider}/labels")
    async def integration_resources(
        provider: str, user: User = Depends(get_authenticated_user)
    ) -> IntegrationResourceListDTO:
        """List selectable resources for a connected integration.

        ``/folders`` and ``/labels`` are readable aliases for SDK callers that
        want provider vocabulary; ``/resources`` is the stable generic path.
        The returned selection is three-state: ``null`` means all resources,
        an empty list means none, and a non-empty list is an allowlist.
        """
        integration = _integration_or_404(provider)
        if integration.resource_selection_key is None:
            raise HTTPException(status_code=404, detail=f"{provider} has no selectable resources")
        credential = await get_active_credential_for_user(user.id, integration.provider)
        if credential is None:
            raise HTTPException(status_code=404, detail=f"{provider} is not connected")
        try:
            resources = await integration.list_resources(credential)
        except Exception:
            logger.exception("Could not list %s resources for user %s", provider, user.id)
            raise HTTPException(status_code=502, detail=f"Could not list {provider} resources")
        if resources is None:
            raise HTTPException(status_code=404, detail=f"{provider} has no selectable resources")

        metadata = credential.provider_metadata or {}
        raw_selected = metadata.get(integration.resource_selection_key)
        selected = (
            None
            if raw_selected is None
            else [str(resource_id) for resource_id in raw_selected]
            if isinstance(raw_selected, list)
            else None
        )
        selected_set = set(selected or []) if selected is not None else None
        return IntegrationResourceListDTO(
            resources=[
                IntegrationResourceDTO(
                    id=str(resource["id"]),
                    name=str(resource.get("name") or resource["id"]),
                    description=resource.get("description"),
                    attributes=resource.get("attributes") or {},
                    selected=selected_set is None or str(resource["id"]) in selected_set,
                )
                for resource in resources
            ],
            selected=selected,
        )

    @integrations_router.put("/{provider}/resources")
    @integrations_router.put("/{provider}/folders")
    @integrations_router.put("/{provider}/labels")
    async def set_integration_resources(
        provider: str,
        payload: IntegrationResourceSelectionPayload,
        user: User = Depends(get_authenticated_user),
    ) -> IntegrationResourceSelectionResultDTO:
        """Persist a full resource selection for a connected integration."""
        integration = _integration_or_404(provider)
        selection_key = integration.resource_selection_key
        if selection_key is None:
            raise HTTPException(status_code=404, detail=f"{provider} has no selectable resources")
        credential = await get_active_credential_for_user(user.id, integration.provider)
        if credential is None:
            raise HTTPException(status_code=404, detail=f"{provider} is not connected")
        resource_ids = None
        if payload.resource_ids is not None:
            resource_ids = list(
                dict.fromkeys(str(resource_id) for resource_id in payload.resource_ids)
            )
        updated = await update_provider_metadata(
            provider,
            credential.provider_account_id,
            {selection_key: resource_ids},
        )
        return IntegrationResourceSelectionResultDTO(
            selected=(updated.provider_metadata or {}).get(selection_key)
        )

    @integrations_router.post("/{provider}/sync")
    async def sync_integration(
        provider: str, user: User = Depends(get_authenticated_user)
    ) -> IntegrationSyncResultDTO:
        """Start a provider sync without changing its stored selection."""
        integration = _integration_or_404(provider)
        credential = await get_active_credential_for_user(user.id, integration.provider)
        if credential is None:
            raise HTTPException(status_code=404, detail=f"{provider} is not connected")
        _spawn_background(integration.sync_now(credential), description=f"{provider} manual sync")
        return IntegrationSyncResultDTO(accepted=True)

    @integrations_router.delete("/{provider}/connection")
    async def disconnect(
        provider: str,
        delete_data: bool = False,
        user: User = Depends(get_authenticated_user),
    ) -> DisconnectResultDTO:
        """Disconnect the account connected by the requesting user.

        Marks the stored installation revoked, and best-effort asks the
        provider to kill the token on its own side via
        ``integration.revoke_remote``. That call is wrapped here too, on top
        of each adapter's own best-effort handling — a third-party
        integration that doesn't honor the "never raise" contract on
        ``revoke_remote`` still must not block the local disconnect.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).
        """
        with new_span("cognee.integrations.disconnect") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)
            credential = await get_active_credential_for_user(user.id, integration.provider)
            if credential is None or credential.provider_account_id is None:
                return DisconnectResultDTO(disconnected=False)

            dataset_name = None
            if delete_data:
                dataset_name = integration.dataset_name(credential)
                if not dataset_name:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{provider} does not expose a deletable dataset",
                    )

            try:
                await integration.revoke_remote(credential)
            except Exception:  # a remote-revoke failure must never block disconnect
                logger.exception(
                    "%s revoke_remote raised for account %s",
                    provider,
                    credential.provider_account_id,
                )

            await revoke_credential_by_account(integration.provider, credential.provider_account_id)
            if delete_data:
                try:
                    from cognee.api.v1.forget.forget import forget

                    await forget(dataset=dataset_name, user=user)
                except HTTPException:
                    raise
                except Exception:
                    logger.exception(
                        "Could not delete %s dataset %s during disconnect",
                        provider,
                        dataset_name,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"{provider} disconnected, but its data could not be deleted",
                    )
            return DisconnectResultDTO(disconnected=True)

    return integrations_router
