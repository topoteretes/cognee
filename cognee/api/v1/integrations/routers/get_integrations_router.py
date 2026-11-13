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
agent sub-user, holding one labeled API key per machine the plugin runs on, so
status, attribution, and revocation key off that identity instead of a shared
tenant key. Declared before the
generic ``{provider}`` routes so ``plugins`` is never captured as a
provider name.
"""

import asyncio
import hmac
import logging
import secrets
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi_users.exceptions import UserAlreadyExists
from pydantic import Field
from sqlalchemy.exc import IntegrityError

from cognee.api.DTO import InDTO, OutDTO
from cognee.api.v1.integrations.routers.get_sync_progress_router import get_sync_progress_router
from cognee.modules.agents.create_agent import create_agent
from cognee.modules.agents.list_agents import list_agents
from cognee.modules.agents.registry import (
    AGENT_CONFIG_NAME,
    build_agent_connection_id,
    deactivate_agent_connection,
    register_agent_connection,
)
from cognee.modules.integrations.background import spawn_background
from cognee.modules.integrations.base import (
    OAuthIntegration,
    ResourceParentError,
    ResourceParentNotFoundError,
)
from cognee.modules.integrations.connect import complete_installation
from cognee.modules.integrations.credentials import (
    REVOKED_BY_OWNER,
    CredentialInactiveError,
    CrossUserConflictError,
    change_provider_metadata,
    get_active_credential_for_user,
    get_ended_connection,
    list_active_credentials_for_provider,
    list_active_credentials_for_user,
    owner_principal_id,
    revoke_credential_by_account,
)
from cognee.modules.integrations.crypto import oauth_state_signing_key
from cognee.modules.integrations.linear.resume_sync import linear_resume_lifespan
from cognee.modules.integrations.oauth_flow import (
    DEFAULT_STATE_TTL_SECONDS,
    make_state,
    validate_state,
)
from cognee.modules.integrations.outbound import (
    OutboundHttpDisabledError,
    require_outbound_http,
)
from cognee.modules.integrations.plugin_keys import (
    PluginAgentKeyLimitError,
    add_plugin_agent_key,
    revoke_plugin_agent_keys,
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
from cognee.modules.integrations.scheduled_sync import sync_scheduler_lifespan
from cognee.modules.integrations.sync_policy import (
    CUSTOM_HOURS,
    INTERVAL_PRESETS,
    SyncPolicy,
    delete_sync_policy,
    get_caller_effective_sync_policy,
    get_sync_pause_level,
    get_sync_policy,
    pause_sync_policy,
    policy_level,
    require_sync_not_paused,
    resume_sync_policy,
    save_sync_policy,
)
from cognee.modules.integrations.sync_policy.models import IntegrationSyncPolicy
from cognee.modules.observability import new_span
from cognee.modules.session_lifecycle.visibility import visible_user_ids
from cognee.modules.users.api_key.delete_api_key import delete_api_key
from cognee.modules.users.api_key.exceptions import ApiKeyNotFoundError
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


def _as_set(selection: list[str] | None) -> frozenset[str] | None:
    """A selection compared without its order; ``None`` (everything) stays ``None``."""
    return None if selection is None else frozenset(selection)


# How long a disconnect waits for its staging copy to be dropped before it answers and lets
# the drop finish in the background. The drop takes the lock every dlt run in the process
# holds, which another connection's backfill can keep for minutes; a drop that fails within
# the wait is reported to the owner.
STAGING_DROP_WAIT_SECONDS = 10


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
    dataset_id: str | None = None
    stored_items: int | None = None
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
    # Set only when ``connected`` is false for a connection the user had: "owner" (a
    # disconnect) or "provider" (an uninstall, a rejected token), absent when unknown.
    # Its data stays in ``dataset_id``.
    disconnected_by: str | None = None
    disconnected_at: datetime | None = None
    # When the data of an ended connection is deleted (Slack: 14 days after an uninstall);
    # absent when it is kept.
    data_deleted_after: datetime | None = None
    # When that deletion ran; set, the data is gone.
    data_deleted_at: datetime | None = None


class DisconnectResultDTO(OutDTO):
    disconnected: bool


class ConnectionDisconnectResultDTO(DisconnectResultDTO):
    # True when ALLOW_HTTP_REQUESTS=false kept cognee from asking the provider to
    # revoke the token: the local connection is gone, the token is still live at
    # the provider until it expires or the user removes the app there.
    remote_revoke_skipped: bool = False


class ConnectionSummaryDTO(OutDTO):
    provider_account_id: str
    account_label: str | None = None
    connected_at: datetime | None = None


class ConnectionListDTO(OutDTO):
    connections: list[ConnectionSummaryDTO]


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


class SyncPolicyDTO(OutDTO):
    """A saved sync policy, with the intervals a UI may offer, so it hard-codes none.

    ``level`` is ``default`` for the owner's default and ``connection`` for one
    connection's own policy. On a connection's routes ``held_by`` names the level that
    holds it paused: a paused default holds it even when its own row is not. Times are UTC.
    """

    level: Literal["default", "connection"]
    interval_minutes: int
    weekdays: list[int]
    timezone: str
    paused: bool
    paused_by: UUID | None
    paused_at: datetime | None
    next_due_at: datetime | None
    held_by: Literal["default", "connection"] | None = None
    interval_presets: list[int] = list(INTERVAL_PRESETS)
    custom_hours: list[int] = list(CUSTOM_HOURS)


def _sync_policy_dto(row: IntegrationSyncPolicy, held_by: str | None = None) -> SyncPolicyDTO:
    return SyncPolicyDTO(
        level=policy_level(row),
        held_by=held_by,
        interval_minutes=row.interval_minutes,
        weekdays=row.weekdays,
        timezone=row.timezone,
        paused=row.paused,
        paused_by=row.paused_by,
        # SQLite returns the stored UTC values without their offset.
        paused_at=as_utc(row.paused_at),
        next_due_at=as_utc(row.next_due_at),
    )


async def _connection_policy_dto(row: IntegrationSyncPolicy, credential) -> SyncPolicyDTO:
    held_by = await get_sync_pause_level(
        credential_id=credential.id, principal_id=owner_principal_id(credential)
    )
    return _sync_policy_dto(row, held_by=held_by)


class PluginProvisionDTO(OutDTO):
    plugin_key: str
    agent_id: UUID
    api_key: str
    key_id: UUID
    created: bool


class PluginKeyDTO(OutDTO):
    id: UUID
    name: str | None = None
    label: str | None = None
    created_at: datetime | None = None
    last_used_at: datetime | None = None


class PluginKeyListDTO(OutDTO):
    keys: list[PluginKeyDTO]


class PluginKeyRevokeResultDTO(OutDTO):
    revoked: bool


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


def _outbound_http_or_503(provider: str) -> None:
    """Refuse a connect or sync while ``ALLOW_HTTP_REQUESTS=false``.

    Every provider call is already refused at its ``client_session``; here
    the check answers the caller directly with a 503, instead of a sync
    accepted into the background that fails where nobody sees it, or an
    OAuth round trip whose callback cannot complete.
    """
    try:
        require_outbound_http(provider)
    except OutboundHttpDisabledError as error:
        raise HTTPException(status_code=503, detail=error.message) from error


def _integration_or_404(provider: str) -> OAuthIntegration:
    try:
        return get_integration(provider)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown integration provider: {provider!r}")


async def _credential_or_404(user: User, integration: OAuthIntegration, account_id: str | None):
    """The user's newest active connection, or the one ``account_id`` names.

    ``None`` means the user has no connection at all. A named account that is not the
    user's (unknown, disconnected, or someone else's) is a 404 and never falls back to
    another account: the caller asked for that one. A blank id names nothing and is a 422.
    """
    _require_non_blank(account_id)
    credential = await get_active_credential_for_user(user.id, integration.provider, account_id)
    if credential is None and account_id is not None:
        raise _account_not_connected(integration)
    return credential


def _require_non_blank(account_id: str | None) -> None:
    if account_id is not None and not account_id.strip():
        raise HTTPException(status_code=422, detail="account_id must not be blank")


def _account_not_connected(integration: OAuthIntegration) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{integration.provider} account is not connected")


async def _connected_credential_or_404(
    user: User, integration: OAuthIntegration, account_id: str | None
):
    """``_credential_or_404`` for routes that act on a connection: none at all is a 404 too."""
    credential = await _credential_or_404(user, integration, account_id)
    if credential is None:
        raise HTTPException(status_code=404, detail=f"{integration.provider} is not connected")
    return credential


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
    ``provisioned_at`` sticks across re-provisions (another machine's key isn't a new
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


async def _add_plugin_key(agent_user_id: UUID, plugin_key: str):
    """Add a key for this machine; a full agent is a 409, never an eviction."""
    try:
        return await add_plugin_agent_key(agent_user_id, plugin_key)
    except PluginAgentKeyLimitError as error:
        raise HTTPException(
            status_code=409,
            detail=(
                f"The {plugin_key!r} agent already holds the maximum of {error.limit} API "
                "keys. Revoke an old machine's key (GET and DELETE "
                f"/api/v1/integrations/plugins/{plugin_key}/keys) and try again."
            ),
        )


def _reject_agent_caller(user: User) -> None:
    """The key routes manage an agent's keys on behalf of its human owner only."""
    if user.parent_user_id is not None:
        raise HTTPException(status_code=403, detail="An agent cannot manage plugin keys.")


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
    safe_provider = quote(provider, safe="")
    return f"{_INSTALL_NONCE_COOKIE_PREFIX}{safe_provider}_{nonce}"


def _install_nonce_cookie_prefix(provider: str) -> str:
    safe_provider = quote(provider, safe="")
    return f"{_INSTALL_NONCE_COOKIE_PREFIX}{safe_provider}_"


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


def _start_authorization(provider: str, user: User) -> tuple[str, str]:
    """The provider's authorize URL for ``user`` and the nonce its state carries; the
    caller sets the nonce cookie on its response: the state says which user started it,
    the cookie that the browser finishing it is the same one (:func:`_set_install_nonce`)."""
    integration = _integration_or_404(provider)
    _outbound_http_or_503(provider)
    try:
        nonce = secrets.token_urlsafe(32)
        state = make_state(user.id, nonce, signing_secret=oauth_state_signing_key())
        return integration.authorize_url(state), nonce
    except RuntimeError:
        # A provider's require()-style settings guard raises when its client id/secret/
        # signing key aren't configured — a deploy-time gap, not a per-request fault.
        # Surface it as a clear 503 instead of a bare 500.
        logger.exception("%s authorize requested but is not configured", provider)
        raise HTTPException(
            status_code=503,
            detail=f"{provider} integration is not configured on this server.",
        )


async def _drop_staging_copy(provider: str, dataset_name: str, drop_dlt_dataset) -> None:
    """Drop the staging copy, waiting at most ``STAGING_DROP_WAIT_SECONDS``.

    A failure within the wait raises, so the owner is told. The drop is background work from
    the start, so whatever ends the wait (the timeout, a cancelled request) it still finishes,
    the shutdown drains it, and a failure is logged.
    """
    drop = asyncio.ensure_future(drop_dlt_dataset(dataset_name))
    spawn_background(_finish(drop), description=f"{provider} staging copy drop on disconnect")
    try:
        await asyncio.wait_for(asyncio.shield(drop), timeout=STAGING_DROP_WAIT_SECONDS)
    except asyncio.TimeoutError:
        logger.info("%s staging copy drop continues in the background", provider)


async def _finish(task: asyncio.Future) -> None:
    await task


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
    integrations_router = APIRouter(lifespan=linear_resume_lifespan)
    integrations_router.include_router(get_sync_progress_router())
    # A router holds one lifespan, and an included router's lifespan runs with its parent's.
    integrations_router.include_router(APIRouter(lifespan=sync_scheduler_lifespan))

    # ------------------------------------------------------------------ #
    # Sync policies: the caller's own default (fixed /sync/policy path, before the
    # {provider} routes). Scoped by construction: no route here takes an owner.
    # ------------------------------------------------------------------ #

    @integrations_router.get("/sync/policy")
    async def get_default_sync_policy(
        user: User = Depends(get_authenticated_user),
    ) -> SyncPolicyDTO:
        """The caller's default sync policy, which every connection without its own inherits."""
        return _sync_policy_dto(await get_sync_policy(user, user.id))

    @integrations_router.put("/sync/policy")
    async def set_default_sync_policy(
        policy: SyncPolicy, user: User = Depends(get_authenticated_user)
    ) -> SyncPolicyDTO:
        """Create or replace the caller's default sync policy. A pause is kept.

        ## Request Parameters
        - **intervalMinutes** (int): Minutes between syncs for the default policy: one of the
          presets 15, 30, 60, 120, 240, 360, 480, 1440 (daily) or 10080 (weekly), or whole
          hours from 1 to 23 (60 to 1380); any other value is rejected with 400. Daily and
          weekly run once, at 10:00 local; shorter intervals repeat from local midnight.
          Saving recomputes the next due slot.
        - **timezone** (str): IANA timezone name (for example Europe/Belgrade) in which the
          weekdays and slot times are interpreted; a name not present in the tz database is
          rejected with 400. Defaults to 'UTC'.
        - **weekdays** (Tuple[int, Ellipsis]): Local weekdays on which syncing may run, 0 is
          Monday through 6 is Sunday; duplicates are dropped and the days stored sorted. An
          empty list or a day outside 0-6 is rejected with 400, and a weekly interval takes
          exactly one day. Defaults to (0, 1, 2, 3, 4, 5, 6).
        """
        return _sync_policy_dto(await save_sync_policy(user, user.id, policy))

    @integrations_router.delete("/sync/policy", status_code=204)
    async def remove_default_sync_policy(user: User = Depends(get_authenticated_user)) -> None:
        """Remove the caller's default: connections without their own stop syncing on schedule."""
        await delete_sync_policy(user, user.id)

    @integrations_router.post("/sync/policy/pause")
    async def pause_default_sync_policy(
        user: User = Depends(get_authenticated_user),
    ) -> SyncPolicyDTO:
        """Pause every connection of the caller, those with their own policy too."""
        return _sync_policy_dto(await pause_sync_policy(user, user.id))

    @integrations_router.post("/sync/policy/resume")
    async def resume_default_sync_policy(
        user: User = Depends(get_authenticated_user),
    ) -> SyncPolicyDTO:
        """Resume the caller's default; its next slot is the first one after now."""
        return _sync_policy_dto(await resume_sync_policy(user, user.id))

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

    @integrations_router.post(
        "/plugins/{plugin_key}/provision",
        # Capability marker: plugins read it from /openapi.json to learn that
        # provisioning here only adds keys, so it is safe to call on any machine.
        openapi_extra={"x-cognee-additive-plugin-keys": True},
    )
    async def provision_plugin(
        plugin_key: str, user: User = Depends(get_authenticated_user)
    ) -> PluginProvisionDTO:
        """Provision a dedicated agent identity for a plugin, and add a key for it.

        Idempotent get-or-create: the first call creates an agent sub-user
        for ``(user, plugin_key)``; every later call returns the same agent.
        Every call adds one new key, so each machine holds its own. The
        returned key is shown once and never retrievable again.

        Provisioning never deletes a key. An agent already holding
        ``MAX_USER_API_KEYS`` keys (default 10) answers 409 and nothing changes:
        free a slot through ``DELETE /plugins/{plugin_key}/keys/{key_id}``.
        Provisioning does not revoke a leaked key either: revoke it the same
        way, or every key through ``DELETE /plugins/{plugin_key}``.

        ## Path Parameters
        - **plugin_key** (str): Key of a known plugin (see GET /api/v1/integrations/status).
        """
        with new_span("cognee.integrations.plugins.provision") as span:
            span.set_attribute("cognee.integrations.plugin", plugin_key)
            _plugin_or_404(plugin_key)

            created = True
            try:
                # Every key goes through add_plugin_agent_key below.
                agent_user, _ = await create_agent(plugin_key, user, issue_api_key=False)
            except (UserAlreadyExists, IntegrityError):
                # Already provisioned — same agent, one more key.
                # IntegrityError covers the concurrent-first-provision race:
                # fastapi-users' create is check-then-insert, so two
                # simultaneous calls can both pass the email check and the
                # loser hits the unique-email constraint instead of
                # UserAlreadyExists. Same recovery: resolve the agent and add.
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

            new_key = await _add_plugin_key(agent_user.id, plugin_key)

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

            span.set_attribute("cognee.integrations.outcome", "created" if created else "added")
            return PluginProvisionDTO(
                plugin_key=plugin_key,
                agent_id=agent_user.id,
                api_key=new_key.api_key,
                key_id=new_key.id,
                created=created,
            )

    @integrations_router.get("/plugins/{plugin_key}/keys")
    async def list_plugin_keys(
        plugin_key: str, user: User = Depends(get_authenticated_user)
    ) -> PluginKeyListDTO:
        """List the API keys of the caller's agent for a plugin — one per machine.

        Metadata only (id, name, label, created, last used); the key
        itself is never returned. Empty when the plugin was never provisioned.
        An agent caller is a 403: only the agent's owner manages its keys.

        ## Path Parameters
        - **plugin_key** (str): Key of a known plugin (see GET /api/v1/integrations/status).
        """
        _plugin_or_404(plugin_key)
        _reject_agent_caller(user)
        agent_user = await _find_plugin_agent(user, plugin_key)
        if agent_user is None:
            return PluginKeyListDTO(keys=[])
        return PluginKeyListDTO(
            keys=[
                PluginKeyDTO(
                    id=key.id,
                    name=key.name,
                    label=key.label,
                    created_at=key.created_at,
                    last_used_at=key.last_used_at,
                )
                for key in await get_api_keys(agent_user)
            ]
        )

    @integrations_router.delete("/plugins/{plugin_key}/keys/{key_id}")
    async def revoke_plugin_key(
        plugin_key: str, key_id: UUID, user: User = Depends(get_authenticated_user)
    ) -> PluginKeyRevokeResultDTO:
        """Revoke one key of the caller's agent for a plugin (e.g. a retired machine).

        Scoped to that plugin's agent of the authenticated user: a key of any
        other principal is a 404, exactly like an unknown id. Other machines'
        keys stay valid; ``DELETE /plugins/{plugin_key}`` revokes them all. An
        agent caller is a 403.

        ## Path Parameters
        - **plugin_key** (str): Key of a known plugin (see GET /api/v1/integrations/status).
        - **key_id** (UUID): Id of the key (see GET /plugins/{plugin_key}/keys).
        """
        with new_span("cognee.integrations.plugins.revoke_key") as span:
            span.set_attribute("cognee.integrations.plugin", plugin_key)
            _plugin_or_404(plugin_key)
            _reject_agent_caller(user)
            agent_user = await _find_plugin_agent(user, plugin_key)
            if agent_user is None:
                raise HTTPException(status_code=404, detail="API key not found")
            try:
                await delete_api_key(agent_user, key_id)
            except ApiKeyNotFoundError:
                raise HTTPException(status_code=404, detail="API key not found")
            return PluginKeyRevokeResultDTO(revoked=True)

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

            await revoke_plugin_agent_keys(agent_user.id)

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
            authorize_url, nonce = _start_authorization(provider, user)
            _set_install_nonce(response, request, provider, nonce)
            return AuthorizeUrlDTO(authorize_url=authorize_url)

    @integrations_router.get("/{provider}/start", include_in_schema=False)
    async def start(provider: str, request: Request, user: User = Depends(get_authenticated_user)):
        """``authorize`` for a link opened in the browser: the same nonce cookie, then a
        redirect to the provider's authorize page instead of the URL as JSON.

        A link a member follows from the provider, where no page of cognee's runs
        (Slack's ``@cognee link``), starts sign-in here, in the browser where they are
        logged in to cognee; the callback binds the account to that user as after
        ``authorize``. Not logged in, it answers 401.
        """
        with new_span("cognee.integrations.start") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            authorize_url, nonce = _start_authorization(provider, user)
            redirect = RedirectResponse(url=authorize_url, status_code=302)
            _set_install_nonce(redirect, request, provider, nonce)
            return redirect

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

            fields = validate_state(state, signing_secret=oauth_state_signing_key(), field_count=2)
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
            spawn_background(
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
            spawn_background(
                integration.handle_webhook(raw_body, headers),
                description=f"{provider} webhook handling",
            )
            return {"ok": True}

    @integrations_router.get("/{provider}/connections", response_model_exclude_none=True)
    async def list_connections(
        provider: str, user: User = Depends(get_authenticated_user)
    ) -> ConnectionListDTO:
        """The requesting user's active connections for a provider, newest first.

        A provider authorized per workspace (Notion, Slack) can hold several. The
        ``provider_account_id`` of one is what the other routes take as ``account_id``.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).
        """
        with new_span("cognee.integrations.connections") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)
            connections = await list_active_credentials_for_provider(user.id, integration.provider)
            return ConnectionListDTO(
                connections=[
                    ConnectionSummaryDTO(
                        provider_account_id=credential.provider_account_id,
                        account_label=credential.account_label,
                        connected_at=as_utc(credential.created_at),
                    )
                    for credential in connections
                ]
            )

    @integrations_router.get("/{provider}/connection", response_model_exclude_none=True)
    async def connection_status(
        provider: str,
        account_id: str | None = None,
        user: User = Depends(get_authenticated_user),
    ) -> ConnectionStatusDTO:
        """Connection state for the Integrations page.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).

        ## Query Parameters
        - **account_id** (str, optional): The ``provider_account_id`` of one of the
          user's connections (see GET /{provider}/connections). Without it the newest
          connection answers. An id that is not the user's is a 404, never another account.

        With no active connection, the newest one the user had (or the one named) answers
        ``connected: false`` with ``disconnectedBy`` (``"owner"`` for a disconnect,
        ``"provider"`` for an app uninstalled or a token it no longer accepts, absent when
        unknown), ``disconnectedAt`` and the dataset its data stays in; the data is deleted
        only from the datasets, or by the provider's own schedule, which ``dataDeletedAfter``
        (when it is due) and ``dataDeletedAt`` (when it ran) report (Slack: 14 days after an
        uninstall). A named account the user never had is still a 404.
        """
        from cognee.modules.integrations.ingestion import dataset_summary, sync_is_running

        integration = _integration_or_404(provider)
        _require_non_blank(account_id)
        credential = await get_active_credential_for_user(user.id, integration.provider, account_id)
        if credential is None:
            ended = await get_ended_connection(user.id, integration.provider, account_id)
            if ended is None:
                if account_id is not None:
                    raise _account_not_connected(integration)
                return ConnectionStatusDTO(connected=False)
            dataset_id, stored_items = None, None
            dataset_name = integration.dataset_name(ended)
            if dataset_name is not None:
                dataset_id, stored_items = await dataset_summary(ended, dataset_name)
            return ConnectionStatusDTO(
                connected=False,
                dataset_id=dataset_id,
                stored_items=stored_items,
                account_label=ended.account_label,
                provider_account_id=ended.provider_account_id,
                disconnected_by=ended.revoked_by,
                disconnected_at=as_utc(ended.revoked_at),
                data_deleted_after=as_utc(integration.data_deleted_after(ended)),
                data_deleted_at=as_utc(integration.data_deleted_at(ended)),
            )

        dataset_id, stored_items = None, None
        dataset_name = integration.dataset_name(credential)
        if dataset_name is not None:
            dataset_id, stored_items = await dataset_summary(credential, dataset_name)

        # Token material stays server-side; the frontend only needs display state.
        return ConnectionStatusDTO(
            connected=True,
            dataset_id=dataset_id,
            stored_items=stored_items,
            account_label=credential.account_label,
            provider_account_id=credential.provider_account_id,
            connected_at=as_utc(credential.created_at),
            sync_status=(
                "syncing"
                if sync_is_running(provider, credential.provider_account_id)
                else credential.sync_status
            ),
            last_synced_at=as_utc(credential.last_synced_at),
            sync_counts=(getattr(credential, "provider_metadata", None) or {}).get(
                "last_sync_counts"
            ),
        )

    @integrations_router.get("/{provider}/resources")
    @integrations_router.get("/{provider}/folders")
    @integrations_router.get("/{provider}/labels")
    async def integration_resources(
        provider: str,
        account_id: str | None = None,
        parent: str | None = None,
        user: User = Depends(get_authenticated_user),
    ) -> IntegrationResourceListDTO:
        """List selectable resources for a connected integration.

        ``/folders`` and ``/labels`` are readable aliases for SDK callers that
        want provider vocabulary; ``/resources`` is the stable generic path.
        The returned selection is three-state: ``null`` means all resources,
        an empty list means none, and a non-empty list is an allowlist.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).

        ## Query Parameters
        - **account_id** (Optional[str]): Connection to use (from GET
          /api/v1/integrations/{provider}/connections); the newest one is used when omitted.
          An id that is not the caller's returns 404 and never falls back to another
          connection; a blank id returns 422.
        - **parent** (Optional[str]): For a provider that lists its resources one level at a
          time (Dropbox folders), the resource whose children to list; omitted means the top
          level. A provider that lists everything at once, or a parent that is not a valid id,
          answers 422; a parent the provider cannot find answers 404.
        """
        integration = _integration_or_404(provider)
        _outbound_http_or_503(provider)
        if integration.resource_selection_key is None:
            raise HTTPException(status_code=404, detail=f"{provider} has no selectable resources")
        credential = await _credential_or_404(user, integration, account_id)
        if credential is None:
            raise HTTPException(status_code=404, detail=f"{provider} is not connected")
        from cognee.modules.integrations.ingestion import RateLimitedError

        try:
            resources = await integration.list_resources(credential, parent=parent)
        except ResourceParentNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from None
        except ResourceParentError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except RateLimitedError as error:
            # Never waited out inside the request: the client is told when to try again.
            raise HTTPException(
                status_code=429,
                detail=f"{provider} is limiting requests; try again shortly",
                headers={"Retry-After": str(error.retry_after)} if error.retry_after else None,
            )
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

    async def _validate_selection_or_http_error(integration, credential, resource_ids):
        """Ask the provider to check a selection, mapped like the listing route's errors.

        A refused selection is 422 with the provider's message. A connection that is no
        longer active is 404, as on the progress route; outbound HTTP switched off is 503;
        any other provider failure (a refused token, the network, a 5xx) is 502 as when
        listing resources, with a fixed message and the cause in the server log only.
        """
        provider = integration.provider
        try:
            await integration.validate_resource_selection(credential, resource_ids)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except CredentialInactiveError:
            raise HTTPException(status_code=404, detail=f"{provider} is not connected") from None
        except OutboundHttpDisabledError as error:
            raise HTTPException(status_code=503, detail=error.message) from None
        except Exception:
            logger.exception("Could not check the %s selection for its connection", provider)
            raise HTTPException(
                status_code=502, detail=f"Could not check the {provider} selection; try again"
            ) from None

    @integrations_router.put("/{provider}/resources")
    @integrations_router.put("/{provider}/folders")
    @integrations_router.put("/{provider}/labels")
    async def set_integration_resources(
        provider: str,
        payload: IntegrationResourceSelectionPayload,
        account_id: str | None = None,
        user: User = Depends(get_authenticated_user),
    ) -> IntegrationResourceSelectionResultDTO:
        """Persist a full resource selection for a connected integration.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).

        ## Query Parameters
        - **account_id** (Optional[str]): Connection to use (from GET
          /api/v1/integrations/{provider}/connections); the newest one is used when omitted.
          An id that is not the caller's returns 404 and never falls back to another
          connection; a blank id returns 422.

        ## Request Parameters
        - **resourceIds** (Optional[List[str]]): Full replacement list of selected resource
          ids; duplicates are dropped, blank or whitespace-only ids give 422. Null (also the
          value when omitted) means every resource and is a 422 for providers that do not
          allow it.

        A changed selection starts a sync at once, or one more run of the sync already
        running, so added resources arrive and removed ones leave memory without waiting
        for the schedule. An unchanged selection starts nothing.
        """
        integration = _integration_or_404(provider)
        selection_key = integration.resource_selection_key
        if selection_key is None:
            raise HTTPException(status_code=404, detail=f"{provider} has no selectable resources")
        credential = await _credential_or_404(user, integration, account_id)
        if credential is None:
            raise HTTPException(status_code=404, detail=f"{provider} is not connected")
        if payload.resource_ids is None and not integration.resource_selection_allows_all:
            raise HTTPException(
                status_code=422,
                detail=f"{provider} needs an explicit list of resource ids; null is not supported",
            )
        resource_ids = None
        if payload.resource_ids is not None:
            # A stored blank id fails every later sync of the providers that check the
            # shape, and the account stays degraded until the user saves again.
            if any(not resource_id.strip() for resource_id in payload.resource_ids):
                raise HTTPException(
                    status_code=422, detail="Resource ids must not be blank or whitespace only"
                )
            resource_ids = list(dict.fromkeys(payload.resource_ids))
            invalid = integration.invalid_resource_ids(resource_ids)
            if invalid:
                raise HTTPException(
                    status_code=422,
                    detail=f"Not a {provider} resource id: {', '.join(invalid)}",
                )
            await _validate_selection_or_http_error(integration, credential, resource_ids)
        changed = False

        def select(current: dict) -> dict:
            # Compared with what is stored under the connection's lock, not with the copy
            # read above: of two saves at once, the second must see the first's selection.
            nonlocal changed
            changed = _as_set(current.get(selection_key)) != _as_set(resource_ids)
            return {selection_key: resource_ids}

        updated = await change_provider_metadata(provider, credential.provider_account_id, select)
        if changed:
            from cognee.modules.integrations import ingestion

            # A paused connection keeps the new selection for its first sync after the
            # resume.
            paused = await ingestion.sync_is_paused(provider, updated, "selection change")
            if not paused and not ingestion.request_rerun_if_running(
                provider, credential.provider_account_id
            ):
                spawn_background(
                    integration.sync_now(updated),
                    description=f"{provider} sync after a selection change",
                )
        return IntegrationResourceSelectionResultDTO(
            selected=(updated.provider_metadata or {}).get(selection_key)
        )

    @integrations_router.post("/{provider}/sync")
    async def sync_integration(
        provider: str,
        account_id: str | None = None,
        user: User = Depends(get_authenticated_user),
    ) -> IntegrationSyncResultDTO:
        """Start a provider sync without changing its stored selection.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).

        ## Query Parameters
        - **account_id** (Optional[str]): Connection to use (from GET
          /api/v1/integrations/{provider}/connections); the newest one is used when omitted.
          An id that is not the caller's returns 404 and never falls back to another
          connection; a blank id returns 422.
        """
        integration = _integration_or_404(provider)
        _outbound_http_or_503(provider)
        credential = await _connected_credential_or_404(user, integration, account_id)
        await require_sync_not_paused(
            credential_id=credential.id, principal_id=owner_principal_id(credential)
        )
        from cognee.modules.integrations.ingestion import sync_is_running

        if sync_is_running(provider, credential.provider_account_id):
            # The running sync owns the cursor and a second request is dropped, so
            # saying "accepted" would be telling the user a sync started.
            return IntegrationSyncResultDTO(accepted=False)
        spawn_background(integration.sync_now(credential), description=f"{provider} manual sync")
        return IntegrationSyncResultDTO(accepted=True)

    # ------------------------------------------------------------------ #
    # Sync policy of one connection. The connection is found among the
    # caller's own, like every route above, and the owner is read off it, so
    # a connection is never paired with a principal that does not own it.
    # ------------------------------------------------------------------ #

    @integrations_router.get("/{provider}/sync/policy")
    async def get_connection_sync_policy(
        provider: str, account_id: str | None = None, user: User = Depends(get_authenticated_user)
    ) -> SyncPolicyDTO:
        """The policy the connection syncs by: its own, else its owner's default.

        ``level`` says which. No policy at either level is a 404.

        ## Query Parameters
        - **account_id** (str, optional): As for POST /{provider}/sync.
        """
        credential = await _connected_credential_or_404(
            user, _integration_or_404(provider), account_id
        )
        row, held_by = await get_caller_effective_sync_policy(
            user, owner_principal_id(credential), credential.id
        )
        return _sync_policy_dto(row, held_by=held_by)

    @integrations_router.put("/{provider}/sync/policy")
    async def set_connection_sync_policy(
        provider: str,
        policy: SyncPolicy,
        account_id: str | None = None,
        user: User = Depends(get_authenticated_user),
    ) -> SyncPolicyDTO:
        """Create or replace the connection's own policy, which overrides the default.

        ## Query Parameters
        - **account_id** (str, optional): As for POST /{provider}/sync.

        ## Request Parameters
        - **intervalMinutes** (int): Minutes between syncs for this connection, overriding
          the default policy: one of the presets 15, 30, 60, 120, 240, 360, 480, 1440 (daily)
          or 10080 (weekly), or whole hours from 1 to 23 (60 to 1380); any other value is
          rejected with 400. Daily and weekly run once, at 10:00 local; shorter intervals
          repeat from local midnight. Saving recomputes the next due slot.
        - **timezone** (str): IANA timezone name (for example Europe/Belgrade) in which this
          connection's weekdays and slot times are interpreted; a name not present in the tz
          database is rejected with 400. Defaults to 'UTC'.
        - **weekdays** (Tuple[int, Ellipsis]): Local weekdays on which this connection may
          sync, 0 is Monday through 6 is Sunday; duplicates are dropped and the days stored
          sorted. An empty list or a day outside 0-6 is rejected with 400, and a weekly
          interval takes exactly one day. Defaults to (0, 1, 2, 3, 4, 5, 6).
        """
        credential = await _connected_credential_or_404(
            user, _integration_or_404(provider), account_id
        )
        return await _connection_policy_dto(
            await save_sync_policy(user, owner_principal_id(credential), policy, credential.id),
            credential,
        )

    @integrations_router.delete("/{provider}/sync/policy", status_code=204)
    async def remove_connection_sync_policy(
        provider: str, account_id: str | None = None, user: User = Depends(get_authenticated_user)
    ) -> None:
        """Remove the connection's own policy; it inherits the owner's default again.

        ## Query Parameters
        - **account_id** (str, optional): As for POST /{provider}/sync.
        """
        credential = await _connected_credential_or_404(
            user, _integration_or_404(provider), account_id
        )
        await delete_sync_policy(user, owner_principal_id(credential), credential.id)

    @integrations_router.post("/{provider}/sync/policy/pause")
    async def pause_connection_sync_policy(
        provider: str, account_id: str | None = None, user: User = Depends(get_authenticated_user)
    ) -> SyncPolicyDTO:
        """Pause this connection only. It needs a policy of its own: a 404 otherwise.

        ## Query Parameters
        - **account_id** (str, optional): As for POST /{provider}/sync.
        """
        credential = await _connected_credential_or_404(
            user, _integration_or_404(provider), account_id
        )
        return await _connection_policy_dto(
            await pause_sync_policy(user, owner_principal_id(credential), credential.id), credential
        )

    @integrations_router.post("/{provider}/sync/policy/resume")
    async def resume_connection_sync_policy(
        provider: str, account_id: str | None = None, user: User = Depends(get_authenticated_user)
    ) -> SyncPolicyDTO:
        """Resume this connection; its next slot is the first one after now.

        ## Query Parameters
        - **account_id** (str, optional): As for POST /{provider}/sync.
        """
        credential = await _connected_credential_or_404(
            user, _integration_or_404(provider), account_id
        )
        return await _connection_policy_dto(
            await resume_sync_policy(user, owner_principal_id(credential), credential.id),
            credential,
        )

    @integrations_router.delete("/{provider}/connection")
    async def disconnect(
        provider: str,
        delete_data: bool = False,
        account_id: str | None = None,
        user: User = Depends(get_authenticated_user),
    ) -> ConnectionDisconnectResultDTO:
        """Disconnect the account connected by the requesting user.

        Marks the stored installation revoked by its owner, then best-effort asks
        the provider to kill the token on its own side via
        ``integration.revoke_remote``. That call is wrapped here too, on top
        of each adapter's own best-effort handling — a third-party
        integration that doesn't honor the "never raise" contract on
        ``revoke_remote`` still must not block the local disconnect.

        ## Path Parameters
        - **provider** (str): Key of a registered OAuth provider (see GET
          /api/v1/integrations/status).

        ## Query Parameters
        - **account_id** (Optional[str]): Connection to use (from GET
          /api/v1/integrations/{provider}/connections); the newest one is used when omitted.
          An id that is not the caller's returns 404 and never falls back to another
          connection; a blank id returns 422.
        - **delete_data** (bool): When true, the integration's dataset is deleted after the
          connection is revoked, with the dlt staging copy it was synced through (the staged
          rows, the sync state and the load packages), together with what the provider's
          app stored for the caller outside it (Slack: the messages the caller remembered
          from that workspace). A dataset that was never created (nothing synced yet) is
          nothing to delete, and the app's data is deleted whatever happens to the dataset. A
          provider that exposes no deletable dataset returns 400, and a sync of the connection
          still running returns 409, both before anything is revoked; a deletion that fails after
          the disconnect returns 502, unless the deletion itself answered with an HTTP error,
          whose status is kept. A staging copy still waiting for its lock after
          ``STAGING_DROP_WAIT_SECONDS`` is dropped in the background, and a failure there is
          logged. Ignored when account_id is omitted and the caller has no connection. Defaults
          to False.

        ## Response
        - **disconnected** (bool): Whether a connection was revoked locally.
        - **remoteRevokeSkipped** (bool): True when ``ALLOW_HTTP_REQUESTS=false`` kept the
          server from asking the provider to revoke the token, which then stays live at the
          provider until it expires or the user removes the app there.
        """
        with new_span("cognee.integrations.disconnect") as span:
            span.set_attribute("cognee.integrations.provider", provider)
            integration = _integration_or_404(provider)
            credential = await _credential_or_404(user, integration, account_id)
            if credential is None or credential.provider_account_id is None:
                return ConnectionDisconnectResultDTO(disconnected=False)

            dataset_name = None
            if delete_data:
                dataset_name = integration.dataset_name(credential)
                if not dataset_name:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{provider} does not expose a deletable dataset",
                    )
                from cognee.modules.integrations.ingestion import sync_is_running

                # A sync stops at its next check once the connection is revoked, but a
                # scope it already started could store into the dataset deleted below.
                if sync_is_running(provider, credential.provider_account_id):
                    raise HTTPException(
                        status_code=409,
                        detail=f"{provider} is syncing; disconnect once the sync has finished",
                    )

            # Ended locally first, so the provider's echo of this revoke (Slack's
            # tokens_revoked, a refresh that meets the revoked token) finds it ended by
            # its owner and records nothing over it.
            await revoke_credential_by_account(
                integration.provider,
                credential.provider_account_id,
                revoked_by=REVOKED_BY_OWNER,
            )

            remote_revoke_skipped = False
            try:
                require_outbound_http(provider)
            except OutboundHttpDisabledError:
                # Not attempted rather than attempted and swallowed: the caller is
                # told, since the token stays live at the provider.
                remote_revoke_skipped = True
                logger.warning(
                    "%s token for account %s was not revoked at the provider: "
                    "ALLOW_HTTP_REQUESTS disables outbound HTTP",
                    provider,
                    credential.provider_account_id,
                )
            else:
                try:
                    await integration.revoke_remote(credential)
                except Exception:  # a remote-revoke failure must never block disconnect
                    logger.exception(
                        "%s revoke_remote raised for account %s",
                        provider,
                        credential.provider_account_id,
                    )

            if delete_data:
                from cognee.api.v1.forget.forget import forget
                from cognee.modules.data.exceptions import DatasetNotFoundError
                from cognee.tasks.ingestion.ingest_dlt_source import drop_dlt_dataset

                try:
                    try:
                        try:
                            await forget(dataset=dataset_name, user=user)
                        except DatasetNotFoundError:
                            pass  # nothing was synced yet, so there is no dataset to delete
                    finally:
                        # The credential is already revoked, so this is the only chance to
                        # delete the app's own data: it runs whatever the dataset did.
                        await integration.forget_related_data(credential, user)
                    # forget leaves dlt's staging copy, state and load packages, which hold
                    # every synced row's text: the next connect and sync would build every
                    # document again.
                    await _drop_staging_copy(provider, dataset_name, drop_dlt_dataset)
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
            return ConnectionDisconnectResultDTO(
                disconnected=True, remote_revoke_skipped=remote_revoke_skipped
            )

    return integrations_router
