"""Slack as an ``OAuthIntegration`` adapter.

Copied from cognee.modules.integrations.slack.adapter (cognee dev, 2026-11-12).

Wraps the already-tested free functions in
:mod:`cognee_connector_slack.cognee_host.oauth` behind the generic contract so
:mod:`cognee.api.v1.integrations.routers.get_integrations_router` can
dispatch to Slack the same way it will dispatch to any future provider.
Nothing here changes Slack's own OAuth behavior — it only gives that
behavior a uniform shape a second provider can copy.
"""

import asyncio
import logging
from collections.abc import Collection
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration, refuse_parent
from cognee.modules.integrations.credentials import (
    decrypt_token_payload,
    require_active_credential,
)
from cognee.modules.integrations.crypto import encrypt_credentials
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.outbound import client_session
from cognee.modules.users.models import User
from sqlalchemy import update

from cognee_connector_slack.cognee_host import oauth as _oauth
from cognee_connector_slack.cognee_host.channels import list_channels
from cognee_connector_slack.cognee_host.remember_message import forget_remembered
from cognee_connector_slack.cognee_host.sync import (
    _CHANNEL,
    ARCHIVED,
    SELECTION_KEY,
    UNINSTALL_DELETED_AT_KEY,
    UNINSTALLED_AT_KEY,
    SlackChannelAccessError,
    channel_members,
    channel_verdict,
    dataset_name_for_credential,
    selected_channels,
    sync_slack,
)
from cognee_connector_slack.settings import SlackSettings, require

logger = logging.getLogger(__name__)

_REVOKE_URL = "https://slack.com/api/auth.revoke"
_TIMEOUT = aiohttp.ClientTimeout(total=30)
# Refresh this far before a rotating token dies (the former history importer's margin).
_REFRESH_MARGIN = timedelta(seconds=120)
# One lock per workspace, so two callers that find the token expiring do not each spend
# the same refresh token. Process-local, like the sync guard in ``ingestion``; the
# compare-and-swap in ``refresh`` covers a second process.
_refresh_locks: dict[str, asyncio.Lock] = {}


# Slack's answers to a refresh whose refresh token it no longer accepts.
_REFRESH_TOKEN_REJECTED = frozenset({"invalid_refresh_token", "token_revoked", "invalid_grant"})


class SlackReconnectError(RuntimeError):
    """The connection holds no token Slack still accepts and cannot get one by itself (no
    access token stored, a refresh Slack refused): only connecting again helps (G85)."""


class SlackIntegration(OAuthIntegration):
    provider = "slack"
    settings_cls = SlackSettings
    resource_selection_key = SELECTION_KEY
    # Nothing syncs until the user picks channels: channel history costs LLM tokens.
    resource_selection_allows_all = False

    def authorize_url(self, state: str) -> str:
        return _oauth.build_authorize_url(state)

    async def exchange_code(self, code: str) -> dict[str, Any]:
        return await _oauth.exchange_code(code)

    def parse_installation(self, token_response: dict[str, Any]) -> OAuthInstallation:
        team = token_response.get("team") or {}
        enterprise = token_response.get("enterprise") or {}

        # Slack's routing id: team.id normally, enterprise.id for a Grid org
        # install.
        account_id = team.get("id") or enterprise.get("id")
        if not account_id:
            raise ValueError("oauth.v2.access response carries neither team.id nor enterprise.id")

        token_payload = {
            "access_token": token_response.get("access_token"),
            "refresh_token": token_response.get("refresh_token"),
        }
        # Non-secret, Slack-specific fields needed later (posting messages,
        # Grid-awareness) — kept out of the encrypted blob so they stay queryable.
        # installed_by_slack_user_id: the Slack user who completed this OAuth
        # install (Slack always populates authed_user.id, even for a bot-only
        # install with no user scopes). Cognee has no real Slack-user <->
        # cognee-user linking yet (tracked separately), so this is used as a
        # stopgap authorization check — see handle_cognee_ask.py and
        # handle_slack_interactive.py's "Remember this" handler — to stop any
        # workspace member from querying/writing to the connecting user's
        # memory, not just the person who actually connected it.
        provider_metadata = {
            "bot_user_id": token_response.get("bot_user_id"),
            "enterprise_id": enterprise.get("id"),
            "installed_by_slack_user_id": (token_response.get("authed_user") or {}).get("id"),
            # A new connection syncs no channel until the user picks some.
            # complete_installation keeps an existing choice on reconnect.
            SELECTION_KEY: [],
            # Connected again: an uninstall's pending deletion is cancelled.
            UNINSTALLED_AT_KEY: None,
            UNINSTALL_DELETED_AT_KEY: None,
        }

        token_expires_at = None
        expires_in = token_response.get("expires_in")
        if expires_in:
            token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

        return OAuthInstallation(
            provider_account_id=account_id,
            token_payload=token_payload,
            provider_metadata=provider_metadata,
            account_label=team.get("name"),
            scopes=token_response.get("scope"),
            token_expires_at=token_expires_at,
        )

    def frontend_base_url(self) -> str:
        return require("frontend_base_url")

    def dataset_name(self, credential: IntegrationCredential) -> str:
        return dataset_name_for_credential(credential)

    def data_deleted_after(self, credential: IntegrationCredential) -> datetime | None:
        """14 days after an uninstall, unless the workspace is connected again (G28)."""
        from cognee_connector_slack.cognee_host.sync import UNINSTALL_RETENTION, _uninstalled_at

        uninstalled = _uninstalled_at(credential)
        return uninstalled + UNINSTALL_RETENTION if uninstalled is not None else None

    def data_deleted_at(self, credential: IntegrationCredential) -> datetime | None:
        """When an uninstall's deletion ran (G84)."""
        from cognee_connector_slack.cognee_host.sync import _uninstall_deleted_at

        return _uninstall_deleted_at(credential)

    async def forget_related_data(self, credential: IntegrationCredential, user: User) -> None:
        """Also forget what the disconnecting user remembered from this workspace, and the
        answers recorded for rating under them."""
        from cognee_connector_slack.cognee_host.handle_cognee_ask import forget_answer_records

        await forget_remembered(user, team_id=credential.provider_account_id)
        await forget_answer_records(str(credential.provider_account_id), user.id)

    async def sync_now(self, credential: IntegrationCredential) -> None:
        await sync_slack(credential)

    def invalid_resource_ids(self, resource_ids: list[str]) -> list[str]:
        """Anything but a channel id (``C…`` or ``G…``): a DM, a name, a lowercase id (G51)."""
        return [resource_id for resource_id in resource_ids if not _CHANNEL.fullmatch(resource_id)]

    async def list_resources(
        self, credential: IntegrationCredential, *, parent: str | None = None
    ) -> list[dict[str, Any]]:
        """The channels a user can pick for the sync: the ones the app is in, and the
        selected ones Slack has archived since.

        Only names the person who connected the workspace may see (``visible_channels``).
        Which channels are synced is this selection's alone: the command allowlist never
        narrows it. The sync checks membership again, for every channel, before it reads one.
        An archived channel is listed only while it is selected, marked ``is_archived``:
        Slack's channel list leaves it out, and a client saving back the listed channels
        would drop it from the selection and retire the data the sync keeps for it. A
        selected private channel whose members cannot be read right now stays listed for
        the same reason (``visible_channels``).
        """
        refuse_parent(self.provider, parent)
        token = await access_token_for(credential)
        selected = selected_channels(credential)
        channels = [
            channel
            for channel in await visible_channels(credential, token, selected=selected)
            if channel["is_member"]
        ]
        listed = {channel["id"] for channel in channels}
        unlisted = [channel for channel in selected if channel not in listed]
        channels += await selected_archived_channels(credential, token, unlisted)
        return [
            {
                "id": channel["id"],
                "name": f"#{channel['name']}",
                "description": None,
                "attributes": {
                    "is_private": channel["is_private"],
                    "is_archived": channel.get("is_archived", False),
                    "is_shared": channel.get("is_shared", False),
                },
            }
            for channel in channels
        ]

    async def revoke_remote(self, credential: IntegrationCredential) -> None:
        """Best-effort ``auth.revoke`` so a disconnected workspace's token
        can't still be used against the Slack API afterwards.

        Never raises — a Slack-side outage or an already-revoked token must
        not block the local disconnect the caller is trying to complete. No
        retry on ``ratelimited``: disconnect doesn't depend on this call
        succeeding, so it's not worth delaying. ``already_revoked`` and
        ``ratelimited`` are logged at info (expected, not actionable);
        anything else is a warning, since that's a real misconfiguration
        (bad credentials, revoked app) worth surfacing in logs.
        """
        token_payload = decrypt_token_payload(credential)
        access_token = token_payload.get("access_token")
        if not access_token:
            return

        try:
            async with (
                client_session("slack", timeout=_TIMEOUT) as session,
                session.post(
                    _REVOKE_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                    allow_redirects=False,
                ) as response,
            ):
                payload = await response.json()
        except Exception:  # a failed revoke must never block disconnect
            logger.exception(
                "Slack auth.revoke request failed for account %s", credential.provider_account_id
            )
            return

        error = payload.get("error")
        if not payload.get("ok") and error != "already_revoked":
            log = logger.info if error == "ratelimited" else logger.warning
            log(
                "Slack auth.revoke returned %s for account %s",
                error,
                credential.provider_account_id,
            )

    async def refresh(self, credential: IntegrationCredential) -> None:
        """Rotate the access token via Slack's refresh-token grant.

        A no-op when the stored token payload carries no ``refresh_token`` —
        that's the normal case for most Slack apps, since token rotation is
        an opt-in, irreversible setting on the Slack app itself
        (https://docs.slack.dev/authentication/rotating-and-refreshing-credentials).
        Raises ``RuntimeError`` (naming Slack's error code) if a refresh that
        *should* have worked was rejected — callers decide whether that's
        worth surfacing or just logging.
        """
        token_payload = decrypt_token_payload(credential)
        refresh_token = token_payload.get("refresh_token")
        if not refresh_token:
            return

        async with (
            client_session("slack", timeout=_TIMEOUT) as session,
            session.post(
                _oauth._ACCESS_URL,
                data={
                    "client_id": require("client_id"),
                    "client_secret": require("client_secret"),
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                # The body carries the client secret and the refresh token:
                # never let a redirect replay it elsewhere.
                allow_redirects=False,
            ) as response,
        ):
            if response.status != 200:
                raise RuntimeError(f"Slack token refresh failed: HTTP {response.status}")
            payload = await response.json()

        if not payload.get("ok"):
            code = payload.get("error", "unknown")
            # Only a refresh token Slack rejects needs a reconnect; any other answer (a
            # passing ``internal_error``, a server misconfiguration) does not (G95).
            if code in _REFRESH_TOKEN_REJECTED:
                raise SlackReconnectError(f"Slack token refresh failed: {code}")
            raise RuntimeError(f"Slack token refresh failed: {code}")

        # Refresh responses need not contain the installation's team/user
        # metadata. Update token columns only, preserving channel restrictions,
        # sync selections, installer identity and workspace ownership. A
        # concurrent reconnect/revocation must win over an in-flight refresh.
        if not payload.get("access_token") or not payload.get("refresh_token"):
            raise RuntimeError("Slack token refresh returned an incomplete token response")
        encrypted = encrypt_credentials(
            {
                **token_payload,
                "access_token": payload["access_token"],
                "refresh_token": payload["refresh_token"],
            }
        )
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(payload["expires_in"]))
        engine = get_relational_engine()
        async with engine.get_async_session() as db:
            result = await db.execute(
                update(IntegrationCredential)
                .where(
                    IntegrationCredential.id == credential.id,
                    IntegrationCredential.user_id == credential.user_id,
                    IntegrationCredential.status == "active",
                    IntegrationCredential.ciphertext == credential.ciphertext,
                )
                .values(
                    ciphertext=encrypted[0],
                    nonce=encrypted[1],
                    encryption_version=encrypted[2],
                    key_id=encrypted[3],
                    token_expires_at=expires_at,
                    scopes=payload.get("scope") or credential.scopes,
                )
            )
            if result.rowcount != 1:
                raise RuntimeError("Slack connection changed while refreshing its token")
            await db.commit()


async def visible_channels(
    credential: IntegrationCredential, token: str, *, selected: Collection[str] = ()
) -> list[dict[str, Any]]:
    """The channels whose names the person who connected the workspace may see.

    Every public channel, and each private channel the app is in that this person is in
    too: the app may be in private channels its installer cannot see, and listing them
    would show their names. Without a known installer no private channel is listed, and a
    private channel whose members cannot be read right now is left out, unless it is in
    ``selected`` (the sync's selection, or the command allowlist): its owner picked it
    already, and a list saved back would otherwise drop it, which for the sync retires its
    data. Backs both pickers,
    the sync's and the command allowlist's. Raises ``RuntimeError`` when Slack refuses the
    channel list (``list_channels``).
    """
    from cognee_connector_slack.source import SlackAPIError

    installer = (credential.provider_metadata or {}).get("installed_by_slack_user_id")
    client = _picker_client(token)
    visible = []
    unreadable = 0
    for channel in await list_channels(token, types="public_channel,private_channel"):
        if channel["is_private"]:
            if not (installer and channel["is_member"]):
                continue
            try:
                members = await asyncio.to_thread(channel_members, client, channel["id"])
            except (SlackAPIError, SlackChannelAccessError):
                # Rate limited or refused: checked again by the sync before it reads it.
                if channel["id"] in selected:
                    visible.append(channel)
                else:
                    unreadable += 1
                continue
            if installer not in members:
                continue
        visible.append(channel)
    if unreadable:
        logger.warning(
            "Slack picker hid %s private channel(s) of workspace %s whose members "
            "could not be read",
            unreadable,
            credential.provider_account_id,
        )
    return visible


async def selected_archived_channels(
    credential: IntegrationCredential, token: str, channel_ids: list[str]
) -> list[dict[str, Any]]:
    """The given selected channels that Slack has archived, named by the sync's own verdict.

    A channel gets the verdict ``ARCHIVED`` only when the person who connected the workspace
    may read it, so no private channel they cannot see is named. A channel that is not
    archived, or that Slack cannot describe right now, is left out: the sync reports it.
    """
    from cognee_connector_slack.source import SlackAPIError

    installer = (credential.provider_metadata or {}).get("installed_by_slack_user_id") or ""
    client = _picker_client(token)
    archived = []
    for channel_id in channel_ids:
        try:
            verdict, info = await asyncio.to_thread(channel_verdict, client, channel_id, installer)
        except (SlackAPIError, SlackChannelAccessError):
            continue
        if verdict == ARCHIVED:
            archived.append(
                {
                    "id": channel_id,
                    "name": info.get("name") or channel_id,
                    "is_private": bool(info.get("is_private")),
                    "is_archived": True,
                }
            )
    return archived


def _picker_client(access_token: str) -> Any:
    """A Slack client that never waits out a rate limit: the picker answers an HTTP request."""
    from cognee_connector_slack.source import SlackClient

    return SlackClient(access_token, max_retry_after=0)


async def access_token_for(credential: IntegrationCredential) -> str:
    """A bot token, refreshed first when a rotating one is within two minutes of expiry."""
    credential = await require_active_credential(credential)
    if _is_expiring(credential):
        async with _refresh_locks.setdefault(str(credential.provider_account_id), asyncio.Lock()):
            # A caller that held the lock first may have rotated the token already, so
            # read the row back before spending the refresh token.
            credential = await require_active_credential(credential)
            if _is_expiring(credential):
                await SlackIntegration().refresh(credential)
                # refresh() writes through its own session; read the new token back.
                credential = await require_active_credential(credential)

    token = decrypt_token_payload(credential).get("access_token")
    if not token:
        raise SlackReconnectError(
            f"Slack credential for workspace {credential.provider_account_id} "
            "holds no access token; reconnect Slack"
        )
    return token


def _is_expiring(credential: IntegrationCredential) -> bool:
    expires_at = credential.token_expires_at
    if expires_at is None:
        return False
    # SQLite stores no timezone; the value is written as UTC.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at - _REFRESH_MARGIN <= datetime.now(timezone.utc)
