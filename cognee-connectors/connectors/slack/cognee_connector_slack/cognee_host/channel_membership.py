"""Who is in a Slack channel, as Slack says: the app, for remembering, and a member, for
answering.

Copied from cognee.modules.integrations.slack.channel_membership (cognee dev, 2026-11-12).

A message remembered from a conversation is forgotten when Slack reports it edited or
deleted, and Slack reports that only for the channels the app is a member of. Direct
messages and group DMs are never reported (the app does not subscribe to them), so
nothing is remembered from them either. ``@cognee`` answers a direct message from the
synced channels Slack says the member belongs to (``member_channels``), and a question in a
channel from that channel and the public ones (``public_channels``), unless the channel is
shared with another organization (``shared_with_other_organizations``) or the member asking
is a guest (``is_guest``).
"""

import asyncio
import logging
from dataclasses import dataclass

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.outbound import require_outbound_http

# This package's source and client, not ``ingestion.build_service`` / ``source_factory``:
# on a cognee that bundles Slack those resolve to its own copy, whose errors are not the
# classes this module catches.
from cognee_connector_slack import source as slack_source_module
from cognee_connector_slack.cognee_host.adapter import access_token_for
from cognee_connector_slack.cognee_host.persistence import PROVIDER
from cognee_connector_slack.cognee_host.sync import selected_channels

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelScope:
    """What an answer in a channel reads (``channels``), whether the channel is shared with
    another organization (``shared``), whether the member asking is a guest (``guest``), and
    whether Slack answered (``known``)."""

    channels: list[str]
    shared: bool
    known: bool
    guest: bool = False


async def channel_scope(
    credential: IntegrationCredential, channel: str, member: str
) -> ChannelScope:
    """What everyone in ``channel`` can read, asked by ``member``: the channel itself when it
    is synced, and the selected public channels. The one rule both the answer and
    ``@cognee status`` use.

    A Slack Connect channel gets itself alone: its other organization's members cannot read
    this workspace's public channels. A guest asking gets it alone too, since a guest cannot
    read them either; whether a channel has other guests Slack does not say per channel
    (the Limits in ``slack.md``). When Slack cannot say, the channel alone too, with
    ``known`` false. Never raises.
    """
    from cognee_connector_slack.source import SlackSourceError

    selected = selected_channels(credential)
    own = [channel] if channel in selected else []
    team_id = credential.provider_account_id
    try:
        if await shared_with_other_organizations(credential, channel):
            return ChannelScope(own, shared=True, known=True)
        if await is_guest(credential, member):
            return ChannelScope(own, shared=False, known=True, guest=True)
        public = await public_channels(credential)
    except SlackSourceError as error:
        logger.warning("Slack did not describe the channels for team %s: %s", team_id, error)
        return ChannelScope(own, shared=False, known=False)
    except Exception:
        # Anything else on the way to Slack: just as unknown, so it fails closed the same way.
        logger.exception("Could not ask Slack which channels are public for team %s", team_id)
        return ChannelScope(own, shared=False, known=False)
    others = [other for other in selected if other in public and other != channel]
    return ChannelScope(own + others, shared=False, known=True)


async def app_follows_channel(credential: IntegrationCredential, channel_id: str) -> bool:
    """Whether the app is a member of ``channel_id``, a public or private channel.

    ``False`` for a direct message, a group DM, and a channel the app is not in or
    cannot see (Slack answers ``channel_not_found`` for a private channel without it).
    Raises when Slack gives no clear answer, so an uncertain answer never remembers.
    """
    from cognee_connector_slack.source import SlackAPIError, SlackMissingScopeError

    if channel_id.startswith("D"):  # a direct message: the app has no scope to read about it
        return False
    require_outbound_http(PROVIDER)
    client = slack_source_module.build_slack_service(await access_token_for(credential))
    try:
        answer = await asyncio.to_thread(client.call, "conversations.info", channel=channel_id)
    except SlackAPIError as error:
        # A group DM needs a scope the app does not request, so Slack refuses to describe it.
        if error.code == "channel_not_found" or isinstance(error, SlackMissingScopeError):
            return False
        raise
    info = answer.get("channel") or {}
    return (
        info.get("id") == channel_id
        and not info.get("is_im")
        and not info.get("is_mpim")
        and bool(info.get("is_member"))
    )


async def member_channels(credential: IntegrationCredential, member: str) -> set[str]:
    """The public and private channels Slack says ``member`` belongs to.

    Asked with the bot token (``users.conversations``), every page. Raises
    ``SlackSourceError`` when Slack gives no clear answer, so a caller never takes an
    unknown membership for an empty or a full one.
    """
    return await _conversations(credential, user=member, types="public_channel,private_channel")


async def public_channels(credential: IntegrationCredential) -> set[str]:
    """The public channels the app is in, as Slack says now.

    Read when answering, not kept from the sync: a channel made private drops out at once.
    Raises like ``member_channels``.
    """
    return await _conversations(credential, types="public_channel")


async def shared_with_other_organizations(
    credential: IntegrationCredential, channel_id: str
) -> bool:
    """Whether ``channel_id`` is shared beyond this workspace: a Slack Connect channel
    (another organization) or, on Enterprise Grid, a channel shared with other workspaces
    of the organization.

    Members from elsewhere cannot read this workspace's public channels. Raises when Slack
    gives no clear answer.
    """
    from cognee_connector_slack.source import SlackAPIError

    require_outbound_http(PROVIDER)
    client = slack_source_module.build_slack_service(await access_token_for(credential))
    answer = await asyncio.to_thread(client.call, "conversations.info", channel=channel_id)
    info = answer.get("channel") or {}
    if info.get("id") != channel_id:
        raise SlackAPIError("conversations.info", "invalid_response")
    return bool(info.get("is_ext_shared") or info.get("is_org_shared"))


async def is_guest(credential: IntegrationCredential, member: str) -> bool:
    """Whether Slack says ``member`` is a multi- or single-channel guest (``users.info``),
    who cannot read the workspace's public channels. Raises when Slack gives no clear
    answer."""
    from cognee_connector_slack.source import SlackAPIError

    require_outbound_http(PROVIDER)
    client = slack_source_module.build_slack_service(await access_token_for(credential))
    answer = await asyncio.to_thread(client.call, "users.info", user=member)
    user = answer.get("user") or {}
    if user.get("id") != member:
        raise SlackAPIError("users.info", "invalid_response")
    return bool(user.get("is_restricted") or user.get("is_ultra_restricted"))


async def _conversations(credential: IntegrationCredential, **params: str) -> set[str]:
    """The channel ids ``users.conversations`` lists for ``params``, every page."""
    from cognee_connector_slack.source import PAGE_SIZE, SlackAPIError

    require_outbound_http(PROVIDER)
    client = slack_source_module.build_slack_service(await access_token_for(credential))
    channels: set[str] = set()
    cursor = ""
    cursors: set[str] = set()
    while True:
        page_params: dict[str, object] = {**params, "limit": PAGE_SIZE}
        if cursor:
            page_params["cursor"] = cursor
        page = await asyncio.to_thread(client.call, "users.conversations", **page_params)
        batch = page.get("channels")
        if not isinstance(batch, list):
            raise SlackAPIError("users.conversations", "invalid_response")
        channels |= {
            item["id"]
            for item in batch
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        cursor = ((page.get("response_metadata") or {}).get("next_cursor") or "").strip()
        if not cursor:
            return channels
        if cursor in cursors:
            raise SlackAPIError("users.conversations", "repeated_cursor")
        cursors.add(cursor)
