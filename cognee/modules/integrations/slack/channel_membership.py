"""Who is in a Slack channel, as Slack says: the app, for remembering, and a member, for
answering.

A message remembered from a conversation is forgotten when Slack reports it edited or
deleted, and Slack reports that only for the channels the app is a member of. Direct
messages and group DMs are never reported (the app does not subscribe to them), so
nothing is remembered from them either. ``@cognee`` answers a member from the synced
channels Slack says they belong to (``member_channels``).
"""

import asyncio

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.outbound import require_outbound_http
from cognee.modules.integrations.slack.adapter import access_token_for
from cognee.modules.integrations.slack.persistence import PROVIDER


async def app_follows_channel(credential: IntegrationCredential, channel_id: str) -> bool:
    """Whether the app is a member of ``channel_id``, a public or private channel.

    ``False`` for a direct message, a group DM, and a channel the app is not in or
    cannot see (Slack answers ``channel_not_found`` for a private channel without it).
    Raises when Slack gives no clear answer, so an uncertain answer never remembers.
    """
    from cognee.tasks.ingestion.connectors.slack import SlackAPIError, SlackMissingScopeError

    if channel_id.startswith("D"):  # a direct message: the app has no scope to read about it
        return False
    require_outbound_http(PROVIDER)
    client = ingestion.build_service(PROVIDER, await access_token_for(credential))
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
    from cognee.tasks.ingestion.connectors.slack import PAGE_SIZE, SlackAPIError

    require_outbound_http(PROVIDER)
    client = ingestion.build_service(PROVIDER, await access_token_for(credential))
    channels: set[str] = set()
    cursor = ""
    cursors: set[str] = set()
    while True:
        params = {"user": member, "types": "public_channel,private_channel", "limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        page = await asyncio.to_thread(client.call, "users.conversations", **params)
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
