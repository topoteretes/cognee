"""``@cognee status``: who the member is to cognee, what cognee does with this channel, and
the workspace's sync.

Every row comes from data the code already keeps: the member's link, the connection's
metadata (installer, selection, command allowlist, last sync counts), the answer's own scope
rule (``channel_scope``), the sync schedule (``next_sync_at``) and the workspace dataset. A
row that needs Slack and cannot reach it says "Unknown" and the failure is logged, so Slack
being down never hides the rest; a failing read of cognee's own data fails the status. A selected private channel is only ever counted: the member asking may
not be in it.
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.ingestion import sync_is_running
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.slack import channel_membership
from cognee.modules.integrations.slack.persistence import PROVIDER, channel_allowed
from cognee.modules.integrations.slack.slack_settings import slack_settings
from cognee.modules.integrations.slack.sync import (
    LAST_SYNC_COUNTS_KEY,
    SYNC_STATUS_DEGRADED,
    degraded_reasons,
    is_paused,
    next_sync_at,
    selected_channels,
    workspace_item_count,
)
from cognee.modules.users.methods import get_user

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"
FOOTER = "Manage channels in cognee → Integrations. Only you can see this."


async def status_message(
    credential: IntegrationCredential,
    owner_user_id: UUID | None,
    *,
    channel: str | None,
    member: str,
) -> dict[str, Any]:
    """The status of ``channel`` (``None`` in a direct message) for the Slack ``member``
    linked as ``owner_user_id`` (``None`` when not linked), as ``{"text", "blocks"}``."""
    installer = (credential.provider_metadata or {}).get("installed_by_slack_user_id")
    groups = [
        [
            ("You", await _you(owner_user_id)),
            ("Connected by", f"<@{installer}>" if installer else UNKNOWN),
        ],
        [
            ("This channel", _this_channel(credential, channel, installer)),
            ("Answers here", await _answers_here(credential, channel, member)),
        ],
        [
            ("Memory", await _memory(credential)),
            ("Last sync", _last_sync(credential)),
            ("Next sync", await _next_sync(credential)),
        ],
    ]
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "*cognee status*"}}
    ]
    for rows in groups:
        blocks += [
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*{label}*\n{value}"} for label, value in rows
                ],
            },
        ]
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": FOOTER}]})
    text = "\n".join(
        ["cognee status", *(f"{label}: {value}" for rows in groups for label, value in rows)]
    )
    return {"text": text, "blocks": blocks}


async def _you(owner_user_id: UUID | None) -> str:
    if owner_user_id is None:
        return "Not linked: send `@cognee link`"
    try:
        return f"Linked as {(await get_user(owner_user_id)).email}"
    except EntityNotFoundError:
        logger.error("A Slack member is linked to a deleted cognee user")
        return UNKNOWN


def _this_channel(
    credential: IntegrationCredential, channel: str | None, installer: str | None
) -> str:
    if channel is None:
        return "This is a direct message"
    if channel in selected_channels(credential):
        synced = "Synced"
    else:
        synced = f"Not synced: ask <@{installer}> to add it" if installer else "Not synced"
    commands = "commands on" if channel_allowed(credential, channel) else "Commands are off here"
    return f"{synced} · {commands}"


async def _answers_here(credential: IntegrationCredential, channel: str | None, member: str) -> str:
    if channel is None:
        return "Your channels and your memory"
    scope = await channel_membership.channel_scope(credential, channel, member)
    if not scope.known:
        return UNKNOWN
    if scope.shared:
        return "Read this channel only (shared with another organization)"
    if scope.guest:
        return "Read this channel only (you are a guest here)"
    others = len([scoped for scoped in scope.channels if scoped != channel])
    counted = f"{others} public channel{'' if others == 1 else 's'}"
    if channel not in scope.channels:
        return f"Read {counted}"
    return f"Read <#{channel}>" + (f" and {counted}" if others else "")


async def _memory(credential: IntegrationCredential) -> str:
    selected = selected_channels(credential)
    try:
        public = await channel_membership.public_channels(credential)
    except Exception:  # the way to Slack: one row says Unknown, the rest still shows
        logger.exception("Could not ask Slack which channels are public for the status")
        return UNKNOWN
    items = await workspace_item_count(credential)
    shown = len([channel for channel in selected if channel in public])
    return (
        f"{_count(len(selected), 'channel')} synced"
        f" ({shown} public, {len(selected) - shown} private) · {_count(items, 'conversation')}"
    )


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _last_sync(credential: IntegrationCredential) -> str:
    last = credential.last_synced_at
    if last is None:
        return "Not yet"
    counts = (credential.provider_metadata or {}).get(LAST_SYNC_COUNTS_KEY) or {}
    lines = [
        f"{_slack_date(last)} · {counts.get('emitted', 0)} new, {counts.get('failed', 0)} failed"
    ]
    if credential.sync_status == SYNC_STATUS_DEGRADED:
        lines += degraded_reasons(counts)
    return "\n".join(lines)


async def _next_sync(credential: IntegrationCredential) -> str:
    if sync_is_running(PROVIDER, str(credential.provider_account_id)):
        return "Syncing now"
    if await is_paused(credential):
        return "Paused by the owner"
    if not slack_settings.history_sync_enabled:
        return "Off on this server"
    if not selected_channels(credential):
        return "No channels selected"
    due = next_sync_at(credential)
    if due is None or due <= datetime.now(timezone.utc):
        return "Within a minute"
    return _slack_date(due)


def _slack_date(moment: datetime) -> str:
    """A Slack date token, shown in the reader's own time zone, with a UTC fallback."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)  # SQLite stores UTC without a zone
    fallback = moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"<!date^{int(moment.timestamp())}^{{date_short_pretty}} at {{time}}|{fallback}>"
