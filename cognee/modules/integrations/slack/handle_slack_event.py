"""Events API dispatch for the Slack app.

Handles:

* ``app_uninstalled`` / ``tokens_revoked`` — lifecycle hygiene, revokes the
  stored installation so no dangling bot token survives an uninstall.
* ``app_home_opened`` — publishes the (static, informational) Home tab
  content. Slack shows a blank Home tab until the app calls
  ``views.publish``, so this is what makes clicking Cognee in the sidebar
  show anything at all.
* ``message`` events with the ``message_deleted`` subtype, and ``message_changed``
  when the message was edited or became a deleted-message placeholder — a message
  someone remembered with "Remember this" is forgotten, because what was remembered
  may be what its author took back. Every other message event is acked and ignored.

Slack delivers events at-least-once (3 retries: immediate, 1 min, 5 min) and
does NOT guarantee ordering between event types, so every handler here is
idempotent and independent of the others.

Channel history is synced by the dlt source in
``cognee/tasks/ingestion/connectors/slack.py`` (see sync.py), which re-walks
each selected channel and does not depend on receiving every message event
while the server is online.
"""

import asyncio
import json
import logging
from typing import Any

from cognee.infrastructure.background_tasks import register_background_task
from cognee.modules.integrations.credentials import decrypt_token_payload
from cognee.modules.integrations.slack.handle_cognee_mention import (
    handle_cognee_mention,
    is_direct_message,
)
from cognee.modules.integrations.slack.home import publish_home_view
from cognee.modules.integrations.slack.persistence import get_by_team, is_active, revoke_by_team
from cognee.modules.integrations.slack.remember_message import forget_remembered_message
from cognee.modules.integrations.slack.sync import mark_thread_changed, record_uninstall

logger = logging.getLogger(__name__)

_REVOKING_EVENT_TYPES = {"app_uninstalled", "tokens_revoked"}


def _ends_the_connection(event: dict) -> bool:
    """An uninstall always does; ``tokens_revoked`` only when the bot token is among the
    revoked ones. A linked member's user token (Sign in with Slack) is reported under
    ``oauth`` and leaves the workspace connected (G52)."""
    if event.get("type") != "tokens_revoked":
        return True
    return bool((event.get("tokens") or {}).get("bot"))


# Slack's ``X-Slack-Retry-Reason`` when it sends an event again because the ack took longer
# than 3 seconds: the first delivery reached the app and is being handled.
_SLOW_ACK = "http_timeout"


async def handle_slack_event(raw_body: bytes, *, retry_reason: str | None = None) -> dict[str, Any]:
    """Dispatch one signature-verified Events API request body.

    ``retry_reason`` is Slack's ``X-Slack-Retry-Reason`` on a delivery it sends again.
    Returns the JSON-serializable response for the router. Always answers
    within the 3-second window — ``app_home_opened``'s actual work (one
    ``views.publish`` call) comfortably fits that budget, and never turns
    into a non-200 response regardless; see ``_publish_home_view``.
    """
    envelope = json.loads(raw_body)

    # One-time endpoint ownership handshake: Slack POSTs a challenge when the
    # request URL is first configured and expects it echoed back.
    if envelope.get("type") == "url_verification":
        return {"challenge": envelope.get("challenge", "")}

    if envelope.get("type") != "event_callback":
        return {"ok": True}

    event = envelope.get("event") or {}
    event_type = event.get("type")
    team_id = envelope.get("team_id")

    if event_type in _REVOKING_EVENT_TYPES and team_id and _ends_the_connection(event):
        found = await revoke_by_team(team_id)
        if found and event_type == "app_uninstalled":
            # Slack's Developer Policy: the data is deleted 14 days later unless the
            # workspace is connected again by then (sync.forget_uninstalled_workspaces).
            await record_uninstall(team_id)
        logger.info(
            "Slack %s for team %s — installation %s",
            event_type,
            team_id,
            "revoked" if found else "not found",
        )
    elif event_type == "app_home_opened" and team_id:
        await _publish_home_view(team_id, event.get("user"))
    elif (
        event_type == "app_mention"
        and team_id
        or event_type == "message"
        and team_id
        and (is_direct_message(event))
    ):
        # Answered detached: a search outlives Slack's 3-second window. A question Slack
        # sends again after a slow ack is already being answered; one sent again after a
        # failed delivery is answered now, since the first never reached the app.
        if retry_reason != _SLOW_ACK:
            register_background_task(asyncio.create_task(handle_cognee_mention(team_id, event)))
    elif event_type == "message" and team_id:
        await _follow_message_event(team_id, event)

    # Unhandled event types are acked, not errored: erroring makes Slack
    # retry (and eventually disable the endpoint), which can never improve
    # the outcome for an event this app doesn't act on.
    return {"ok": True}


async def _follow_message_event(team_id: str, event: dict[str, Any]) -> None:
    """Forget the remembered copies of a message Slack reports deleted or edited.

    The copies are their members' memory, so this does not depend on the workspace's
    connection still being active. Unlike the other handlers this one lets a failure
    out: the event is then answered with an error and Slack sends it again (immediately,
    after a minute, after five), the only retry a missed forget gets. Forgetting is
    idempotent, so a retry is safe.
    """
    message = event.get("message") or {}
    await _mark_thread(team_id, event)
    if event.get("subtype") == "message_deleted":
        message_ts = event.get("deleted_ts")
    elif event.get("subtype") == "message_changed" and (
        message.get("edited") or message.get("subtype") == "tombstone"
    ):
        # message_changed also reports changes nobody made by hand (unfurls, language
        # detection). Two count: an edit carries "edited", and a thread parent deleted
        # while its replies stay is not removed but becomes a "tombstone" placeholder.
        # "edited" stays on a message once edited, so an unfurl on a message edited before
        # it was remembered forgets the copy too: forgetting is the safe side.
        message_ts = message.get("ts")
    else:
        return
    channel_id = event.get("channel")
    if not (channel_id and message_ts):
        return
    await forget_remembered_message(team_id, channel_id=channel_id, message_ts=message_ts)


def _file_ids(message: dict[str, Any]) -> list[str]:
    return [
        f.get("id") or f.get("name") or ""
        for f in message.get("files") or []
        if isinstance(f, dict)
    ]


async def _mark_thread(team_id: str, event: dict[str, Any]) -> None:
    """Mark the thread of a change its root cannot show: an edited or deleted reply, or a
    file removed from a message (G6). The next sync reads that thread again."""
    message = event.get("message") or {}
    previous = event.get("previous_message") or {}
    if event.get("subtype") == "message_deleted":
        target = previous
    elif event.get("subtype") == "message_changed" and (
        message.get("edited") or _file_ids(message) != _file_ids(previous)
    ):
        target = message
    else:
        return
    channel_id = event.get("channel")
    root_ts = target.get("thread_ts") or target.get("ts")
    if channel_id and root_ts:
        await mark_thread_changed(team_id, channel_id, root_ts)


async def _publish_home_view(team_id: str, slack_user_id: str | None) -> None:
    """Best-effort Home tab refresh — never raises.

    A broken Home tab is a cosmetic problem, not a reason to fail this
    event's ack — that would make Slack retry three times over six minutes
    for no benefit, since a views.publish failure won't fix itself on retry
    any faster than on the next app_home_opened.
    """
    if not slack_user_id:
        return

    credential = await get_by_team(team_id)
    if not is_active(credential):
        return

    try:
        access_token = decrypt_token_payload(credential).get("access_token")
        if access_token:
            await publish_home_view(access_token, slack_user_id)
    except Exception:  # a broken Home tab must never fail the event ack
        logger.exception("Failed to publish App Home view for team %s", team_id)
