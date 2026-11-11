"""Interactive payload dispatch — message shortcuts, block actions, etc.

Slack POSTs ``application/x-www-form-urlencoded`` for interactive payloads,
with a single ``payload`` field holding a URL-encoded JSON string — unlike
the Events API, which POSTs raw JSON directly. Parsed here from the
verified raw bytes (a parsed-body parameter would break signature verification).

Handles the "Remember this" and "Forget this" message shortcuts and the Share/Discard
buttons of an
``@cognee`` answer (see handle_cognee_ask.py). Every
other interactive payload type this app doesn't act on (other block
actions, modals, future shortcuts) acks empty rather than erroring — same
reasoning as unhandled slash commands and events: erroring only earns a
Slack retry, never a better outcome for a payload this app doesn't act on.
"""

import asyncio
import json
import logging
from functools import partial
from typing import Any
from urllib.parse import parse_qs
from uuid import UUID

from cognee.infrastructure.background_tasks import register_background_task
from cognee.modules.integrations.slack.handle_cognee_ask import (
    ASK_DISCARD_ACTION_ID,
    ASK_SHARE_ACTION_ID,
    handle_cognee_ask_discard,
    handle_cognee_ask_share,
)
from cognee.modules.integrations.slack.handle_cognee_mention import (
    NOT_CONNECTED_MESSAGE,
    NOTHING_TO_SAVE_MESSAGE,
    forget_and_confirm,
    notice,
    remember_and_confirm,
)
from cognee.modules.integrations.slack.handle_slack_link import NOT_LINKED_MESSAGE
from cognee.modules.integrations.slack.persistence import (
    CHANNEL_NOT_ENABLED_MESSAGE,
    channel_allowed,
    get_by_team,
    is_active,
    resolve_owner_user_id,
)
from cognee.modules.integrations.slack.response_url import post_to_response_url

logger = logging.getLogger(__name__)

REMEMBER_THIS_CALLBACK_ID = "remember_this"
FORGET_THIS_CALLBACK_ID = "forget_this"


async def handle_slack_interactive(raw_body: bytes) -> dict[str, Any]:
    """Dispatch one signature-verified interactive payload within Slack's 3-second window."""
    form = parse_qs(raw_body.decode())
    raw_payload = (form.get("payload") or [""])[0]
    if not raw_payload:
        return {}

    payload = json.loads(raw_payload)
    payload_type = payload.get("type")

    if payload_type == "message_action" and payload.get("callback_id") == REMEMBER_THIS_CALLBACK_ID:
        await _handle_remember_this(payload)
    elif payload_type == "message_action" and payload.get("callback_id") == FORGET_THIS_CALLBACK_ID:
        await _handle_forget_this(payload)
    elif payload_type == "block_actions":
        await _handle_block_action(payload)

    return {}


async def _handle_block_action(payload: dict[str, Any]) -> None:
    """Dispatch a clicked button by its ``action_id``."""
    action = (payload.get("actions") or [{}])[0]
    action_id = action.get("action_id")
    response_url = payload.get("response_url", "")

    if action_id == ASK_SHARE_ACTION_ID:
        response = await handle_cognee_ask_share(response_url, action.get("value", ""))
        await post_to_response_url(response_url, response)
    elif action_id == ASK_DISCARD_ACTION_ID:
        response = await handle_cognee_ask_discard(action.get("value", ""))
        await post_to_response_url(response_url, response)


async def _handle_remember_this(payload: dict[str, Any]) -> None:
    """Save the shortcut's target message to Cognee memory.

    Message actions don't render their direct HTTP response body as a
    message the way slash commands do — confirmation/error text is
    delivered via ``response_url`` instead, the same as the async
    ``@cognee`` answer.
    """
    response_url = payload.get("response_url", "")
    team_id = (payload.get("team") or {}).get("id", "")
    invoking_slack_user_id = (payload.get("user") or {}).get("id", "")

    credential = await get_by_team(team_id) if team_id else None
    if not is_active(credential):
        await post_to_response_url(
            response_url,
            notice(NOT_CONNECTED_MESSAGE),
        )
        return

    channel_id = (payload.get("channel") or {}).get("id", "")
    if not channel_allowed(credential, channel_id):
        await post_to_response_url(response_url, notice(CHANNEL_NOT_ENABLED_MESSAGE))
        return

    # See persistence.resolve_owner_user_id and handle_cognee_ask.py for the
    # full reasoning (real per-member linking, falling back to "only the
    # installer" as a stopgap).
    owner_user_id = await resolve_owner_user_id(credential, team_id, invoking_slack_user_id)
    if owner_user_id is None:
        await post_to_response_url(response_url, notice(NOT_LINKED_MESSAGE))
        return

    message = payload.get("message") or {}
    text = (message.get("text") or "").strip()
    if not text:
        await post_to_response_url(response_url, notice(NOTHING_TO_SAVE_MESSAGE))
        return

    # Asking Slack whether the app is in the channel can take longer than the 3-second
    # window, so the rest runs detached and answers through response_url. Anchored until
    # it finishes, and waited for when the server shuts down.
    register_background_task(
        asyncio.create_task(
            remember_and_confirm(
                partial(post_to_response_url, response_url),
                credential,
                owner_user_id=owner_user_id,
                text=text,
                team_id=team_id,
                channel_id=channel_id,
                message_ts=message.get("ts", ""),
                channel_name=(payload.get("channel") or {}).get("name"),
                author_id=message.get("user"),
            )
        )
    )


async def _handle_forget_this(payload: dict[str, Any]) -> None:
    """Forget the message for the member: their own saved copy, and, from a synced channel,
    the workspace's memory too when they wrote it or own the connection.

    No allowlist check: the allowlist limits where Cognee is used, and taking back
    one's own copy is never limited.
    """
    response_url = payload.get("response_url", "")
    team_id = (payload.get("team") or {}).get("id", "")
    invoking_slack_user_id = (payload.get("user") or {}).get("id", "")

    credential = await get_by_team(team_id) if team_id else None
    if not is_active(credential):
        await post_to_response_url(
            response_url,
            notice(NOT_CONNECTED_MESSAGE),
        )
        return
    owner_user_id = await resolve_owner_user_id(credential, team_id, invoking_slack_user_id)
    if owner_user_id is None:
        await post_to_response_url(response_url, notice(NOT_LINKED_MESSAGE))
        return

    message = payload.get("message") or {}
    register_background_task(
        asyncio.create_task(
            forget_and_confirm(
                partial(post_to_response_url, response_url),
                credential,
                owner_user_id=owner_user_id,
                team_id=team_id,
                channel_id=(payload.get("channel") or {}).get("id", ""),
                message_ts=message.get("ts", ""),
                # A reply re-renders its thread; a message outside one, its day.
                root_ts=message.get("thread_ts") or message.get("ts", ""),
                member=invoking_slack_user_id,
                author_id=message.get("user", ""),
            )
        )
    )
