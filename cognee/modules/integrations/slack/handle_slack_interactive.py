"""Interactive payload dispatch — message shortcuts, block actions, etc.

Slack POSTs ``application/x-www-form-urlencoded`` for interactive payloads,
with a single ``payload`` field holding a URL-encoded JSON string — unlike
the Events API, which POSTs raw JSON directly. Parsed here from the
verified raw bytes, same reasoning as ``handle_slack_command``'s module
docstring (a parsed-body parameter would break signature verification).

Handles the "Remember this" message shortcut and the ``/cognee-ask`` answer
review prompt's Share/Discard buttons (see handle_cognee_ask.py). Every
other interactive payload type this app doesn't act on (other block
actions, modals, future shortcuts) acks empty rather than erroring — same
reasoning as unhandled slash commands and events: erroring only earns a
Slack retry, never a better outcome for a payload this app doesn't act on.
"""

import json
import logging
from typing import Any, Optional
from urllib.parse import parse_qs

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.slack.handle_cognee_ask import (
    ASK_DISCARD_ACTION_ID,
    ASK_SHARE_ACTION_ID,
    handle_cognee_ask_discard,
    handle_cognee_ask_share,
)
from cognee.modules.integrations.slack.persistence import (
    get_by_team,
    is_active,
    resolve_owner_user_id,
)
from cognee.modules.integrations.slack.remember_message import remember_message
from cognee.modules.integrations.slack.response_url import post_to_response_url

logger = logging.getLogger(__name__)

REMEMBER_THIS_CALLBACK_ID = "remember_this"


def _ephemeral(text: str, *, replace_original: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {"response_type": "ephemeral", "text": text}
    if replace_original:
        payload["replace_original"] = True
    return payload


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
    ``/cognee-ask`` answer.
    """
    response_url = payload.get("response_url", "")
    team_id = (payload.get("team") or {}).get("id", "")
    invoking_slack_user_id = (payload.get("user") or {}).get("id", "")

    credential = await get_by_team(team_id) if team_id else None
    if not is_active(credential):
        await post_to_response_url(
            response_url,
            _ephemeral(
                "This Slack workspace is not connected to Cognee. "
                "Connect it from your Integrations settings."
            ),
        )
        return

    # See persistence.resolve_owner_user_id and handle_cognee_ask.py for the
    # full reasoning (real per-member linking, falling back to "only the
    # installer" as a stopgap).
    owner_user_id = await resolve_owner_user_id(credential, team_id, invoking_slack_user_id)
    if owner_user_id is None:
        await post_to_response_url(
            response_url,
            _ephemeral(
                "Link your own Cognee account first: `/cognee-link <api_key>` "
                "(create a key from your Cognee account's API Keys settings)."
            ),
        )
        return

    message = payload.get("message") or {}
    text = (message.get("text") or "").strip()
    if not text:
        await post_to_response_url(
            response_url, _ephemeral("Nothing to remember — that message has no text.")
        )
        return

    channel_name: Optional[str] = (payload.get("channel") or {}).get("name")
    author_id: Optional[str] = message.get("user")

    try:
        await remember_message(
            owner_user_id, text=text, channel_name=channel_name, author_id=author_id
        )
    except EntityNotFoundError:
        logger.error("Slack credential for team %s points at a deleted user", team_id)
        await post_to_response_url(
            response_url,
            _ephemeral(
                "Slack integration is not fully configured. Please disconnect and reconnect."
            ),
        )
        return
    except Exception:  # noqa: BLE001 - any remember failure must degrade to a chat message, not a crash
        logger.exception("Failed to remember a Slack message for team %s", team_id)
        await post_to_response_url(
            response_url, _ephemeral("Could not save that message. Please try again.")
        )
        return

    await post_to_response_url(response_url, _ephemeral("Saved to Cognee memory."))
