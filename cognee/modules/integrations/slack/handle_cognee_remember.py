"""Handle ``/cognee-remember`` slash command — save free text to memory.

There are two ways to remember, and they answer different needs. The
"Remember this" message shortcut covers what Slack already has on screen:
the thing worth keeping was written by someone else, and retyping it is
exactly the friction that stops anyone bothering. This command covers what
Slack never saw — a decision from a call, a conclusion someone is recording
after the fact.

Slack's 3-second ack budget applies here as much as it does to
``/cognee-ask``. ``cognee.remember`` resolves the target dataset before it
returns even with ``run_in_background=True``, and on a first save that
dataset does not exist yet, so the write runs detached and its confirmation
arrives afterwards via ``response_url``.

Every reply is ``ephemeral``. What someone chooses to commit to memory is
between them and their graph — announcing it to the channel is not this
app's call to make, the same reasoning that keeps ``/cognee-ask`` private
until its asker clicks Share.
"""

import asyncio
import logging
from typing import Any
from urllib.parse import parse_qs
from uuid import UUID

from cognee.modules.integrations.slack.handle_slack_link import NOT_LINKED_MESSAGE
from cognee.modules.integrations.slack.persistence import (
    get_by_team,
    is_active,
    resolve_owner_user_id,
)
from cognee.modules.integrations.slack.remember_message import remember_note
from cognee.modules.integrations.slack.response_url import post_to_response_url

logger = logging.getLogger(__name__)

# asyncio.create_task() only holds a weak reference to its result — without a
# strong one the save can be garbage-collected mid-flight the instant this
# handler returns. Each task removes itself on completion. Same reasoning
# (and same shape) as _pending_searches in handle_cognee_ask.
_pending_saves: set = set()

# Keep the ack readable even if someone pastes an essay.
_NOTE_PREVIEW_MAX_CHARS = 200

NOT_CONNECTED_MESSAGE = (
    "This Slack workspace is not connected to Cognee. Connect it from your Integrations settings."
)

USAGE_MESSAGE = (
    "Tell me what to remember — e.g. "
    "`/cognee-remember we chose Neon for v2, branching is cheaper`.\n"
    "To remember something already in Slack, use *Remember this* from a "
    "message's ⋯ menu instead."
)

# "Recallable shortly", not "saved": ingestion keeps running after the write
# returns, so asking for it back this second may legitimately find nothing.
REMEMBERED_MESSAGE = "✅ Remembered — it'll be recallable shortly."

SAVE_FAILED_MESSAGE = "Could not save that. Please try again."


def _ephemeral(text: str, *, replace_original: bool = False) -> dict[str, Any]:
    """Build a reply visible only to the person who ran the command."""
    payload: dict[str, Any] = {"response_type": "ephemeral", "text": text}
    if replace_original:
        payload["replace_original"] = True
    return payload


def _preview(text: str) -> str:
    if len(text) <= _NOTE_PREVIEW_MAX_CHARS:
        return text
    return text[:_NOTE_PREVIEW_MAX_CHARS] + "…"


async def handle_cognee_remember(raw_body: bytes) -> dict[str, Any]:
    """Ack ``/cognee-remember {text}`` within Slack's 3-second window.

    The save itself is not this function's return value — it runs detached
    and confirms via ``response_url``.
    """
    form = parse_qs(raw_body.decode())
    team_id = (form.get("team_id") or [""])[0]
    invoking_slack_user_id = (form.get("user_id") or [""])[0]
    text = (form.get("text") or [""])[0].strip()
    response_url = (form.get("response_url") or [""])[0]

    credential = await get_by_team(team_id) if team_id else None
    if not is_active(credential):
        return _ephemeral(NOT_CONNECTED_MESSAGE)

    # Which cognee account this lands in has to be settled before anything is
    # written — a note saved into the installer's memory because the author
    # never linked is worse than one refused with a reason.
    owner_user_id = await resolve_owner_user_id(credential, team_id, invoking_slack_user_id)
    if owner_user_id is None:
        return _ephemeral(NOT_LINKED_MESSAGE)

    if not text:
        return _ephemeral(USAGE_MESSAGE)

    task = asyncio.create_task(
        _remember_and_confirm(
            response_url,
            team_id=team_id,
            user_id=owner_user_id,
            text=text,
            slack_user_id=invoking_slack_user_id,
        )
    )
    _pending_saves.add(task)
    task.add_done_callback(_pending_saves.discard)

    return _ephemeral(f"💾 Remembering: _{_preview(text)}_")


async def _remember_and_confirm(
    response_url: str,
    *,
    team_id: str,
    user_id: UUID,
    text: str,
    slack_user_id: str,
) -> None:
    """Write the note and report the outcome via ``response_url``.

    Never raises — this runs detached from the request/response cycle, so
    there is no caller left to catch anything; a failure has to reach the
    person as a message or it reaches them as silence.
    """
    try:
        await remember_note(user_id, text=text, author_id=slack_user_id)
    except Exception:  # noqa: BLE001 - a save failure must degrade to a chat message, not a crash
        logger.exception("Remember failed for Slack team %s", team_id)
        await post_to_response_url(
            response_url, _ephemeral(SAVE_FAILED_MESSAGE, replace_original=True)
        )
        return

    await post_to_response_url(response_url, _ephemeral(REMEMBERED_MESSAGE, replace_original=True))
