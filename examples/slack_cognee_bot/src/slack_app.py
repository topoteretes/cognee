"""Slack Bolt app + event handlers for the cognee memory bot (issue #3609).

Wires Slack (Socket Mode) onto the Commit-3 :class:`IngestionBuffer`:

* ``message`` events in opted-in channels are silently ingested (with each
  message's permalink resolved via ``chat.getPermalink``);
* ``app_mention`` ("@cognee …") and the ``/recall`` slash command answer from
  memory, flushing pending messages first.

slack_bolt import isolation
---------------------------
``slack_bolt`` is an **example-only** dependency and is NOT part of the cognee
core install. So this module must import cleanly when slack_bolt is absent — the
import is therefore deferred into :func:`build_app` / :func:`start_socket_mode`
(guarded by :func:`_load_bolt`, which raises a clear "install the slack extra"
message). The event-handling *logic* lives in plain async functions
(:func:`handle_message_event`, :func:`handle_app_mention`,
:func:`handle_recall_command`) that take a mockable Slack client/say — no
slack_bolt needed to import or unit-test them.

Out of scope here (later commits): rich Block Kit citation rendering (commit 5 —
see the seam in :func:`format_reply`); the ``/forget`` command and opt-in/opt-out
management (commit 6). This commit wires ingestion, @mention, and /recall only.
"""

from __future__ import annotations

import re
from typing import Any

from src.config import SlackSettings
from src.ingestion_buffer import IngestionBuffer
from src.memory_adapter import Answer, ConversationRef

# Matches a leading Slack user mention token like "<@U12345>" so we can strip
# the "@cognee" prefix off an app_mention to get the bare question.
_MENTION_TOKEN = re.compile(r"<@[A-Z0-9]+>")

_NO_ANSWER_TEXT = "I couldn't find anything about that in this channel's memory yet."


def _load_bolt():
    """Import slack_bolt lazily, with an actionable error if it's missing."""
    try:
        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
        from slack_bolt.async_app import AsyncApp
    except ImportError as error:  # pragma: no cover - trivial re-raise
        raise ImportError(
            "slack_bolt is required to run the Slack bot. Install the example's "
            "dependencies with `pip install -e examples/slack_cognee_bot` "
            "(or `pip install slack-bolt`)."
        ) from error
    return AsyncApp, AsyncSocketModeHandler


# --------------------------------------------------------------------------- #
# Pure handler logic (slack_bolt-agnostic, unit-testable with mocks)          #
# --------------------------------------------------------------------------- #


def should_ingest(event: dict, opted_in: set[str], bot_user_id: str | None) -> bool:
    """Decide whether a ``message`` event should be ingested.

    Skips: message subtypes (edits ``message_changed``, deletions, joins, and
    ``bot_message``), any bot-authored message, the bot's own messages, empty
    text, and channels the bot hasn't been opted into.
    """
    if event.get("subtype"):
        return False
    if event.get("bot_id"):
        return False
    if bot_user_id and event.get("user") == bot_user_id:
        return False
    if not (event.get("text") or "").strip():
        return False
    if event.get("channel") not in opted_in:
        return False
    return True


def _conversation_ref(event: dict, default_team_id: str) -> ConversationRef:
    return ConversationRef(
        team_id=event.get("team") or default_team_id,
        channel_id=event["channel"],
        thread_ts=event.get("thread_ts"),
    )


async def _resolve_permalink(client: Any, channel: str, ts: str) -> str:
    """Resolve a message permalink, degrading to "" if Slack can't provide one.

    A blank permalink is handled downstream (the citation renderer falls back to
    plain text), so a getPermalink failure must never drop the message.
    """
    try:
        response = await client.chat_getPermalink(channel=channel, message_ts=ts)
        return response.get("permalink", "") or ""
    except Exception:  # noqa: BLE001 - external SDK call; never fail ingestion
        return ""


async def handle_message_event(
    event: dict,
    client: Any,
    buffer: IngestionBuffer,
    *,
    opted_in: set[str],
    bot_user_id: str | None = None,
    default_team_id: str = "",
) -> bool:
    """Ingest a channel message (permalink resolved). Returns True if ingested."""
    if not should_ingest(event, opted_in, bot_user_id):
        return False

    channel = event["channel"]
    ts = event["ts"]
    permalink = await _resolve_permalink(client, channel, ts)
    await buffer.add_message(
        _conversation_ref(event, default_team_id),
        ts=ts,
        text=event.get("text", ""),
        permalink=permalink,
        author=event.get("user", ""),
    )
    return True


def format_reply(answer: Answer) -> str:
    """Render an :class:`Answer` to a Slack message string.

    NOTE (commit 5 seam): this is the minimal plain/mrkdwn renderer. Commit 5
    replaces it with the rich Block Kit citations renderer (deduped Sources
    context block, stale-permalink fallback). The answer handlers already carry
    ``answer.citations``, so the swap is renderer-only.
    """
    lines = [answer.text or _NO_ANSWER_TEXT]
    if answer.citations:
        lines.append("")
        lines.append("Sources:")
        for cite in answer.citations:
            if cite.ok and cite.permalink:
                lines.append(f"• <{cite.permalink}|{cite.author or 'message'}>")
            else:
                lines.append(f"• {cite.snippet or 'source'}")
    return "\n".join(lines)


async def _answer_and_reply(
    ref: ConversationRef,
    question: str,
    buffer: IngestionBuffer,
    say: Any,
) -> None:
    answer = await buffer.answer(ref, query=question)
    await say(format_reply(answer))


async def handle_app_mention(
    event: dict,
    say: Any,
    buffer: IngestionBuffer,
    *,
    default_team_id: str = "",
) -> None:
    """Answer an "@cognee …" mention from the channel's memory."""
    question = _MENTION_TOKEN.sub("", event.get("text", "")).strip()
    await _answer_and_reply(_conversation_ref(event, default_team_id), question, buffer, say)


async def handle_recall_command(
    command: dict,
    ack: Any,
    say: Any,
    buffer: IngestionBuffer,
    *,
    default_team_id: str = "",
) -> None:
    """Answer a ``/recall <question>`` slash command."""
    await ack()
    question = (command.get("text") or "").strip()
    ref = ConversationRef(
        team_id=command.get("team_id") or default_team_id,
        channel_id=command["channel_id"],
    )
    await _answer_and_reply(ref, question, buffer, say)


# --------------------------------------------------------------------------- #
# Bolt wiring (imports slack_bolt lazily)                                      #
# --------------------------------------------------------------------------- #


def build_app(
    buffer: IngestionBuffer,
    settings: SlackSettings,
    opted_in: set[str],
    *,
    bot_user_id: str | None = None,
):
    """Construct the Bolt ``AsyncApp`` and register the handlers.

    ``opted_in`` is a live set (commit 6's opt-out command mutates it).
    """
    AsyncApp, _ = _load_bolt()
    app = AsyncApp(token=settings.bot_token)
    default_team_id = settings.default_team_id

    @app.event("message")
    async def _on_message(event, client):
        await handle_message_event(
            event,
            client,
            buffer,
            opted_in=opted_in,
            bot_user_id=bot_user_id,
            default_team_id=default_team_id,
        )

    @app.event("app_mention")
    async def _on_app_mention(event, say):
        await handle_app_mention(event, say, buffer, default_team_id=default_team_id)

    @app.command("/recall")
    async def _on_recall(ack, command, say):
        await handle_recall_command(command, ack, say, buffer, default_team_id=default_team_id)

    return app


async def start_socket_mode(app, settings: SlackSettings) -> None:
    """Start the app over Socket Mode (no public URL needed)."""
    _, AsyncSocketModeHandler = _load_bolt()
    handler = AsyncSocketModeHandler(app, settings.app_token)
    await handler.start_async()
