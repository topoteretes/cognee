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

Out of scope here (later commits): the ``/forget`` command and opt-in/opt-out
management (commit 6). This commit wires ingestion, @mention, and /recall; the
reply is rendered by the Block Kit citation renderer (:mod:`src.citations`).
"""

from __future__ import annotations

import re
from typing import Any

from src.citations import notification_text, render_answer
from src.config import SlackSettings
from src.ingestion_buffer import IngestionBuffer
from src.memory_adapter import ConversationRef

# Matches a leading Slack user mention token like "<@U12345>" so we can strip
# the "@cognee" prefix off an app_mention to get the bare question.
_MENTION_TOKEN = re.compile(r"<@[A-Z0-9]+>")


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
    ``bot_message``), any bot-authored message, the bot's own messages, messages
    that @mention the bot, empty text, and channels not opted into.

    The @mention skip matters because Slack delivers a plain ``message`` event
    *alongside* the ``app_mention`` for the same text; that question is answered
    by the mention handler, so ingesting it too would feed the bot's own
    questions back into the channel's memory.
    """
    if event.get("subtype"):
        return False
    if event.get("bot_id"):
        return False
    text = (event.get("text") or "").strip()
    if not text:
        return False
    if bot_user_id and (event.get("user") == bot_user_id or f"<@{bot_user_id}>" in text):
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


_ERROR_REPLY = (
    "Sorry — I hit an error answering that. Please try again; check the bot logs if it persists."
)


async def _answer_and_reply(
    ref: ConversationRef,
    question: str,
    buffer: IngestionBuffer,
    say: Any,
) -> None:
    # A fresh/empty channel is handled inside answer() (calm empty reply). Any
    # OTHER failure (LLM auth/rate-limit, cognify error) must still get a reply
    # back to the user rather than a silent non-response.
    try:
        answer = await buffer.answer(ref, query=question)
    except Exception:  # noqa: BLE001 - never leave the user without a reply
        await say(_ERROR_REPLY)
        return
    # Rich Block Kit citations reply; text= is the notification/accessibility
    # fallback Slack shows when blocks can't be rendered.
    await say(blocks=render_answer(answer), text=notification_text(answer))


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
# forget + opt-in / opt-out commands                                          #
# --------------------------------------------------------------------------- #

# Deletion is dataset-level (whole channel). cognee has no per-user delete
# (recon: forget() deletes by dataset / data_id / everything — there is no
# per-author scope), so a per-user "forget me" is intentionally NOT offered and
# is left as a follow-up. Do not imply per-user deletion in these replies.
_FORGET_REPLY = (
    ":wastebasket: Deleted this channel's memory — the entire `{dataset}` dataset "
    "(messages, graph, and citations).\n"
    "_Note: cognee forgets at the channel level. Removing just one person's messages "
    "(\"forget me\") isn't supported yet — that's a follow-up._"
)
_OPTIN_DISCLOSURE = (
    ":wave: I'll now remember messages in this channel — I ingest them into cognee "
    "memory so anyone here can ask *@cognee …* or */recall …* and get cited answers.\n"
    "Run */cognee-optout* to stop, or */cognee-forget* to erase what I've stored."
)
_OPTIN_ALREADY = "This channel is already opted in — I'm already remembering it."
_OPTOUT_REPLY = (
    ":mute: Opted out — I'll stop ingesting new messages in this channel.\n"
    "_Existing memory is kept. Run */cognee-forget* to delete it too._"
)


async def handle_forget_command(
    command: dict,
    ack: Any,
    say: Any,
    buffer: IngestionBuffer,
    *,
    default_team_id: str = "",
) -> None:
    """Delete the current channel's memory (dataset-level forget)."""
    await ack()
    channel_id = command["channel_id"]
    ref = ConversationRef(
        team_id=command.get("team_id") or default_team_id,
        channel_id=channel_id,
    )
    await buffer.forget(ref)
    await say(_FORGET_REPLY.format(dataset=ref.dataset_name))


async def handle_optin_command(
    command: dict,
    ack: Any,
    say: Any,
    opted_in: set[str],
) -> None:
    """Opt a channel in to ingestion; post the disclosure on first opt-in."""
    await ack()
    channel_id = command["channel_id"]
    first_time = channel_id not in opted_in
    opted_in.add(channel_id)
    await say(_OPTIN_DISCLOSURE if first_time else _OPTIN_ALREADY)


async def handle_optout_command(
    command: dict,
    ack: Any,
    say: Any,
    opted_in: set[str],
) -> None:
    """Opt a channel out of ingestion (stops future ingest; keeps existing data)."""
    await ack()
    opted_in.discard(command["channel_id"])
    await say(_OPTOUT_REPLY)


# --------------------------------------------------------------------------- #
# Bolt wiring (imports slack_bolt lazily)                                      #
# --------------------------------------------------------------------------- #


def build_app(
    buffer: IngestionBuffer,
    settings: SlackSettings,
    opted_in: set[str],
):
    """Construct the Bolt ``AsyncApp`` and register the handlers.

    ``opted_in`` is a live set (the opt-in/opt-out commands mutate it). The bot's
    own user id comes from Bolt's ``context`` (populated by its auth middleware),
    so the own-message / self-mention skips work without a manual auth.test.
    """
    AsyncApp, _ = _load_bolt()
    app = AsyncApp(token=settings.bot_token)
    default_team_id = settings.default_team_id

    @app.event("message")
    async def _on_message(event, client, context):
        await handle_message_event(
            event,
            client,
            buffer,
            opted_in=opted_in,
            bot_user_id=context.get("bot_user_id"),
            default_team_id=default_team_id,
        )

    @app.event("app_mention")
    async def _on_app_mention(event, say):
        await handle_app_mention(event, say, buffer, default_team_id=default_team_id)

    @app.command("/recall")
    async def _on_recall(ack, command, say):
        await handle_recall_command(command, ack, say, buffer, default_team_id=default_team_id)

    # forget + opt-in/opt-out all close over the SAME `opted_in` set the message
    # handler reads — a single source of truth, no divergent store.
    @app.command("/cognee-forget")
    async def _on_forget(ack, command, say):
        await handle_forget_command(command, ack, say, buffer, default_team_id=default_team_id)

    @app.command("/cognee-optin")
    async def _on_optin(ack, command, say):
        await handle_optin_command(command, ack, say, opted_in)

    @app.command("/cognee-optout")
    async def _on_optout(ack, command, say):
        await handle_optout_command(command, ack, say, opted_in)

    return app


async def start_socket_mode(app, settings: SlackSettings) -> None:
    """Start the app over Socket Mode (no public URL needed)."""
    _, AsyncSocketModeHandler = _load_bolt()
    handler = AsyncSocketModeHandler(app, settings.app_token)
    await handler.start_async()
