"""``@cognee``: the one way a member talks to Cognee from Slack.

A mention in a channel the app is in (``app_mention``) or a direct message to the app
(``message`` with ``channel_type: im``) carries one of:

* ``@cognee <question>``: answered from the member's memory, with Share and Discard;
* ``@cognee remember <text>``: saves a note that Slack never saw (a decision from a call);
* ``@cognee remember`` in a thread: saves the thread's first message, followed like
  "Remember this" (an edit or a deletion in Slack forgets the copy);
* ``@cognee forget`` in a thread: forgets the member's own copy of the thread's first
  message (the "Forget this" shortcut does the same for any message);
* ``@cognee here <question>``: the same, within this channel's synced messages only;
* ``@cognee link``: the page where the member links their Slack account (Sign in with
  Slack, ``member_link.py``); ``@cognee unlink`` ends that link;
* ``@cognee status``: whether this channel is synced, the last sync, and which Cognee
  account the member is linked to;
* ``@cognee help`` or a bare mention: what the app does.

Every answer is ephemeral (``chat.postEphemeral``): only the member who asked sees it. The
mention itself is visible in the channel, so a private question goes in a direct message.
A direct message is always allowed; in a channel the command allowlist applies, as for
"Remember this", except for ``forget``, ``unlink`` and ``status``: taking back one's
own copy or link, and asking what Cognee does here, are never limited. The event handler acknowledges Slack at once and runs this detached
(``handle_slack_event``), so a slow search never earns a retry.
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any
from uuid import UUID

import aiohttp

from cognee.api.v1.search.search import search as cognee_search
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.integrations import ingestion
from cognee.modules.integrations.credentials import (
    REVOKED_BY_OWNER,
    get_credential_by_account,
    revoke_credential_by_account,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.outbound import client_session, require_outbound_http
from cognee.modules.integrations.slack.adapter import access_token_for
from cognee.modules.integrations.slack.channel_membership import (
    app_follows_channel,
    member_channels,
)
from cognee.modules.integrations.slack.handle_cognee_ask import (
    ASK_DISCARD_ACTION_ID,
    ASK_SHARE_ACTION_ID,
    _format_answer,
    ask_session_id,
    keep_answer,
)
from cognee.modules.integrations.slack.handle_slack_link import (
    MEMBER_LINK_PROVIDER,
    NOT_LINKED_MESSAGE,
    link_message,
    member_link_account_id,
)
from cognee.modules.integrations.slack.persistence import (
    CHANNEL_NOT_ENABLED_MESSAGE,
    PROVIDER,
    channel_allowed,
    get_by_team,
    is_active,
    resolve_owner_user_id,
)
from cognee.modules.integrations.slack.remember_message import (
    forget_my_remembered_message,
    remember_message,
    remember_note,
)
from cognee.modules.integrations.slack.sync import (
    dataset_name_for_credential,
    exclude_message,
    selected_channels,
)
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_user
from cognee.tasks.ingestion.connectors.slack import SlackSourceError

logger = logging.getLogger(__name__)

_POST_EPHEMERAL_URL = "https://slack.com/api/chat.postEphemeral"
_TIMEOUT = aiohttp.ClientTimeout(total=10)
_PREVIEW_MAX_CHARS = 200
_ANY_LEADING_MENTION = re.compile(r"^\s*<@[UW][A-Z0-9]+>")

# Every reply is a short bold title on its first line and one plain explanation below it
# (``notice``), in cognee's product voice: direct, no emoji (Goran, 2026-11-10).
HELP_MESSAGE = (
    "Cognee in Slack\n"
    "• `@cognee <question>` answers from your Cognee memory.\n"
    "• `@cognee here <question>` answers from this channel's synced messages only.\n"
    "• `@cognee remember <text>` saves a note, e.g. a decision from a call.\n"
    "• `@cognee remember` in a thread saves the thread's first message.\n"
    "• `@cognee forget` in a thread forgets your saved copy of the thread's first message; "
    "its author or the person who connected Slack also removes it from the synced memory.\n"
    "• `@cognee link` links your Slack account to your Cognee account; "
    "`@cognee unlink` ends that.\n"
    "• `@cognee status` says whether this channel is synced and who you are linked as.\n"
    "Only you see these replies. *Remember this* and *Forget this* are in every message's "
    "⋯ menu."
)
REMEMBERED_MESSAGE = (
    "Saved to your memory\n"
    "Recallable in about a minute. Reply `@cognee forget` in this thread to remove it."
)
SAVE_FAILED_MESSAGE = "Couldn't save that\nTry again in a moment."
SAVED_MESSAGE = (
    "Saved to your memory\nIf the message is edited or deleted in Slack, Cognee forgets its copy."
)
NOT_FOLLOWABLE_MESSAGE = (
    "Add Cognee to this channel first\n"
    "Cognee saves messages only from channels it is in, so it can forget its copy when a "
    "message is edited or deleted."
)
NOTHING_TO_SAVE_MESSAGE = "Nothing to save\nThat message has no text."
FORGOTTEN_MESSAGE = "Forgotten\nCognee no longer has your copy of this message."
NOTHING_TO_FORGET_MESSAGE = "Nothing to forget\nYou have no saved copy of this message."
FORGET_FAILED_MESSAGE = "Couldn't forget that\nTry again in a moment."
EXCLUDED_MESSAGE = (
    "Removed from workspace memory\n"
    "Cognee leaves this message out from now on. Answers drop it once the sync that just "
    "started finishes."
)
NOT_YOURS_TO_EXCLUDE_MESSAGE = (
    "Still in workspace memory\n"
    "Only its author or the person who connected Slack can remove it. Any copy you saved "
    "yourself is gone."
)
FORGET_WHERE_MESSAGE = (
    "Use forget in a thread\n"
    "Reply `@cognee forget` in the message's thread, or choose *Forget this* in its ⋯ menu."
)
UNLINKED_MESSAGE = (
    "Unlinked\n"
    "Your questions and saved messages no longer use that Cognee account. Send "
    "`@cognee link` to link again."
)
NOT_LINKED_TO_UNLINK_MESSAGE = "Not linked\nYour Slack account is not linked to a Cognee account."
HERE_NEEDS_A_CHANNEL_MESSAGE = (
    "Use here in a channel\n`@cognee here` searches one channel's synced messages."
)
NOT_CONFIGURED_MESSAGE = (
    "Slack needs reconnecting\n"
    "The connection to Cognee is not complete. Disconnect Slack and connect it again."
)
SEARCH_FAILED_MESSAGE = "Search failed\nTry again in a moment."
NOTHING_FOUND_MESSAGE = (
    "Nothing found\nYour Cognee memory has nothing on that yet. `@cognee remember` saves it."
)
SHARE_UNAVAILABLE_TEXT = (
    "Sharing is off here: it needs Cognee's session memory (CACHING), which this server "
    "has switched off."
)
MEMBERSHIP_UNKNOWN_MESSAGE = (
    "Synced channels left out\n"
    "Slack did not say which channels you are in, so this answer does not use the "
    "workspace's synced channels. Try again in a moment."
)
NOT_CONNECTED_MESSAGE = (
    "Slack is not connected\nConnect this workspace from Integrations in Cognee."
)

Reply = Callable[[dict[str, Any]], Awaitable[None]]

# Taking back one's own copy or link, and asking what Cognee does here, work everywhere.
_NEVER_LIMITED = {"forget", "unlink", "status"}


def notice(text: str, *, muted: bool = False, **extra: Any) -> dict[str, Any]:
    """A reply only the member sees. ``text`` is its title, a line break, its explanation;
    it is also the plain fallback for notifications. ``muted`` renders one grey line
    (progress). An explicit ``blocks`` (an answer) wins."""
    title, _, body = text.partition("\n")
    if muted:
        blocks = [{"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}]
    else:
        section = f"*{title}*" + (f"\n{body}" if body else "")
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": section}}]
    return {"response_type": "ephemeral", "text": text, "blocks": blocks, **extra}


def _preview(text: str) -> str:
    return text if len(text) <= _PREVIEW_MAX_CHARS else text[:_PREVIEW_MAX_CHARS] + "…"


def is_direct_message(event: dict[str, Any]) -> bool:
    """A member's own message in their DM with the app; the app's own messages are not."""
    return (
        event.get("channel_type") == "im"
        and not event.get("subtype")
        and not event.get("bot_id")
        and bool(event.get("user"))
    )


async def handle_cognee_mention(team_id: str, event: dict[str, Any]) -> None:
    """Answer one ``@cognee`` mention or direct message. Never raises: it runs detached."""
    try:
        await _handle(team_id, event)
    except Exception:  # detached: a failure is logged, never an unhandled task error
        logger.exception("Answering a Slack mention failed for team %s", team_id)


async def _handle(team_id: str, event: dict[str, Any]) -> None:
    credential = await get_by_team(team_id) if team_id else None
    if not is_active(credential):
        return  # no token to answer with
    channel, member = event.get("channel", ""), event.get("user", "")
    # A mention in a thread is answered in that thread, where the member is looking.
    reply = partial(post_ephemeral, credential, channel, member, thread_ts=event.get("thread_ts"))
    direct = event.get("channel_type") == "im"
    text = _without_the_mention(event.get("text") or "", credential)
    word, _, rest = text.partition(" ")
    command = word.lower()
    if command not in _NEVER_LIMITED and not direct and not channel_allowed(credential, channel):
        await reply(notice(CHANNEL_NOT_ENABLED_MESSAGE))
        return
    if command == "link":
        await reply(notice(link_message()))
        return
    if not text or command == "help":
        await reply(notice(HELP_MESSAGE))
        return

    if command == "unlink":
        await unlink(reply, team_id=team_id, member=member)
        return

    owner_user_id = await resolve_owner_user_id(credential, team_id, member)
    if command == "status":
        await reply(
            notice(
                await status_text(credential, owner_user_id, channel=None if direct else channel)
            )
        )
        return
    if owner_user_id is None:
        await reply(notice(NOT_LINKED_MESSAGE))
        return

    if command == "remember":
        note = rest.strip()
        if note:
            await remember_text(
                reply,
                team_id=team_id,
                owner_user_id=owner_user_id,
                text=note,
                author_id=member,
                channel_id=channel,
                message_ts=event.get("ts", ""),
            )
        elif event.get("thread_ts") and not direct:
            await remember_thread(
                reply, credential, team_id=team_id, owner_user_id=owner_user_id, event=event
            )
        else:
            await reply(notice(HELP_MESSAGE))
        return

    if command == "forget":
        # The thread's first message: a remembered message, a note made in a DM or a
        # channel, or a synced message to take out of the workspace's memory.
        if event.get("thread_ts"):
            await forget_and_confirm(
                reply,
                credential,
                owner_user_id=owner_user_id,
                team_id=team_id,
                channel_id=channel,
                message_ts=event["thread_ts"],
                root_ts=event["thread_ts"],
                member=member,
            )
        else:
            await reply(notice(FORGET_WHERE_MESSAGE))
        return

    if command == "here":
        question = rest.strip()
        if direct:
            await reply(notice(HERE_NEEDS_A_CHANNEL_MESSAGE))
        elif not question:
            await reply(notice(HELP_MESSAGE))
        else:
            await ask(
                reply,
                credential,
                member=member,
                owner_user_id=owner_user_id,
                question=question,
                here=channel,
            )
        return

    await ask(reply, credential, member=member, owner_user_id=owner_user_id, question=text)


def _without_the_mention(text: str, credential: IntegrationCredential) -> str:
    bot = (credential.provider_metadata or {}).get("bot_user_id")
    stripped = text.replace(f"<@{bot}>", " ") if bot else _ANY_LEADING_MENTION.sub(" ", text)
    return " ".join(stripped.split())


async def ask(
    reply: Reply,
    credential: IntegrationCredential,
    *,
    member: str,
    owner_user_id: UUID,
    question: str,
    here: str | None = None,
) -> None:
    """Answer ``question`` from the owner's memory, privately, with Share and Discard.

    Every selected channel lands in one workspace dataset, which its owner may share, so
    that dataset is searched only within the selected channels Slack says ``member``
    belongs to; the owner's other datasets are searched whole. ``here`` (``@cognee here``)
    narrows both to that one channel. When Slack cannot say which channels the member is
    in, the workspace dataset is left out and the member is told.
    """
    team_id = str(credential.provider_account_id)
    await reply(notice(f"Searching your memory for “{_preview(question)}”", muted=True))
    try:
        owner = await get_user(owner_user_id)
    except EntityNotFoundError:
        logger.error("Slack credential for team %s points at a deleted user", team_id)
        await reply(notice(NOT_CONFIGURED_MESSAGE))
        return
    try:
        channels = await member_channels(credential, member)
    except SlackSourceError as error:
        logger.warning("Slack did not list a member's channels for team %s: %s", team_id, error)
        channels = set()
        await reply(notice(MEMBERSHIP_UNKNOWN_MESSAGE))
    scope = [here] if here else selected_channels(credential)
    visible = [channel for channel in scope if channel in channels]

    workspace_name = dataset_name_for_credential(credential)
    readable = await get_authorized_existing_datasets(None, "read", owner)
    # A previous owner's copy of the workspace carries the same name (G44): scoped too.
    workspace = [dataset.id for dataset in readable if dataset.name == workspace_name]
    others = [dataset.id for dataset in readable if dataset.name != workspace_name]
    searches = []
    if workspace and visible:
        searches.append((workspace, [f"slack:channel:{channel}" for channel in visible]))
    if others:
        searches.append((others, [f"slack:channel:{here}"] if here else None))
    try:
        answers = await asyncio.gather(
            *(
                cognee_search(
                    query_text=question,
                    # Graph traversal alone can come back empty on a thin graph, where the
                    # vector-search half still finds a relevant chunk.
                    query_type=SearchType.HYBRID_COMPLETION,
                    user=owner,
                    dataset_ids=dataset_ids,
                    node_name=node_name,
                    # G10: each result carries the chunks it read (``objects_result``),
                    # whose payloads name their documents by title and link (SDK-1500);
                    # the answer shows them as its sources.
                    include_references=True,
                    verbose=True,
                )
                for dataset_ids, node_name in searches
            )
        )
    except Exception:  # a search failure degrades to a chat message
        logger.exception("Search failed for Slack team %s", team_id)
        await reply(notice(SEARCH_FAILED_MESSAGE))
        return
    results = [result for answer in answers for result in answer]

    fallback_text, blocks = _format_answer(results)
    if not blocks:
        # Every dataset answered that it does not know: nothing to share.
        await reply(notice(NOTHING_FOUND_MESSAGE))
        return
    answer_ref = await keep_answer(owner_user_id, ask_session_id(team_id, member), question, blocks)
    if answer_ref is None:
        # Share reads the answer back from the session cache, which is switched off.
        await reply(
            notice(
                fallback_text,
                blocks=[
                    *blocks,
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": SHARE_UNAVAILABLE_TEXT}],
                    },
                ],
            )
        )
        return
    await reply(
        notice(
            f"{fallback_text}\n\nShare this in the channel?",
            blocks=[
                *blocks,
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Share"},
                            "style": "primary",
                            "action_id": ASK_SHARE_ACTION_ID,
                            "value": answer_ref,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Discard"},
                            "action_id": ASK_DISCARD_ACTION_ID,
                            "value": answer_ref,
                        },
                    ],
                },
            ],
        )
    )


async def remember_text(
    reply: Reply,
    *,
    team_id: str,
    owner_user_id: UUID,
    text: str,
    author_id: str,
    channel_id: str,
    message_ts: str,
) -> None:
    """Save a note Slack never saw into the owner's memory, as the item of the message
    that asked for it (``remember_note``)."""
    try:
        await remember_note(
            owner_user_id,
            text=text,
            team_id=team_id,
            channel_id=channel_id,
            message_ts=message_ts,
            author_id=author_id,
        )
    except Exception:  # a save failure degrades to a chat message
        logger.exception("Remember failed for Slack team %s", team_id)
        await reply(notice(SAVE_FAILED_MESSAGE))
        return
    await reply(notice(REMEMBERED_MESSAGE))


async def remember_thread(
    reply: Reply,
    credential: IntegrationCredential,
    *,
    team_id: str,
    owner_user_id: UUID,
    event: dict[str, Any],
) -> None:
    """Save the first message of the thread the mention is in, like "Remember this"."""
    channel = event.get("channel", "")
    require_outbound_http(PROVIDER)
    client = ingestion.build_service(PROVIDER, await access_token_for(credential))
    page = await asyncio.to_thread(
        client.call, "conversations.replies", channel=channel, ts=event["thread_ts"], limit=1
    )
    root = (page.get("messages") or [{}])[0]
    text = (root.get("text") or "").strip()
    if not text:
        await reply(notice(NOTHING_TO_SAVE_MESSAGE))
        return
    await remember_and_confirm(
        reply,
        credential,
        owner_user_id=owner_user_id,
        text=text,
        team_id=team_id,
        channel_id=channel,
        message_ts=root.get("ts", ""),
        channel_name=None,
        author_id=root.get("user"),
    )


async def remember_and_confirm(
    reply: Reply,
    credential: IntegrationCredential,
    *,
    owner_user_id: UUID,
    text: str,
    team_id: str,
    channel_id: str,
    message_ts: str,
    channel_name: str | None,
    author_id: str | None,
) -> None:
    """Remember a Slack message if Cognee can keep its copy true, and say how it went.

    Shared by "Remember this" and ``@cognee remember`` in a thread. Never raises.
    """
    try:
        if not await app_follows_channel(credential, channel_id):
            await reply(notice(NOT_FOLLOWABLE_MESSAGE))
            return
        await remember_message(
            owner_user_id,
            text=text,
            team_id=team_id,
            channel_id=channel_id,
            message_ts=message_ts,
            channel_name=channel_name,
            author_id=author_id,
        )
    except EntityNotFoundError:
        logger.error("Slack credential for team %s points at a deleted user", team_id)
        await reply(notice(NOT_CONFIGURED_MESSAGE))
        return
    except Exception:  # any remember failure degrades to a chat message
        logger.exception("Failed to remember a Slack message for team %s", team_id)
        await reply(notice(SAVE_FAILED_MESSAGE))
        return
    await reply(notice(SAVED_MESSAGE))


async def unlink(reply: Reply, *, team_id: str, member: str) -> None:
    """End the member's own link to a Cognee account."""
    account = member_link_account_id(team_id, member)
    link = await get_credential_by_account(MEMBER_LINK_PROVIDER, account)
    if not is_active(link):
        await reply(notice(NOT_LINKED_TO_UNLINK_MESSAGE))
        return
    await revoke_credential_by_account(MEMBER_LINK_PROVIDER, account, revoked_by=REVOKED_BY_OWNER)
    await reply(notice(UNLINKED_MESSAGE))


async def status_text(
    credential: IntegrationCredential, owner_user_id: UUID | None, *, channel: str | None
) -> str:
    """Who the member is to Cognee, and what Cognee does with ``channel``."""
    lines = ["Cognee status"]
    if owner_user_id is None:
        lines.append(
            "Your Slack account is not linked to a Cognee account yet: send `@cognee link`."
        )
    else:
        try:
            lines.append(f"You are linked to Cognee as {(await get_user(owner_user_id)).email}.")
        except EntityNotFoundError:
            lines.append(NOT_CONFIGURED_MESSAGE)
    if channel is not None:
        if channel in selected_channels(credential):
            lines.append("This channel is synced into the workspace's Cognee memory.")
        else:
            lines.append("This channel is not synced into Cognee memory.")
        if not channel_allowed(credential, channel):
            lines.append(CHANNEL_NOT_ENABLED_MESSAGE)
    last = credential.last_synced_at
    if last is None:
        lines.append("No sync has run yet.")
    else:
        status = f" ({credential.sync_status})" if credential.sync_status else ""
        lines.append(f"Last sync: {last:%Y-%m-%d %H:%M} UTC{status}.")
    return "\n".join(lines)


async def forget_and_confirm(
    reply: Reply,
    credential: IntegrationCredential,
    *,
    owner_user_id: UUID,
    team_id: str,
    channel_id: str,
    message_ts: str,
    root_ts: str,
    member: str,
    author_id: str | None = None,
) -> None:
    """Forget one message for the member and say how it went. Never raises.

    Shared by ``@cognee forget`` in a thread and the "Forget this" shortcut. The member's
    own saved copy (a remembered message or a note) is always forgotten. A message of a
    synced channel is also taken out of the workspace's memory when its author or the
    connection's owner asks (Goran, 2026-11-10); anyone else is told who can.
    ``author_id`` is the message's Slack author when the caller has it; otherwise it is
    read from Slack only if the member is not the owner.
    """
    try:
        forgotten = await forget_my_remembered_message(
            owner_user_id, team_id=team_id, channel_id=channel_id, message_ts=message_ts
        )
        if channel_id not in selected_channels(credential):
            await reply(notice(FORGOTTEN_MESSAGE if forgotten else NOTHING_TO_FORGET_MESSAGE))
            return
        is_owner = str(owner_user_id) == str(credential.user_id)
        if not is_owner and author_id is None:
            author_id = await _author_of(credential, channel_id, message_ts)
        if not (is_owner or author_id == member):
            await reply(notice(NOT_YOURS_TO_EXCLUDE_MESSAGE))
            return
        await exclude_message(
            credential, channel_id=channel_id, message_ts=message_ts, root_ts=root_ts
        )
    except Exception:  # a failure degrades to a chat message
        logger.exception("Failed to forget a Slack message for team %s", team_id)
        await reply(notice(FORGET_FAILED_MESSAGE))
        return
    await reply(notice(EXCLUDED_MESSAGE))


async def _author_of(credential: IntegrationCredential, channel_id: str, message_ts: str) -> str:
    """The Slack user who wrote the message at ``message_ts`` (a thread's first message)."""
    require_outbound_http(PROVIDER)
    client = ingestion.build_service(PROVIDER, await access_token_for(credential))
    page = await asyncio.to_thread(
        client.call, "conversations.replies", channel=channel_id, ts=message_ts, limit=1
    )
    return ((page.get("messages") or [{}])[0]).get("user", "")


async def post_ephemeral(
    credential: IntegrationCredential,
    channel: str,
    user: str,
    payload: dict[str, Any],
    thread_ts: str | None = None,
) -> None:
    """Show ``payload`` to ``user`` alone in ``channel``, in the thread ``thread_ts`` when
    given. Logs a failure, never raises."""
    body = {"channel": channel, "user": user, "text": payload["text"]}
    if payload.get("blocks"):
        body["blocks"] = payload["blocks"]
    if thread_ts:
        body["thread_ts"] = thread_ts
    try:
        token = await access_token_for(credential)
        async with (
            client_session(PROVIDER, timeout=_TIMEOUT) as session,
            session.post(
                _POST_EPHEMERAL_URL,
                json=body,
                headers={"Authorization": f"Bearer {token}"},
                # A Location header must not send the bot token elsewhere (SDK-832).
                allow_redirects=False,
            ) as response,
        ):
            if response.status != 200:
                raise RuntimeError(f"Slack chat.postEphemeral failed: HTTP {response.status}")
            answer = await response.json()
        if not answer.get("ok"):
            logger.warning("Slack chat.postEphemeral failed: %s", answer.get("error", "unknown"))
    except Exception:  # an undelivered reply must not crash the detached handler
        logger.exception(
            "Slack chat.postEphemeral failed for team %s", credential.provider_account_id
        )
