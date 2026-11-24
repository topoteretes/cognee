"""``@cognee``: the one way a member talks to Cognee from Slack.

Copied from cognee.modules.integrations.slack.handle_cognee_mention (cognee dev, 2026-11-12).

A mention in a channel the app is in (``app_mention``) or a direct message to the app
(``message`` with ``channel_type: im``) carries one of:

* ``@cognee <question>``: answered with a message (``chat.postMessage``). In a channel it
  goes in the question's thread, for everyone there, so it uses only what all of them can
  read (``ask``); in a direct message it stays between the member and the app;
* ``@cognee remember <text>``: saves a note that Slack never saw (a decision from a call);
* ``@cognee remember`` in a thread: saves the thread's first message, followed like
  "Remember this" (an edit or a deletion in Slack forgets the copy);
* ``@cognee forget`` in a thread: forgets the member's own copy of the thread's first
  message (the "Forget this" shortcut does the same for any message);
* ``@cognee link``: the page where the member links their Slack account (Sign in with
  Slack, ``member_link.py``); ``@cognee unlink`` ends that link;
* ``@cognee status``: whether this channel is synced, the last sync, and which Cognee
  account the member is linked to;
* ``@cognee help`` or a bare mention: what the app does.

Everything else (progress, a miss, a failure, every other command) is ephemeral
(``chat.postEphemeral``): only the member who asked sees it. A direct message is always
allowed; in a channel the command allowlist applies, as for "Remember this", except for
``forget``, ``unlink`` and ``status``: taking back one's own copy or link, and asking what
Cognee does here, are never limited. The event handler acknowledges Slack at once and runs
this detached (``handle_slack_event``), so a slow search never earns a retry.
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
from cognee.modules.retrieval.hybrid.context import extract_context_object_ids
from cognee.modules.retrieval.utils.used_graph_elements import extract_from_scored_results
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_user

# This package's source and client, not ``ingestion.build_service`` / ``source_factory``:
# on a cognee that bundles Slack those resolve to its own copy, whose errors are not the
# classes this module catches.
from cognee_connector_slack import source as slack_source_module
from cognee_connector_slack.cognee_host.adapter import access_token_for
from cognee_connector_slack.cognee_host.channel_membership import (
    app_follows_channel,
    channel_scope,
    member_channels,
)
from cognee_connector_slack.cognee_host.handle_cognee_ask import (
    _format_answer,
    finish_answer,
    record_answer,
)
from cognee_connector_slack.cognee_host.handle_slack_link import (
    MEMBER_LINK_PROVIDER,
    NOT_LINKED_MESSAGE,
    link_message,
    member_link_account_id,
)
from cognee_connector_slack.cognee_host.persistence import (
    CHANNEL_NOT_ENABLED_MESSAGE,
    PROVIDER,
    channel_allowed,
    get_by_team,
    is_active,
    resolve_owner_user_id,
)
from cognee_connector_slack.cognee_host.remember_message import (
    forget_my_remembered_message,
    remember_message,
    remember_note,
)
from cognee_connector_slack.cognee_host.status import status_message
from cognee_connector_slack.cognee_host.sync import (
    dataset_name_for_credential,
    exclude_message,
    selected_channels,
)
from cognee_connector_slack.source import SlackSourceError

logger = logging.getLogger(__name__)

_SLACK_API = "https://slack.com/api/"
_TIMEOUT = aiohttp.ClientTimeout(total=10)
_PREVIEW_MAX_CHARS = 200
_ANY_LEADING_MENTION = re.compile(r"^\s*<@[UW][A-Z0-9]+>")

# Every reply is a short bold title on its first line and one plain explanation below it
# (``notice``), in cognee's product voice: direct, no emoji (Goran, 2026-11-10).
# What the app does, as label and value pairs per group (Goran, 2026-11-12). Laid out in
# Block Kit by ``help_message``; ``HELP_MESSAGE`` is the same in plain text.
_HELP_LEAD = "Ask, remember and forget from any channel cognee is in."
_HELP_GROUPS = (
    (
        "Ask",
        (
            ("`@cognee <question>`", "An answer in the thread, for everyone in the channel."),
            ("Message cognee directly", "A private answer from all your memory."),
            ("Rate", "The buttons under an answer you asked for."),
        ),
    ),
    (
        "Remember",
        (
            ("`@cognee remember <text>`", "Saves a note."),
            ("`@cognee remember` in a thread", "Saves the thread's first message."),
            ("*Remember this* (⋯ menu)", "Saves any message."),
        ),
    ),
    (
        "Forget",
        (
            ("`@cognee forget` in a thread", "Forgets your saved copy."),
            (
                "*Forget this* (⋯ menu)",
                "Forgets a synced message, for its author or the person who connected Slack.",
            ),
        ),
    ),
)
_HELP_ACCOUNT = "`@cognee link` · `@cognee unlink` · `@cognee status`"
_ONLY_YOU = "Only you can see this."
HELP_MESSAGE = "\n".join(
    [
        "cognee in Slack",
        _HELP_LEAD,
        *(
            f"{group}: " + "; ".join(f"{label}: {value}" for label, value in rows)
            for group, rows in _HELP_GROUPS
        ),
        f"Account: {_HELP_ACCOUNT}",
        _ONLY_YOU,
    ]
).replace("*", "")


def help_message() -> dict[str, Any]:
    """``@cognee help`` in Block Kit: a title, the groups as two columns (label, value)
    between dividers, the account commands, and who sees it. ``HELP_MESSAGE`` is its
    notification text."""
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*cognee in Slack*\n{_HELP_LEAD}"}},
        {"type": "divider"},
    ]
    for group, rows in _HELP_GROUPS:
        if group == "Remember":
            blocks.append({"type": "divider"})  # asking, then what cognee keeps
        blocks += [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{group}*"}},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": text}
                    for label, value in rows
                    for text in (label, value)
                ],
            },
        ]
    blocks += [
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Account*\n{_HELP_ACCOUNT}"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": _ONLY_YOU}]},
    ]
    return notice(HELP_MESSAGE, blocks=blocks)


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
NOT_CONFIGURED_MESSAGE = (
    "Slack needs reconnecting\n"
    "The connection to Cognee is not complete. Disconnect Slack and connect it again."
)
SEARCH_FAILED_MESSAGE = "Search failed\nTry again in a moment."
NOTHING_FOUND_MESSAGE = (
    "Nothing found\nYour Cognee memory has nothing on that yet. `@cognee remember` saves it."
)
ANSWER_NOT_POSTED_MESSAGE = "Couldn't post the answer\nTry again in a moment."
MEMBERSHIP_UNKNOWN_MESSAGE = (
    "Synced channels left out\n"
    "Slack did not say which channels you are in, so this answer does not use the "
    "workspace's synced channels. Try again in a moment."
)
PUBLIC_CHANNELS_UNKNOWN_MESSAGE = (
    "Only this channel used\n"
    "Slack did not say which channels everyone here can read, so this answer uses only "
    "this channel's synced messages. Try again in a moment."
)
NOT_CONNECTED_MESSAGE = (
    "Slack is not connected\nConnect this workspace from Integrations in Cognee."
)

Reply = Callable[[dict[str, Any]], Awaitable[None]]
# Posts an answer as a message in the conversation it was asked in; raises when Slack
# does not take it.
Publish = Callable[[dict[str, Any]], Awaitable[None]]

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
        await reply(help_message())
        return

    if command == "unlink":
        await unlink(reply, team_id=team_id, member=member)
        return

    owner_user_id = await resolve_owner_user_id(credential, team_id, member)
    if command == "status":
        message = await status_message(
            credential, owner_user_id, channel=None if direct else channel, member=member
        )
        await reply(notice(message["text"], blocks=message["blocks"]))
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
            await reply(help_message())
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

    # In a channel the answer starts a thread under the question, or joins the thread it
    # was asked in; in a direct message it is the next message, or joins its thread.
    thread = event.get("thread_ts") if direct else event.get("thread_ts") or event.get("ts")
    await ask(
        reply,
        partial(post_message, credential, channel, thread_ts=thread),
        credential,
        member=member,
        owner_user_id=owner_user_id,
        question=text,
        channel=None if direct else channel,
    )


def _without_the_mention(text: str, credential: IntegrationCredential) -> str:
    bot = (credential.provider_metadata or {}).get("bot_user_id")
    stripped = text.replace(f"<@{bot}>", " ") if bot else _ANY_LEADING_MENTION.sub(" ", text)
    return " ".join(stripped.split())


async def ask(
    reply: Reply,
    publish: Publish,
    credential: IntegrationCredential,
    *,
    member: str,
    owner_user_id: UUID,
    question: str,
    channel: str | None,
) -> None:
    """Answer ``question`` with a message (``publish``); progress, a miss and a failure
    reach the member alone (``reply``).

    Every selected channel lands in one workspace dataset. A question in ``channel`` is
    answered for everyone there, so it reads only what all of them can: the workspace
    dataset within that channel and the selected public channels (``channel_scope``),
    never the asker's other datasets. A direct message (``channel`` is ``None``) is
    answered from the workspace dataset within the selected channels Slack says ``member``
    belongs to, and from the owner's other datasets whole; when Slack cannot say, the
    workspace dataset is left out and the member is told.
    """
    team_id = str(credential.provider_account_id)
    await reply(notice(f"Searching your memory for “{_preview(question)}”", muted=True))
    try:
        owner = await get_user(owner_user_id)
    except EntityNotFoundError:
        logger.error("Slack credential for team %s points at a deleted user", team_id)
        await reply(notice(NOT_CONFIGURED_MESSAGE))
        return
    if channel is None:
        visible = await _direct_message_scope(reply, credential, member)
    else:
        scope = await channel_scope(credential, channel, member)
        if not scope.known:
            await reply(notice(PUBLIC_CHANNELS_UNKNOWN_MESSAGE))
        visible = scope.channels

    workspace_name = dataset_name_for_credential(credential)
    readable = await get_authorized_existing_datasets(None, "read", owner)
    # A previous owner's copy of the workspace carries the same name (G44): scoped too.
    workspace = [dataset.id for dataset in readable if dataset.name == workspace_name]
    others = [dataset.id for dataset in readable if dataset.name != workspace_name]
    searches = []
    if workspace and visible:
        searches.append((workspace, [f"slack:channel:{scope}" for scope in visible]))
    if others and channel is None:
        searches.append((others, None))
    if not searches:
        # A channel answer reads only the workspace's channels; none of them is readable
        # here, so nothing was searched and "Nothing found" would be the wrong advice.
        await reply(notice(_nothing_to_read_message(credential)))
        return
    try:
        answers = await asyncio.gather(
            *(
                cognee_search(
                    query_text=question,
                    # A channel scope is a node set filter, and an entity is one node for
                    # every channel that names it: the hybrid entity lane would print a
                    # relation extracted from a channel outside the scope. Chunks belong to
                    # one channel each. Unscoped, hybrid's vector half still finds a chunk
                    # where graph traversal alone comes back empty on a thin graph.
                    query_type=SearchType.RAG_COMPLETION
                    if node_name
                    else SearchType.HYBRID_COMPLETION,
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
        # Every dataset answered that it does not know: told to the member alone.
        await reply(notice(NOTHING_FOUND_MESSAGE))
        return
    # The graph elements of the workspace search alone: feedback is applied to the
    # workspace dataset (``apply_answer_feedback``).
    used = _used_graph_elements(answers[0] if workspace and visible else [])
    try:
        # A direct message's question and answer stay out of the record (Goran,
        # 2026-11-12): it is kept under the workspace's owner, and the answer may quote the
        # member's own datasets. Its graph elements are all a rating needs.
        rating_ref = await record_answer(
            credential,
            member,
            question=question if channel else "",
            answer=fallback_text if channel else "",
            used_graph_element_ids=used,
        )
    except Exception:  # the answer is found: a cache that cannot keep it costs the rating
        logger.exception("Could not record a Slack answer for rating for team %s", team_id)
        rating_ref = None
    footer = _footer(channel, visible)
    try:
        await publish(
            {
                "text": fallback_text,
                "blocks": finish_answer(blocks, footer=footer, rating_ref=rating_ref),
            }
        )
    except Exception:  # an undelivered answer degrades to a chat message
        logger.exception("Slack did not take an @cognee answer for team %s", team_id)
        await reply(notice(ANSWER_NOT_POSTED_MESSAGE))


def _used_graph_elements(results: list[Any]) -> dict[str, list[str]] | None:
    """The node and edge ids the results of one search read, for feedback weights."""
    nodes: set[str] = set()
    edges: set[str] = set()
    for result in results:
        read = result.get("objects_result")
        # Hybrid returns its sections as a dict; a chunk search returns its scored chunks.
        if isinstance(read, dict):
            used = extract_context_object_ids(read) or {}
        else:
            used = extract_from_scored_results(read or []) or {}
        nodes |= set(used.get("node_ids", []))
        edges |= set(used.get("edge_ids", []))
    if not nodes and not edges:
        return None
    return {"node_ids": sorted(nodes), "edge_ids": sorted(edges)}


def _nothing_to_read_message(credential: IntegrationCredential) -> str:
    """Asked in a channel that is not synced, where no synced public channel is readable."""
    installer = (credential.provider_metadata or {}).get("installed_by_slack_user_id")
    who = f"<@{installer}>" if installer else "the person who connected Slack"
    return f"Nothing to read here\nThis channel is not synced. Ask {who} to add it."


def _footer(channel: str | None, visible: list[str]) -> str:
    """What the answer read, in one plain line. ``<#C…>`` renders as the channel's name."""
    if channel is None:
        return "From your channels and your memory"
    others = len([scope for scope in visible if scope != channel])
    counted = f"{others} public channel{'' if others == 1 else 's'}"
    if channel not in visible:
        return f"From {counted}"
    return f"From <#{channel}>" + (f" and {counted}" if others else "")


async def _direct_message_scope(
    reply: Reply, credential: IntegrationCredential, member: str
) -> list[str]:
    """The selected channels Slack says ``member`` is in; none when Slack cannot say."""
    team_id = credential.provider_account_id
    try:
        channels = await member_channels(credential, member)
    except SlackSourceError as error:
        logger.warning("Slack did not list a member's channels for team %s: %s", team_id, error)
        await reply(notice(MEMBERSHIP_UNKNOWN_MESSAGE))
        return []
    except Exception:
        # Anything else on the way to Slack (outbound HTTP switched off, a token that cannot
        # be read): the membership is just as unknown, so it fails closed the same way.
        logger.exception("Could not ask Slack for a member's channels for team %s", team_id)
        await reply(notice(MEMBERSHIP_UNKNOWN_MESSAGE))
        return []
    return [selected for selected in selected_channels(credential) if selected in channels]


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
    client = slack_source_module.build_slack_service(await access_token_for(credential))
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
    client = slack_source_module.build_slack_service(await access_token_for(credential))
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
    try:
        await _post(
            credential, "chat.postEphemeral", _message(channel, payload, thread_ts) | {"user": user}
        )
    except Exception:  # an undelivered reply must not crash the detached handler
        logger.exception(
            "Slack chat.postEphemeral failed for team %s", credential.provider_account_id
        )


async def post_message(
    credential: IntegrationCredential,
    channel: str,
    payload: dict[str, Any],
    thread_ts: str | None = None,
) -> None:
    """Post ``payload`` as a message in ``channel``, in the thread ``thread_ts`` when given,
    visible to everyone in it. Raises when Slack does not take it."""
    await _post(credential, "chat.postMessage", _message(channel, payload, thread_ts))


def _message(channel: str, payload: dict[str, Any], thread_ts: str | None) -> dict[str, Any]:
    # The sources are links; unfolded, each became a preview of a whole message under a
    # one-line answer (live, 2026-11-12).
    body = {
        "channel": channel,
        "text": payload["text"],
        "unfurl_links": False,
        "unfurl_media": False,
    }
    if payload.get("blocks"):
        body["blocks"] = payload["blocks"]
    if thread_ts:
        body["thread_ts"] = thread_ts
    return body


async def _post(credential: IntegrationCredential, method: str, body: dict[str, Any]) -> None:
    """Call the Slack Web API ``method`` with the bot token. Raises on any refusal."""
    token = await access_token_for(credential)
    async with (
        client_session(PROVIDER, timeout=_TIMEOUT) as session,
        session.post(
            _SLACK_API + method,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            # A Location header must not send the bot token elsewhere (SDK-832).
            allow_redirects=False,
        ) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"Slack {method} failed: HTTP {response.status}")
        answer = await response.json()
    if not answer.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {answer.get('error', 'unknown')}")
