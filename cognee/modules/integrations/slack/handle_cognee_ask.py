"""Handle ``/cognee-ask`` slash command — recall from knowledge graph.

Nothing about a ``/cognee-ask`` becomes visible to the channel until the
asker has seen the answer and explicitly chosen to share it — neither the
question nor the answer are known to be channel-appropriate before that.
So every step here, including the initial "searching..." ack, is
``ephemeral`` (visible only to whoever ran the command). Once the search
finishes, the answer is shown back privately with Share/Discard buttons:
Share is the *only* thing that ever posts ``in_channel`` (question and
answer together, since nothing was public before it); Discard drops the
answer entirely, visible to no one, not even the asker.

Slack requires an ack within 3 seconds; ``HYBRID_COMPLETION`` search calls an
LLM and routinely takes longer than that. So the fast path here only
validates and acks immediately with a placeholder — the actual search runs
in a background task, and its answer is delivered afterward via the
command's ``response_url`` (see
:mod:`cognee.modules.integrations.slack.response_url`; ``replace_original:
true`` swaps the placeholder for the next prompt instead of leaving both
messages visible).
"""

import asyncio
import logging
from typing import Any, List, Optional, Tuple
from urllib.parse import parse_qs
from uuid import UUID, uuid4

from cognee.api.v1.search.search import search as cognee_search
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.slack.persistence import (
    get_by_team,
    is_active,
    resolve_owner_user_id,
)
from cognee.modules.integrations.slack.response_url import post_to_response_url
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_user

logger = logging.getLogger(__name__)

# Referenced by handle_slack_interactive.py to dispatch the answer review
# prompt's button clicks back here.
ASK_SHARE_ACTION_ID = "cognee_ask_share"
ASK_DISCARD_ACTION_ID = "cognee_ask_discard"

# asyncio.create_task() only holds a weak reference to its result — without
# keeping the task alive somewhere, it can be garbage-collected mid-flight
# the instant handle_cognee_ask returns. Each task removes itself on completion.
_pending_searches: set = set()

# A ready-to-share (question, answer) pair, keyed by a short id (never the
# text itself — Slack button values cap at 2000 chars, easily blown by a
# real question+answer). Popped on Share or Discard; an id missing here
# (server restart, double-click) means "no longer available", not a crash.
_pending_answers: dict[str, Tuple[str, str, List[dict[str, Any]]]] = {}

# Keep the placeholder/confirm text readable even if someone pastes an essay.
_QUERY_PREVIEW_MAX_CHARS = 200


def _ephemeral(
    text: str,
    *,
    replace_original: bool = False,
    blocks: Optional[List[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a reply visible only to the person who ran the command.

    Used for the placeholder ack and every error path — none of those are
    useful to anyone else in the channel/DM.
    """
    return _reply("ephemeral", text, replace_original=replace_original, blocks=blocks)


def _in_channel(
    text: str,
    *,
    replace_original: bool = False,
    blocks: Optional[List[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a reply visible to everyone in the channel/DM the command ran in.

    Used only for a successful answer — the point of asking in a shared
    channel or a DM with a colleague is usually for both people to see it.
    """
    return _reply("in_channel", text, replace_original=replace_original, blocks=blocks)


def _reply(
    response_type: str,
    text: str,
    *,
    replace_original: bool,
    blocks: Optional[List[dict[str, Any]]],
) -> dict[str, Any]:
    # `text` is always set, even when `blocks` is present — Slack renders it
    # in notifications, screen readers, and any client that doesn't support
    # Block Kit, so it must never be an empty fallback.
    payload: dict[str, Any] = {"response_type": response_type, "text": text}
    if blocks:
        payload["blocks"] = blocks
    if replace_original:
        payload["replace_original"] = True
    return payload


async def handle_cognee_ask(raw_body: bytes) -> dict[str, Any]:
    """Ack ``/cognee-ask {query}`` within Slack's 3-second window.

    Everything past the workspace/query/authorization validation happens in
    the background — the real answer is not this function's return value,
    it arrives as a second message via ``response_url`` (first the answer
    review prompt, then whatever Share/Discard resolves to).
    """
    form = parse_qs(raw_body.decode())
    team_id = (form.get("team_id") or [""])[0]
    invoking_slack_user_id = (form.get("user_id") or [""])[0]
    text = (form.get("text") or [""])[0].strip()
    response_url = (form.get("response_url") or [""])[0]

    credential = await get_by_team(team_id) if team_id else None
    if not is_active(credential):
        return _ephemeral(
            "This Slack workspace is not connected to Cognee. "
            "Connect it from your Integrations settings."
        )

    owner_user_id = await resolve_owner_user_id(credential, team_id, invoking_slack_user_id)
    if owner_user_id is None:
        return _ephemeral(
            "Link your own Cognee account first: `/cognee-link <api_key>` "
            "(create a key from your Cognee account's API Keys settings)."
        )

    if not text:
        return _ephemeral("Usage: `/cognee-ask <your question>`")

    task = asyncio.create_task(_search_and_respond(response_url, team_id, owner_user_id, text))
    _pending_searches.add(task)
    task.add_done_callback(_pending_searches.discard)

    # Ephemeral — nothing is visible to the channel until Share is clicked
    # on the answer that follows, so this ack (and the "<@user> used
    # /cognee-ask ..." invocation line an in_channel response would trigger)
    # must not leak the question either.
    return _ephemeral(f'Searching your memory for: "{_preview(text)}"…')


def _preview(text: str) -> str:
    return text if len(text) <= _QUERY_PREVIEW_MAX_CHARS else text[:_QUERY_PREVIEW_MAX_CHARS] + "…"


async def _search_and_respond(response_url: str, team_id: str, user_id: UUID, text: str) -> None:
    """Run the actual search and deliver its answer via ``response_url``.

    Never raises — this runs detached from the request/response cycle, so
    there is no caller left to catch anything; every failure path here ends
    in a best-effort message delivery instead.
    """
    try:
        owner = await get_user(user_id)
    except EntityNotFoundError:
        logger.error("Slack credential for team %s points at a deleted user", team_id)
        await post_to_response_url(
            response_url,
            _ephemeral(
                "Slack integration is not fully configured. Please disconnect and reconnect.",
                replace_original=True,
            ),
        )
        return

    try:
        results = await cognee_search(
            query_text=text,
            # Matches the frontend recall page's default (recallKnowledge.ts):
            # graph traversal alone can come back empty on a thin/loosely
            # connected graph, where the vector-search fallback still finds
            # a relevant chunk.
            query_type=SearchType.HYBRID_COMPLETION,
            user=owner,
            datasets=None,  # search across every dataset the owner can read
        )
    except Exception:  # noqa: BLE001 - any search failure must degrade to a chat message, not a crash
        logger.exception("Search failed for Slack team %s", team_id)
        await post_to_response_url(
            response_url, _ephemeral("Search failed. Please try again.", replace_original=True)
        )
        return

    fallback_text, blocks = _format_answer(results)
    answer_id = uuid4().hex
    _pending_answers[answer_id] = (text, fallback_text, blocks)

    await post_to_response_url(
        response_url,
        _ephemeral(
            f"{fallback_text}\n\nShare this in the channel?",
            replace_original=True,
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
                            "value": answer_id,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Discard"},
                            "action_id": ASK_DISCARD_ACTION_ID,
                            "value": answer_id,
                        },
                    ],
                },
            ],
        ),
    )


async def handle_cognee_ask_share(response_url: str, answer_id: str) -> dict[str, Any]:
    """Handle the answer prompt's Share click — posts the question and its
    answer in_channel together, replacing the ephemeral review prompt. This
    is the first and only point either one becomes visible to anyone else.
    """
    pending = _pending_answers.pop(answer_id, None)
    if pending is None:
        return _ephemeral("This answer is no longer available.", replace_original=True)

    question, fallback_text, blocks = pending
    question_block = {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f'*Asked:* "{_preview(question)}"'},
    }
    return _in_channel(
        f'Asked: "{_preview(question)}"\n\n{fallback_text}',
        replace_original=True,
        blocks=[question_block, {"type": "divider"}, *blocks],
    )


async def handle_cognee_ask_discard(answer_id: str) -> dict[str, Any]:
    """Handle the answer prompt's Discard click — drops the answer for good."""
    _pending_answers.pop(answer_id, None)
    return _ephemeral("Discarded.", replace_original=True)


def _extract_fact(result: Any) -> Optional[str]:
    """Pull the answer text out of one ``cognee.search()`` result.

    Despite the ``List[SearchResult]`` type hint, the public ``search()``
    never returns raw ``SearchResult`` objects — it always runs them through
    ``_backwards_compatible_search_results`` first, which yields a dict with
    a ``"search_result"`` key when backend access control is enabled, or the
    bare completion payload otherwise. Either way that payload can itself be
    a single string or a list of strings (e.g. ``HYBRID_COMPLETION`` returns
    ``["<answer>"]``) — the first non-empty string is used. Anything else is
    skipped rather than raising, since a shape this code doesn't recognize
    shouldn't crash the whole answer.
    """
    value = result.get("search_result") if isinstance(result, dict) else result
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item), None)
    return value if isinstance(value, str) and value else None


# Substrings that mark a fact as "I can't answer this from the given
# context" rather than an actual answer. HYBRID_COMPLETION can return a
# separate per-chunk completion for each matched chunk group, and a chunk
# irrelevant to the question still produces its own such completion —
# _extract_fact has no way to tell that apart from a real answer (it's just
# as valid a non-empty string), so it's filtered out here instead.
_REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i'm unable to",
    "can't answer",
    "cannot answer",
    "no information about",
    "no relevant information",
    "doesn't contain",
    "does not contain",
    "contains no information",
)


def _is_refusal(fact: str) -> bool:
    lowered = fact.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _format_answer(results: List[Any]) -> Tuple[str, List[dict[str, Any]]]:
    """Build a ``/cognee-ask`` answer as ``(fallback_text, blocks)``.

    ``fallback_text`` is a plain-text summary for notifications/screen
    readers; ``blocks`` (Block Kit) is what actually renders when the client
    supports it — a header, one section per fact with dividers between them,
    and a muted context line with the result count. No interactive elements
    yet (that needs a real ``/interactive`` handler, which is a separate
    piece of work); this is presentation only.
    """
    facts = [
        fact
        for fact in (_extract_fact(result) for result in results)
        if fact and not _is_refusal(fact)
    ][:3]
    if not facts:
        return "No relevant information found.", []

    blocks: List[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Answer", "emoji": True}},
    ]
    for fact in facts:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": fact}})
        blocks.append({"type": "divider"})
    blocks.pop()  # no trailing divider after the last fact

    result_word = "result" if len(facts) == 1 else "results"
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Searched across your Cognee memory · {len(facts)} {result_word}",
                }
            ],
        }
    )

    fallback_text = "*Answer:*\n" + "\n\n".join(f"*{i}.* {fact}" for i, fact in enumerate(facts, 1))
    return fallback_text, blocks
