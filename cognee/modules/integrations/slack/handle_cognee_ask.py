"""Handle ``/cognee-ask`` slash command — recall from knowledge graph.

Slack requires an ack within 3 seconds; ``HYBRID_COMPLETION`` search calls an
LLM and routinely takes longer than that. So the fast path here only
validates the workspace/query and acks immediately with a placeholder — the
actual search runs in a background task, and its answer is delivered
afterward via the command's ``response_url`` (see
:mod:`cognee.modules.integrations.slack.response_url`; ``replace_original:
true`` swaps the placeholder for the real answer instead of leaving both
messages visible).

The placeholder ack and every error path are ``ephemeral`` (visible only to
whoever ran the command) — a found answer is posted ``in_channel`` instead,
since asking in a shared channel or a DM with a colleague is usually so both
people can see the answer.
"""

import asyncio
import logging
from typing import Any, List, Optional, Tuple
from urllib.parse import parse_qs
from uuid import UUID

from cognee.api.v1.search.search import search as cognee_search
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.slack.persistence import get_by_team, is_active
from cognee.modules.integrations.slack.response_url import post_to_response_url
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_user

logger = logging.getLogger(__name__)

# asyncio.create_task() only holds a weak reference to its result — without
# keeping the task alive somewhere, it can be garbage-collected mid-flight
# the instant handle_cognee_ask returns. Each task removes itself on completion.
_pending_searches: set = set()

# Keep the placeholder ack readable even if someone pastes an essay.
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

    Everything past the workspace/query validation happens in the
    background — the real answer is not this function's return value, it
    arrives as a second message via ``response_url``.
    """
    form = parse_qs(raw_body.decode())
    team_id = (form.get("team_id") or [""])[0]
    text = (form.get("text") or [""])[0].strip()
    response_url = (form.get("response_url") or [""])[0]

    credential = await get_by_team(team_id) if team_id else None
    if not is_active(credential):
        return _ephemeral(
            "This Slack workspace is not connected to Cognee. "
            "Connect it from your Integrations settings."
        )

    if not text:
        return _ephemeral("Usage: `/cognee-ask <your question>`")

    task = asyncio.create_task(
        _search_and_respond(response_url, team_id, credential.user_id, text)
    )
    _pending_searches.add(task)
    task.add_done_callback(_pending_searches.discard)

    preview = text if len(text) <= _QUERY_PREVIEW_MAX_CHARS else text[:_QUERY_PREVIEW_MAX_CHARS] + "…"
    # This is the synchronous ack (within Slack's 3-second window), and Slack
    # only shows the "<@user> used /cognee-ask <query>" invocation line to
    # everyone in the channel/DM when *this* response is in_channel — an
    # ephemeral ack here hides the question from Slack's UI entirely, no
    # matter what visibility the later response_url follow-up uses.
    return _in_channel(f'Searching your memory for: "{preview}"…')


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
    await post_to_response_url(
        response_url, _in_channel(fallback_text, replace_original=True, blocks=blocks)
    )


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


def _format_answer(results: List[Any]) -> Tuple[str, List[dict[str, Any]]]:
    """Build a ``/cognee-ask`` answer as ``(fallback_text, blocks)``.

    ``fallback_text`` is a plain-text summary for notifications/screen
    readers; ``blocks`` (Block Kit) is what actually renders when the client
    supports it — a header, one section per fact with dividers between them,
    and a muted context line with the result count. No interactive elements
    yet (that needs a real ``/interactive`` handler, which is a separate
    piece of work); this is presentation only.
    """
    facts = [fact for fact in (_extract_fact(result) for result in results[:3]) if fact]
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
