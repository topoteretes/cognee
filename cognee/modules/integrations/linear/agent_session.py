"""Answer Linear agent sessions (@mentions / delegated issues) from cognee memory.

An agent session opens when a workspace member @mentions the agent or
delegates an issue to it (action ``created``) and continues when they reply
inside the session (action ``prompted``). Linear enforces a hard timing
contract: within **10 seconds** of a ``created`` event the agent must emit
an activity or the session is marked unresponsive. ``HYBRID_COMPLETION``
search calls an LLM and routinely takes longer than that, so the handler
posts an acknowledgement "thought" activity FIRST — before any search or
LLM work — and only then goes looking for an answer. (The generic events
route already acked the HTTP delivery; the 10 seconds are Linear's own
clock on the session, which only an activity satisfies.)

Every turn must also *end* in a terminal activity: a "response" completes
the turn, an "error" tells Linear (and the user) the agent gave up. Leaving
a session with only the thought would show it as working forever, so every
failure path here ends in a best-effort error activity instead of a raise —
this runs detached, there is no caller left to catch anything anyway.

The result-unwrapping and refusal-filtering helpers are deliberate local
copies of the Slack adapter's (:mod:`...slack.handle_cognee_ask`) —
provider-local by convention, so one integration's formatting tweaks never
ripple into another's.
"""

import logging
from typing import Any, Optional

from cognee.api.v1.search.search import search as cognee_search
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.integrations.linear.client import create_agent_activity
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_user

logger = logging.getLogger(__name__)

_ACK_THOUGHT = "Searching cognee memory…"
_NO_ANSWER = "No relevant information found in cognee memory."
_ERROR_BODY = "Something went wrong while searching cognee memory. Please try again."


async def handle_agent_session(credential: IntegrationCredential, payload: dict[str, Any]) -> None:
    """Answer one agent session event. Never raises — see module docstring."""
    agent_session_id = (payload.get("agentSession") or {}).get("id")
    if not agent_session_id:
        logger.warning("Linear agent session event without a session id; ignoring")
        return

    # Imported here, not at module top: the adapter imports this module to
    # wire handle_webhook, so a top-level import back into it would be
    # circular.
    from cognee.modules.integrations.linear.adapter import access_token_for

    try:
        access_token = access_token_for(credential)
    except Exception:  # noqa: BLE001 - a bad stored payload must not crash the detached handler
        logger.exception(
            "Linear agent session %s: no usable token for organization %s",
            agent_session_id,
            credential.provider_account_id,
        )
        return

    # The 10-second rule: acknowledge before any search/LLM work, or Linear
    # marks the session unresponsive.
    try:
        await create_agent_activity(
            access_token, agent_session_id, {"type": "thought", "body": _ACK_THOUGHT}
        )
    except Exception:  # noqa: BLE001 - a failed ack degrades the display; a missing response would kill the turn
        logger.exception("Linear agent session %s: acknowledgement failed", agent_session_id)

    try:
        answer = await _answer(credential, payload)
        await create_agent_activity(
            access_token, agent_session_id, {"type": "response", "body": answer}
        )
    except Exception:  # noqa: BLE001 - every failure must end the turn in an error activity, not a raise
        logger.exception("Linear agent session %s: answering failed", agent_session_id)
        try:
            await create_agent_activity(
                access_token, agent_session_id, {"type": "error", "body": _ERROR_BODY}
            )
        except Exception:  # noqa: BLE001 - best effort; nothing left to do but log
            logger.exception(
                "Linear agent session %s: error activity delivery failed", agent_session_id
            )


def _resolve_question(payload: dict[str, Any]) -> str:
    """The user's actual ask, per event shape.

    ``prompted`` carries the new user message at the payload's top level in
    ``agentActivity.body``. ``created`` carries the triggering comment inside
    ``agentSession`` — but a delegation (issue assigned to the agent) has no
    comment, so the issue title + description stand in for the question.
    """
    if payload.get("action") == "prompted":
        return ((payload.get("agentActivity") or {}).get("body") or "").strip()

    session = payload.get("agentSession") or {}
    comment_body = ((session.get("comment") or {}).get("body") or "").strip()
    if comment_body:
        return comment_body

    issue = session.get("issue") or {}
    title = (issue.get("title") or "").strip()
    description = (issue.get("description") or "").strip()
    return f"{title}\n{description}".strip()


def _build_query(payload: dict[str, Any], question: str) -> str:
    """Prepend Linear's own context to the search query.

    ``promptContext`` is a formatted digest Linear assembles (issue details,
    prior comments) and ``guidance`` is workspace-level steering for agents —
    both ground retrieval in what the session is about, so they ride along
    ahead of the question instead of being discarded.
    """
    parts = [part for part in (payload.get("promptContext"), payload.get("guidance")) if part]
    parts.append(question)
    return "\n\n".join(parts)


async def _answer(credential: IntegrationCredential, payload: dict[str, Any]) -> str:
    """Search cognee memory and return the response body for this turn.

    "Nothing found" and "no question asked" are friendly responses, not
    errors — only genuine failures (which raise out of here) become an
    error activity.
    """
    question = _resolve_question(payload)
    if not question:
        return (
            "I couldn't find a question in this session — mention me with one "
            "and I'll search cognee memory."
        )

    try:
        owner = await get_user(credential.user_id)
    except EntityNotFoundError:
        logger.error(
            "Linear credential for organization %s points at a deleted user",
            credential.provider_account_id,
        )
        return (
            "This workspace's cognee connection is not fully configured. "
            "Please disconnect and reconnect the integration."
        )

    results = await cognee_search(
        query_text=_build_query(payload, question),
        # Matches the frontend recall default (and the Slack adapter): graph
        # traversal alone can come back empty on a thin/loosely connected
        # graph, where the vector-search fallback still finds a relevant
        # chunk.
        query_type=SearchType.HYBRID_COMPLETION,
        user=owner,
        datasets=None,  # search across every dataset the owner can read
    )

    facts = [
        fact
        for fact in (_extract_fact(result) for result in results)
        if fact and not _is_refusal(fact)
    ][:3]
    if not facts:
        return _NO_ANSWER
    return "\n\n".join(facts)


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
