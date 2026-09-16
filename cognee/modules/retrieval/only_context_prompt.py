"""The full LLM input for ``only_context`` searches.

``only_context=True`` promises the caller what cognee's completion would have worked
from, so it can hand that to its own LLM instead. A real completion sends more than the
retrieval context: the session guidance block and conversation history ride on the
system prompt, and the question and context are rendered through the retriever's user
template. Returning the bare context handed the caller strictly less than cognee itself
uses.

This module assembles the missing layers through the *same* code the real completion
uses — ``build_session_prompt`` (``session_turn.py``) for the session layer, in its
read-only mode, and ``build_completion_prompts`` (``utils/completion.py``) for the
prompt pair — and renders them as one string. Neither layer is re-implemented here, so
the string cannot drift from the real call as either evolves.

What ``only_context`` callers can rely on:

* **No LLM completion and no turn analysis.** Session state is read as it stands.
* **No session write.** The guidance block is built with ``stamp_served=False``, and no
  QA turn is recorded.
* **Bare context where no prompt exists.** Non-generative retrievers (CHUNKS, SUMMARIES,
  CODE, ...) and retrievers that opt out via ``supports_prompt_preview`` never send one
  templated prompt, so nothing is built for them and the payload falls back to the
  retrieval context. The same holds for an empty retrieval: a prompt wrapped around
  nothing would read as a hit, and "nothing found" must stay detectable.

Cost: the session layer's conversation-history recall embeds the query for a vector
lookup — one embedding call, made once per search and shared across the dataset fan-out
(``SharedSessionHistory``), never once per dataset. It is the only billed step, it is
the same step a real turn pays, and it is paid only when a prompt is actually built.

Retrievers whose *retrieval* stage itself calls an LLM — chain-of-thought validation
and follow-ups, decomposition sub-answers, context-extension rounds, the temporal
retriever's time extraction, graph-summary's summaries — still make those calls under
``only_context``, exactly as they always have; nothing here adds to them. For those types
the string is the final prompt pair over the final context.

Knowingly unfaithful in one place: a real sequential turn rewrites the question first
(``turn_preparation.effective_query``), and that rewrite fills the ``{{ question }}``
slot, drives history selection, and ranks the guidance block; concurrent mode also
merges a second retrieval lane. Producing that rewrite is an LLM call, which this path
must not make, so the raw query is used for all of them. The string is the prompt for
the context actually retrieved, not a replay of a full turn.
"""

from __future__ import annotations

import asyncio
from typing import Any

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_turn import (
    build_session_prompt,
    select_session_history,
)
from cognee.modules.retrieval.utils.completion import build_completion_prompts
from cognee.modules.user_preferences import load_preference_text
from cognee.shared.logging_utils import get_logger

logger = get_logger("OnlyContextPrompt")

# The separator the graph prompt template documents for stacked context entries.
CONTEXT_LIST_SEPARATOR = "\n---\n"

# Role markers for the rendered string. The LLM receives the two prompts as separate
# messages; one string needs a visible seam so a caller can still split them. Fenced
# lines rather than bare "SYSTEM:"/"USER:" so ordinary prose in retrieved text or a
# stored answer is unlikely to forge a boundary. Content is passed through verbatim, so
# the string is a rendering, not a parseable protocol: a marker line inside the content
# is not escaped.
SYSTEM_PROMPT_HEADER = "=== SYSTEM PROMPT ==="
USER_PROMPT_HEADER = "=== USER PROMPT ==="


def render_context_for_prompt(context: Any) -> Any:
    """Flatten a list-valued context the way a prompt template would want it.

    Batch retrievals return one context per query; joining keeps the rendered prompt
    readable instead of interpolating a Python repr. Non-list contexts pass through.
    """
    if isinstance(context, (list, tuple)):
        return CONTEXT_LIST_SEPARATOR.join(str(entry) for entry in context)
    return context


def has_context(context: Any) -> bool:
    """Whether a retriever's context carries anything an LLM could work from.

    Retrievers report a miss as ``None``, ``""`` or ``[]`` (or a list of those); all of
    them mean "nothing found" and none of them deserve a prompt wrapped around them.
    """
    if context is None:
        return False
    if isinstance(context, str):
        return bool(context.strip())
    if isinstance(context, (list, tuple)):
        return any(has_context(entry) for entry in context)
    return True


def render_llm_input(system_prompt: str, user_prompt: str) -> str:
    """Render the ``(system_prompt, user_prompt)`` pair a completion sends as one string."""
    return f"{SYSTEM_PROMPT_HEADER}\n{system_prompt}\n\n{USER_PROMPT_HEADER}\n{user_prompt}"


class SharedSessionHistory:
    """One conversation-history read, shared by every dataset in a search fan-out.

    ``select_session_history`` embeds the query for vector recall — the session layer's
    only billed step. A multi-dataset search builds one prompt per dataset, and the
    history is the same for all of them, so the first caller loads it and the rest
    await the same result. The guidance block is deliberately *not* shared: preferences
    are dataset-scoped, so it renders per dataset. Nothing is read until ``get`` is
    first called, so creating one for a search that ends up building no prompt is free.
    """

    def __init__(self, *, query: str, session_id: str | None):
        self.query = query
        self.session_id = session_id
        self._lock = asyncio.Lock()
        self._history: str | None = None

    async def get(self, session_manager, *, user_id: str, resolved_session_id: str) -> str:
        async with self._lock:
            if self._history is None:
                self._history = await select_session_history(
                    session_manager,
                    user_id,
                    resolved_session_id,
                    query_text=self.query,
                )
            return self._history


async def load_read_only_session_prompt(
    raw_query: str,
    *,
    session_id: str | None = None,
    shared_history: SharedSessionHistory | None = None,
) -> str:
    """The session layer a completion would carry, read without writing or calling an LLM.

    Mirrors the retriever's branch point exactly:

    * caching off, or no user: the real call takes the sessionless path, whose only
      session-layer content is the durable preference block — return that;
    * caching on but the cache backend unavailable: the real call sends a bare prompt;
    * otherwise: ``build_session_prompt`` in read-only mode.

    Fails open to ``""`` — a missing session layer must never take a retrieval-only call
    down.
    """
    try:
        user_uuid = getattr(session_user.get(), "id", None)
        if not (user_uuid and CacheConfig().caching):
            return await load_preference_text()

        session_manager = get_session_manager()
        if not session_manager.is_session_available_for_completion(user_uuid):
            return ""

        user_id = str(user_uuid)
        resolved_session_id = session_manager.resolve_session_id(session_id)
        history = None
        if shared_history is not None:
            history = await shared_history.get(
                session_manager, user_id=user_id, resolved_session_id=resolved_session_id
            )

        prompt, _served_ids = await build_session_prompt(
            session_manager,
            user_id=user_id,
            session_id=resolved_session_id,
            query=raw_query,
            history=history,
            stamp_served=False,
        )
        return prompt
    except Exception as error:
        logger.warning("Only-context session prompt failed open: %s", error, exc_info=True)
        return ""


def retriever_sends_one_prompt(retriever) -> bool:
    """Whether ``get_completion_from_context`` sends exactly one templated prompt pair.

    False for the non-generative retrievers (no template attributes at all) and for the
    ones that opt out via ``supports_prompt_preview`` because they answer through other
    templates or never prompt an LLM.
    """
    return bool(
        getattr(retriever, "supports_prompt_preview", True)
        and getattr(retriever, "user_prompt_path", None)
        and getattr(retriever, "system_prompt_path", None)
    )


async def build_only_context_prompt(
    retriever,
    *,
    query: str,
    context: Any,
    session_id: str | None = None,
    shared_history: SharedSessionHistory | None = None,
) -> str | None:
    """The full LLM input for one ``only_context`` call, or ``None`` when there is none.

    ``None`` means "return the bare context instead": the retriever never sends a single
    templated prompt, or retrieval found nothing. Only when a prompt will be built is
    the session layer read, so those calls pay no embedding call.

    ``session_id`` is the one the caller passed to ``search()``; it wins over the
    retriever's own attribute because the non-generative retrievers do not keep one.

    A missing or unreadable template is not swallowed: the real completion would fail on
    it too, and returning the bare context instead would hide a misconfigured path.
    """
    if not retriever_sends_one_prompt(retriever) or not has_context(context):
        return None

    requested_session_id = (
        session_id if session_id is not None else getattr(retriever, "session_id", None)
    )
    session_context = await load_read_only_session_prompt(
        query, session_id=requested_session_id, shared_history=shared_history
    )

    user_prompt, system_prompt = build_completion_prompts(
        query=query,
        context=render_context_for_prompt(context),
        user_prompt_path=retriever.user_prompt_path,
        system_prompt_path=retriever.system_prompt_path,
        system_prompt=getattr(retriever, "system_prompt", None),
        conversation_history=session_context or None,
    )
    return render_llm_input(system_prompt, user_prompt)
