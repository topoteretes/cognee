"""Faithful preview of a completion's input, for ``only_context`` searches.

``only_context=True`` returns the retrieval context and nothing else — no session
guidance, no conversation history, no rendered prompt — while a real completion sends
all of it. A caller that pipes that context straight into its own LLM therefore works
from strictly less than cognee itself would.

This module rebuilds the missing layers through the *same* code the real completion
uses: ``build_session_prompt`` (``session_turn.py``) for the session layer, in its
read-only mode, and ``build_completion_prompts`` (``utils/completion.py``) for the
prompt pair. Neither layer is re-implemented here, so the preview cannot drift from the
real call as either evolves.

What the preview guarantees, because ``only_context`` callers depend on it:

* **No LLM completion and no turn analysis.** Session state is read as it stands.
* **No session write.** The guidance block is built with ``stamp_served=False``, and no
  QA turn is recorded.

What it costs: the session layer's conversation-history recall embeds the query for a
vector lookup — one embedding call, made once per search and shared across the dataset
fan-out (``SharedSessionHistory``), never once per dataset. It is not free; it is the
only billed step, and it is the same step a real turn pays.

Where it is knowingly unfaithful: a real sequential turn rewrites the question first
(``turn_preparation.effective_query``), and that rewrite fills the ``{{ question }}``
slot, drives history selection, and ranks the guidance block; concurrent mode also
merges a second retrieval lane. Producing that rewrite is an LLM call, which this path
must not make, so the preview uses the raw query for all of them. It reports the prompt
for the context actually retrieved, not a replay of a full turn.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

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

logger = get_logger("ContextPreview")

# The separator the graph prompt template documents for stacked context entries.
CONTEXT_LIST_SEPARATOR = "\n---\n"


@dataclass(frozen=True)
class ContextPreview:
    """What a completion would have been given, minus the completion itself.

    ``user_prompt``/``system_prompt`` stay ``None`` for retrievers that never send a
    single prompt built from their template pair — the non-generative types (CHUNKS,
    SUMMARIES, CODE, ...) and those that opt out via ``supports_prompt_preview`` —
    because inventing one would misrepresent what cognee sends.
    """

    session_context: str = ""
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None


def render_context_for_prompt(context: Any) -> Any:
    """Flatten a list-valued context the way a prompt template would want it.

    Batch retrievals return one context per query; joining keeps the rendered prompt
    readable instead of interpolating a Python repr. Non-list contexts pass through.
    """
    if isinstance(context, (list, tuple)):
        return CONTEXT_LIST_SEPARATOR.join(str(entry) for entry in context)
    return context


class SharedSessionHistory:
    """One conversation-history read, shared by every dataset in a search fan-out.

    ``select_session_history`` embeds the query for vector recall — the session layer's
    only billed step. A multi-dataset search runs one preview per dataset, and the
    history is the same for all of them, so the first caller loads it and the rest
    await the same result. The guidance block is deliberately *not* shared: preferences
    are dataset-scoped, so it renders per dataset.
    """

    def __init__(self, *, query: str, session_id: Optional[str]):
        self.query = query
        self.session_id = session_id
        self._lock = asyncio.Lock()
        self._history: Optional[str] = None

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
    session_id: Optional[str] = None,
    shared_history: Optional[SharedSessionHistory] = None,
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
        logger.warning("Only-context session prompt failed open: %s", error)
        return ""


async def build_context_preview(
    retriever,
    *,
    query: str,
    context: Any,
    session_id: Optional[str] = None,
    shared_history: Optional[SharedSessionHistory] = None,
) -> ContextPreview:
    """Assemble the session layer and the rendered prompts for one ``only_context`` call.

    ``session_id`` is the one the caller passed to ``search()``; it wins over the
    retriever's own attribute because the non-generative retrievers do not keep one.

    A missing or unreadable template is not swallowed: the real completion would fail on
    it too, and reporting "no prompt" instead would hide a misconfigured path.
    """
    requested_session_id = (
        session_id if session_id is not None else getattr(retriever, "session_id", None)
    )
    session_context = await load_read_only_session_prompt(
        query, session_id=requested_session_id, shared_history=shared_history
    )

    user_prompt_path = getattr(retriever, "user_prompt_path", None)
    system_prompt_path = getattr(retriever, "system_prompt_path", None)
    if (
        not getattr(retriever, "supports_prompt_preview", True)
        or not user_prompt_path
        or not system_prompt_path
    ):
        return ContextPreview(session_context=session_context)

    user_prompt, system_prompt = build_completion_prompts(
        query=query,
        context=render_context_for_prompt(context),
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        system_prompt=getattr(retriever, "system_prompt", None),
        conversation_history=session_context or None,
    )
    return ContextPreview(
        session_context=session_context,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
