"""Faithful preview of a completion's input, for ``only_context`` searches.

``only_context=True`` returns the retrieval context and nothing else — no session
guidance, no conversation history, no rendered prompt — while a real completion sends
all of it. A caller that pipes that context straight into its own LLM therefore works
from strictly less than cognee itself would.

This module rebuilds the missing layers and renders them through
``build_completion_prompts``, the same assembly ``generate_completion`` uses, so the
preview cannot drift from the real call as templates change.

Two invariants hold, because ``only_context`` callers depend on them:

* **No LLM call.** The pre-retrieval turn analysis is deliberately not run; session
  state is read as it stands.
* **No session write.** The guidance block is built with ``stamp_served=False``, and
  nothing here records a QA turn.

One honest gap remains: a real turn retrieves with a rewritten query (the analysis's
``effective_query``, or the merged raw/conversational lanes in concurrent mode), while
``only_context`` retrieves on the raw query. The preview reports the prompt for the
context that was actually retrieved rather than pretending otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from cognee.context_global_variables import session_user
from cognee.modules.retrieval.utils.completion import build_completion_prompts
from cognee.shared.logging_utils import get_logger

logger = get_logger("ContextPreview")

# ``context_format`` values. "context" is the historical shape: the bare context string.
# "prompt" returns the full envelope a completion would have received.
CONTEXT_FORMAT_CONTEXT = "context"
CONTEXT_FORMAT_PROMPT = "prompt"
CONTEXT_FORMATS = frozenset({CONTEXT_FORMAT_CONTEXT, CONTEXT_FORMAT_PROMPT})

# The separator the graph prompt template documents for stacked context entries.
CONTEXT_LIST_SEPARATOR = "\n---\n"


@dataclass(frozen=True)
class ContextPreview:
    """What a completion would have been given, minus the completion itself.

    ``user_prompt``/``system_prompt`` stay ``None`` for retrievers that never build a
    prompt (CHUNKS, SUMMARIES, CODE, ...) — inventing one would misrepresent a search
    type whose contract is explicitly non-generative.
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


async def load_read_only_session_prompt(
    retriever, raw_query: str, *, session_id: Optional[str] = None
) -> str:
    """Rebuild the session layer of the prompt without writing or calling an LLM.

    Mirrors ``generate_session_answer``: the guidance block leads, the conversation
    history follows, and durable preferences render through the same owner whether or
    not automatic feedback is on. Fails open to ``""`` — a missing session layer must
    never take a retrieval-only call down.

    ``session_id`` is the one the caller passed to ``search()``. It takes precedence
    over the retriever's own attribute because not every retriever keeps one:
    ``ChunksRetriever`` and the other non-generative types drop it, and falling back to
    the default session there would report the wrong conversation.
    """
    try:
        from cognee.infrastructure.session.get_session_manager import get_session_manager
        from cognee.infrastructure.session.session_context_builder import (
            build_active_context_block,
            render_preference_block,
        )
        from cognee.infrastructure.session.session_turn import (
            compose_session_prompt,
            load_preference_lines_safe,
            select_session_history,
        )

        user_uuid = getattr(session_user.get(), "id", None)
        if not user_uuid:
            return ""

        session_manager = get_session_manager()
        if not session_manager.is_session_available_for_completion(user_uuid):
            return ""

        user_id = str(user_uuid)
        requested_session_id = (
            session_id if session_id is not None else getattr(retriever, "session_id", None)
        )
        session_id = session_manager.resolve_session_id(requested_session_id)

        history = await select_session_history(
            session_manager,
            user_id,
            session_id,
            query_text=raw_query,
        )
        preference_lines = await load_preference_lines_safe()

        active_context_block = ""
        if session_manager.is_auto_feedback_enabled():
            # stamp_served=False is what keeps this read-only: the real answer path
            # stamps last_served_at on every rendered entry, a preview must not.
            active_context_block, _served_ids = await build_active_context_block(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                query=raw_query,
                preference_lines=preference_lines,
                stamp_served=False,
            )
        elif preference_lines:
            active_context_block = render_preference_block(preference_lines)

        return compose_session_prompt(active_context_block, history if history else "")
    except Exception as error:
        logger.warning("Only-context session prompt failed open: %s", error)
        return ""


async def build_context_preview(
    retriever, *, query: str, context: Any, session_id: Optional[str] = None
) -> ContextPreview:
    """Assemble the session layer and the rendered prompts for one ``only_context`` call."""
    session_context = await load_read_only_session_prompt(retriever, query, session_id=session_id)

    user_prompt_path = getattr(retriever, "user_prompt_path", None)
    system_prompt_path = getattr(retriever, "system_prompt_path", None)
    if not user_prompt_path or not system_prompt_path:
        return ContextPreview(session_context=session_context)

    try:
        user_prompt, system_prompt = build_completion_prompts(
            query=query,
            context=render_context_for_prompt(context),
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=getattr(retriever, "system_prompt", None),
            conversation_history=session_context or None,
        )
    except Exception as error:
        logger.warning("Only-context prompt rendering failed open: %s", error)
        return ContextPreview(session_context=session_context)

    return ContextPreview(
        session_context=session_context,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
