"""Session turn analysis from user messages via LLM."""

from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.shared.logging_utils import get_logger

logger = get_logger("feedback_detection")


def _render_served_context(served_context) -> str:
    """Render served session-context entries as an ``id: content`` block.

    Accepts a pre-rendered string (returned as-is) or a list of dicts/objects exposing
    ``id``/``entry_id`` and ``content`` fields. Never raises: malformed items are skipped.
    """
    if served_context is None:
        return ""
    if isinstance(served_context, str):
        return served_context.strip()

    lines = []
    try:
        for item in served_context:
            if isinstance(item, dict):
                entry_id = item.get("id") or item.get("entry_id")
                content = item.get("content")
            else:
                entry_id = getattr(item, "id", None) or getattr(item, "entry_id", None)
                content = getattr(item, "content", None)
            if entry_id is None or content is None:
                continue
            lines.append(f"{str(entry_id).strip()}: {str(content).strip()}")
    except Exception:
        return ""
    return "\n".join(lines)


def is_auto_feedback_enabled() -> bool:
    """True when session caching and automatic turn-feedback analysis are both on."""
    cache_config = get_cache_config()
    return bool(cache_config.caching and cache_config.auto_feedback)


def _append_optional_section(text_input: str, title: str, content: str | None) -> str:
    if not content or not str(content).strip():
        return text_input
    return text_input + f"\n\n{title}:\n" + str(content).strip()


async def analyze_turn_for_session_context(
    user_message: str,
    *,
    previous_question: str | None = None,
    previous_answer: str | None = None,
    served_context: list | str | None = None,
) -> SessionTurnAnalysis:
    """
    Analyze a user message for answer routing and session-context updates.

    When ``served_context`` is provided (a pre-rendered string or a list of session-context
    entries served to the previous answer), it is appended to the LLM input so the single
    turn-analysis call can additionally produce ``served_context_ratings`` and
    ``candidate_context_updates``. This adds no extra LLM call.

    Returns a SessionTurnAnalysis. On LLM failure or timeout, returns an empty analysis so
    the main completion flow is never blocked.
    """
    if not (user_message and str(user_message).strip()):
        return SessionTurnAnalysis()

    try:
        system_prompt = read_query_prompt("feedback_detection_system.txt")
        if not system_prompt:
            logger.warning("Feedback detection: system prompt not found, skipping")
            return SessionTurnAnalysis()

        # The 1-5 previous-answer-rating question is part of automatic feedback
        # analysis: it is asked whenever AUTO_FEEDBACK is on, so the implicit
        # rating is produced and stored even when no consumer is switched on
        # yet. Personalization is one consumer of it, not its gate.
        if is_auto_feedback_enabled():
            rating_section = read_query_prompt("feedback_detection_rating_section.txt")
            if rating_section:
                system_prompt = system_prompt + "\n\n" + rating_section

        text_input = "CURRENT USER MESSAGE:\n" + user_message.strip()
        text_input = _append_optional_section(
            text_input,
            "PREVIOUS QUESTION",
            previous_question,
        )
        text_input = _append_optional_section(
            text_input,
            "PREVIOUS ANSWER",
            previous_answer,
        )
        rendered_context = _render_served_context(served_context)
        if rendered_context:
            text_input = (
                text_input
                + "\n\nSESSION CONTEXT ENTRIES SERVED TO THE PREVIOUS ANSWER (id: content):\n"
                + rendered_context
            )

        result = await LLMGateway.acreate_structured_output(
            text_input=text_input,
            system_prompt=system_prompt,
            response_model=SessionTurnAnalysis,
        )
        return result
    except Exception as e:
        logger.warning(
            "Session turn analysis failed, proceeding with empty analysis: %s",
            e,
            exc_info=False,
        )
        return SessionTurnAnalysis()


async def detect_feedback(
    user_message: str, served_context: list | str | None = None
) -> SessionTurnAnalysis:
    """Compatibility wrapper for older call sites."""
    return await analyze_turn_for_session_context(user_message, served_context=served_context)
