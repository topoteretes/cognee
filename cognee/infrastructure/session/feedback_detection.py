"""Automatic feedback detection from user messages via LLM."""

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult
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


async def detect_feedback(
    user_message: str, served_context: list | str | None = None
) -> FeedbackDetectionResult:
    """
    Analyze a user message to detect whether it contains feedback about a previous response.

    When ``served_context`` is provided (a pre-rendered string or a list of session-context
    entries served to the previous answer), it is appended to the LLM input so the single
    feedback call can additionally produce ``served_context_ratings`` and
    ``candidate_context_updates``. This adds no extra LLM call.

    Returns a FeedbackDetectionResult with feedback_detected, feedback_text, and
    feedback_score. On LLM failure or timeout, returns a result with feedback_detected=False
    so the main completion flow is never blocked.
    """
    if not (user_message and str(user_message).strip()):
        return FeedbackDetectionResult(feedback_detected=False)

    try:
        system_prompt = read_query_prompt("feedback_detection_system.txt")
        if not system_prompt:
            logger.warning("Feedback detection: system prompt not found, skipping")
            return FeedbackDetectionResult(feedback_detected=False)

        text_input = user_message.strip()
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
            response_model=FeedbackDetectionResult,
        )
        return result
    except Exception as e:
        logger.warning(
            "Feedback detection failed, proceeding with no feedback detected: %s", e, exc_info=False
        )
        return FeedbackDetectionResult(feedback_detected=False)
