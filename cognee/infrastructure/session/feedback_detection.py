"""Automatic feedback detection from user messages via LLM."""

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.feedback_models import FeedbackDetectionResult
from cognee.shared.logging_utils import get_logger

logger = get_logger("feedback_detection")


async def detect_feedback(user_message: str) -> FeedbackDetectionResult:
    """
    Analyze a user message to detect whether it contains feedback about a previous response.

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

        result = await LLMGateway.acreate_structured_output(
            text_input=user_message.strip(),
            system_prompt=system_prompt,
            response_model=FeedbackDetectionResult,
        )
        return (
            result
            if isinstance(result, FeedbackDetectionResult)
            else FeedbackDetectionResult(feedback_detected=False)
        )
    except Exception as e:
        logger.warning(
            "Feedback detection failed, proceeding with no feedback detected: %s", e, exc_info=False
        )
        return FeedbackDetectionResult(feedback_detected=False)
