"""Per-step feedback generation for agent trace entries.

A trace step records one agent method call. Its ``session_feedback`` is a one-line
summary of what the step did — generated from the step's return value via an LLM, or
falling back to a deterministic success/failure line. Storage of trace steps stays in
``SessionManager``; only this summary logic lives here.
"""

import json
from typing import Any

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.feedback_models import AgentTraceFeedbackSummary
from cognee.modules.agent_memory.sanitization import sanitize_value
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_agent_trace")


def fallback_agent_trace_feedback(
    origin_function: str,
    status: str,
    error_message: str = "",
) -> str:
    """Deterministic feedback for a trace step, used when no LLM summary is available."""
    normalized_origin = origin_function.strip()
    normalized_status = status.strip().lower()
    normalized_error = error_message.strip()

    if normalized_status == "error":
        if normalized_error:
            return f"{normalized_origin} failed. Reason: {normalized_error}."
        return f"{normalized_origin} failed."
    return f"{normalized_origin} succeeded."


async def generate_agent_trace_feedback(
    *,
    origin_function: str,
    status: str,
    method_return_value: Any,
    error_message: str = "",
) -> str:
    """Summarize a trace step from its return value, or fall back deterministically.

    Fail-open: any LLM/prompt failure returns the deterministic fallback.
    """
    fallback_feedback = fallback_agent_trace_feedback(
        origin_function=origin_function,
        status=status,
        error_message=error_message,
    )

    if method_return_value is None:
        return fallback_feedback

    try:
        system_prompt = read_query_prompt("agent_trace_feedback_summary_system.txt")
        if not system_prompt:
            logger.warning("Agent trace feedback: system prompt not found, using fallback")
            return fallback_feedback

        sanitized_return_value = sanitize_value(method_return_value)
        serialized_return_value = json.dumps(sanitized_return_value, ensure_ascii=False)

        result = await LLMGateway.acreate_structured_output(
            text_input=serialized_return_value,
            system_prompt=system_prompt,
            response_model=AgentTraceFeedbackSummary,
        )
        session_feedback = result.session_feedback.strip()
        return session_feedback if session_feedback else fallback_feedback
    except Exception as e:
        logger.warning(
            "Agent trace feedback generation failed, using fallback: %s",
            e,
            exc_info=False,
        )
        return fallback_feedback
