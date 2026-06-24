"""Turn agent trace evidence into agent-profile session-context lessons.

Flow:

1. LIVE  (cheap, no LLM) — runs after each trace is stored; turns an errored step into a
   failure_lessons candidate built straight from its error text.

The reasoning-heavier agent sections (tool_rules, success_patterns, workflow_state,
environment_facts) need judgment across multiple traces and are left to the LLM batch pass.

Both feed the shared deterministic applier (apply_candidate_updates), so lessons are
confidence-gated and de-duplicated the same way QA lessons are. Fail-open: extraction never
raises into the trace write path — a failure just means no lesson this time.
"""

from cognee.infrastructure.session.session_context_builder import apply_candidate_updates
from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    CandidateFailureLessonUpdate,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("agent_context_extraction")

# Deterministic candidates have no natural confidence; an errored step is a strong signal,
# so it clears the MIN_CANDIDATE_CONFIDENCE (0.75) gate comfortably.
LIVE_FAILURE_CONFIDENCE = 0.85


def _failure_lesson_content(origin_function: str, error_message: str) -> str:
    """Build a compact failure lesson from an errored trace step."""
    origin = origin_function.strip() or "a tool"
    error = " ".join(error_message.split())
    return f"{origin} failed: {error}"[:MAX_CONTEXT_CONTENT_CHARS]


def build_live_agent_candidates(
    *, origin_function: str, status: str, error_message: str
) -> list[CandidateFailureLessonUpdate]:
    """Deterministic, no-LLM candidates from one trace step.

    Scoped to the single section whose content already sits on the trace: an errored step with
    an error message becomes a failure_lessons candidate. Everything else is the batch pass's job.
    """
    if status != "error" or not error_message.strip():
        return []
    content = _failure_lesson_content(origin_function, error_message)
    return [CandidateFailureLessonUpdate(content=content, confidence=LIVE_FAILURE_CONFIDENCE)]


async def extract_live_agent_context(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    trace_id: str,
    origin_function: str,
    status: str,
    error_message: str,
) -> list[str]:
    """Store agent lessons derivable from one just-saved trace step. Fail-open -> []."""
    try:
        candidates = build_live_agent_candidates(
            origin_function=origin_function, status=status, error_message=error_message
        )
        if not candidates:
            return []
        return await apply_candidate_updates(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            source_id=trace_id,
            candidates=candidates,
        )
    except Exception as error:
        logger.warning("Live agent-context extraction failed open: %s", error)
        return []
