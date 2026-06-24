"""Turn agent trace evidence into agent-profile session-context lessons.

Flow:

1. LIVE   (cheap, no LLM) — runs after each trace is stored; turns an errored step into a
   failure_lessons candidate built straight from its error text.
2. BATCH  (LLM, off the hot path) — runs at the improve stage over a session's recent traces and
   proposes the reasoning-heavier sections (tool_rules, success_patterns, workflow_state,
   environment_facts, and richer failure_lessons).

Both feed the shared deterministic applier (apply_candidate_updates), so lessons are
confidence-gated and de-duplicated the same way QA lessons are. Fail-open: extraction never
raises into the trace write path — a failure just means no lesson this time.
"""

import json

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.session_context_builder import apply_candidate_updates
from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    AgentContextExtraction,
    CandidateFailureLessonUpdate,
)
from cognee.modules.agent_memory.sanitization import sanitize_value
from cognee.shared.logging_utils import get_logger

logger = get_logger("agent_context_extraction")

# Deterministic candidates have no natural confidence; an errored step is a strong signal,
# so it clears the MIN_CANDIDATE_CONFIDENCE (0.75) gate comfortably.
LIVE_FAILURE_CONFIDENCE = 0.85

# Batch pass: how many recent trace steps to read, and how much of each output to keep.
BATCH_TRACE_LIMIT = 40
BATCH_RETURN_CHARS = 600
BATCH_PROMPT_FILE = "agent_context_extraction_system.txt"


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


def _trace_line(entry) -> str:
    """Render one trace step as a compact line for the batch prompt."""
    parts = [f"- {entry.origin_function} [{entry.status}]"]
    if entry.session_feedback:
        parts.append(f"feedback: {entry.session_feedback}")
    if entry.error_message:
        parts.append(f"error: {entry.error_message}")
    sanitized_return = sanitize_value(entry.method_return_value)
    if sanitized_return not in (None, "", {}, []):
        serialized = json.dumps(sanitized_return, ensure_ascii=False)[:BATCH_RETURN_CHARS]
        parts.append(f"output: {serialized}")
    return " | ".join(parts)


def build_trace_batch(traces) -> str:
    """Join trace steps into one prompt blob; empty when there is nothing to summarize."""
    lines = [_trace_line(entry) for entry in traces]
    return "\n".join(line for line in lines if line.strip())


async def extract_batch_agent_context(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    last_n: int = BATCH_TRACE_LIMIT,
) -> list[str]:
    """LLM batch pass: read recent traces, propose agent lessons, and store them. Fail-open -> [].

    The applier de-duplicates against existing entries by exact content within (profile, section),
    so a batch lesson identical to a live one links instead of duplicating; reworded lessons
    coexist and are reconciled later by distillation.
    """
    try:
        traces = await session_manager.get_agent_trace_session(
            user_id=user_id, session_id=session_id, last_n=last_n
        )
        if not traces:
            return []

        batch = build_trace_batch(traces)
        if not batch.strip():
            return []

        system_prompt = read_query_prompt(BATCH_PROMPT_FILE)
        if not system_prompt:
            logger.warning("Batch agent-context extraction: system prompt not found")
            return []

        result = await LLMGateway.acreate_structured_output(
            text_input=batch,
            system_prompt=system_prompt,
            response_model=AgentContextExtraction,
        )
        candidates = list(result.lessons)
        if not candidates:
            return []

        # No single source trace for a batch lesson; link none and rely on content dedup.
        return await apply_candidate_updates(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            source_id="",
            candidates=candidates,
        )
    except Exception as error:
        logger.warning("Batch agent-context extraction failed open: %s", error)
        return []
