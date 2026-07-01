"""Turn agent trace evidence into agent-profile session-context lessons.

Flow:

1. LIVE   (cheap, no LLM) — runs after each trace is stored; turns an errored step into a
   failure_lessons candidate built straight from its error text.
2. BATCH  (LLM) — proposes reasoning-heavy sections (tool_rules, success_patterns,
   workflow_state, environment_facts, and richer failure_lessons) over bounded pending trace
   windows. It runs in two places: periodically from the trace-write path (every
   TRACE_EXTRACTION_INTERVAL traces, awaited inline in SessionManager.add_agent_trace_step), and
   again at improve/session end to flush any pending tail. Because the periodic pass is awaited
   inline, a trace write that crosses the interval pays the latency of one LLM call.

Both feed the shared deterministic applier (apply_candidate_updates), so lessons are
confidence-gated and de-duplicated the same way QA lessons are. Fail-open: extraction never
raises into the trace write path — a failure just means no lesson this time.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.session_context_builder import (
    apply_candidate_updates,
    coerce_active_context_entries,
)
from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    AgentContextExtraction,
    CandidateFailureLessonUpdate,
    ContextProfile,
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
# Bound store growth across repeated improve() runs: cap new lessons per run, and show the
# extractor the lessons that already exist so it does not re-propose or reword them.
MAX_BATCH_LESSONS = 5
EXISTING_LESSONS_SHOWN = 40
MAX_ERROR_TEXT_CHARS = 600

# Mid-session extraction: every N new traces, re-read a small overlap so the LLM sees enough
# sequence context to infer patterns without running on every tool call.
TRACE_EXTRACTION_INTERVAL = 10
TRACE_EXTRACTION_OVERLAP = 3
TRACE_EXTRACTION_STATE_ID = "__agent_context_extraction_state__"
TRACE_EXTRACTION_STATE_KIND = "agent_context_extraction_state"

_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_BEARER_RE = re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)"
    r"\s*([=:])\s*([^\s,;]+)"
)
_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_LONG_NUMBER_RE = re.compile(r"\b\d{8,}\b")


@dataclass(frozen=True, slots=True)
class TraceExtractionPlan:
    """A bounded trace window to extract and the watermark to save afterward."""

    total_trace_count: int
    window_size: int


def _sanitize_error_text(error_message: str, max_chars: int = MAX_ERROR_TEXT_CHARS) -> str:
    """Redact sensitive and volatile values before errors become reusable context."""
    error = " ".join(str(error_message or "").split())
    error = _BEARER_RE.sub(r"\1 [redacted]", error)
    error = _SECRET_ASSIGNMENT_RE.sub(r"\1\2[redacted]", error)
    error = _JWT_RE.sub("[redacted]", error)
    error = _LONG_HEX_RE.sub("[redacted]", error)
    error = _UUID_RE.sub("[uuid]", error)
    error = _LONG_NUMBER_RE.sub("[number]", error)
    return error[:max_chars]


def _failure_lesson_content(origin_function: str, error_message: str) -> str:
    """Build a compact failure lesson from an errored trace step."""
    origin = origin_function.strip() or "a tool"
    error = _sanitize_error_text(error_message, max_chars=MAX_CONTEXT_CONTENT_CHARS)
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


def _trace_window_size(pending_trace_count: int, overlap: int, max_window: int) -> int:
    """Return the bounded trace window size for pending-trace extraction."""
    return max(1, min(max_window, pending_trace_count + overlap))


def _extract_state_row(raw_entries: list) -> dict | None:
    """Find this session's internal agent-extraction watermark row, if present."""
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        if raw.get("id") == TRACE_EXTRACTION_STATE_ID:
            return raw
        if raw.get("kind") == TRACE_EXTRACTION_STATE_KIND:
            return raw
    return None


async def _get_processed_trace_count(session_manager, user_id: str, session_id: str) -> int:
    """Read the count watermark. Missing or malformed state means no traces processed yet."""
    raw_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    row = _extract_state_row(raw_entries)
    if row is None:
        return 0
    try:
        return max(0, int(row.get("processed_trace_count") or 0))
    except (TypeError, ValueError):
        return 0


async def _save_processed_trace_count(
    session_manager, user_id: str, session_id: str, processed_trace_count: int
) -> None:
    """Persist the count watermark as an internal non-rendered session-context row."""
    payload = {
        "id": TRACE_EXTRACTION_STATE_ID,
        "kind": TRACE_EXTRACTION_STATE_KIND,
        "processed_trace_count": max(0, int(processed_trace_count)),
        "updated_at": datetime.now(datetime.timezone.utc).isoformat(),
    }
    updated = await session_manager.update_session_context_entry(
        user_id=user_id,
        session_id=session_id,
        entry_id=TRACE_EXTRACTION_STATE_ID,
        merge=payload,
    )
    if not updated:
        await session_manager.create_session_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_dump=payload,
        )


async def _plan_pending_extraction(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    min_new_traces: int,
    overlap: int,
    max_window: int,
) -> TraceExtractionPlan | None:
    """Load trace-count state and decide whether a new extraction window is due."""
    total_trace_count = await session_manager.get_agent_trace_count(
        user_id=user_id, session_id=session_id
    )
    processed_count = await _get_processed_trace_count(session_manager, user_id, session_id)
    pending_count = max(0, total_trace_count - processed_count)
    if pending_count < min_new_traces:
        return None

    return TraceExtractionPlan(
        total_trace_count=total_trace_count,
        window_size=_trace_window_size(pending_count, overlap, max_window),
    )


def _trace_line(entry) -> str:
    """Render one trace step as a compact line for the batch prompt."""
    parts = [f"- {entry.origin_function} [{entry.status}]"]
    if entry.session_feedback:
        parts.append(f"feedback: {entry.session_feedback}")
    if entry.error_message:
        parts.append(f"error: {_sanitize_error_text(entry.error_message)}")
    sanitized_return = sanitize_value(entry.method_return_value)
    if sanitized_return not in (None, "", {}, []):
        serialized = json.dumps(sanitized_return, ensure_ascii=False)[:BATCH_RETURN_CHARS]
        parts.append(f"output: {serialized}")
    return " | ".join(parts)


def build_trace_batch(traces) -> str:
    """Join trace steps into one prompt blob; empty when there is nothing to summarize."""
    lines = [_trace_line(entry) for entry in traces]
    return "\n".join(line for line in lines if line.strip())


async def _existing_agent_lessons(session_manager, user_id: str, session_id: str) -> list:
    """Load this session's current agent-profile lessons (capped for the prompt)."""
    raw_entries = await session_manager.get_session_context_entries(
        user_id=user_id, session_id=session_id
    )
    agent_entries = [
        entry
        for entry in coerce_active_context_entries(raw_entries)
        if entry.context_profile == ContextProfile.AGENT.value
    ]
    return agent_entries[:EXISTING_LESSONS_SHOWN]


def build_extraction_input(trace_batch: str, existing_lessons: list) -> str:
    """Combine already-stored lessons and new traces into one prompt input.

    Showing existing lessons lets the model avoid re-proposing or rewording them, which is what
    keeps the store from growing across repeated improve() runs.
    """
    sections = []
    if existing_lessons:
        rendered = "\n".join(f"- [{entry.section}] {entry.content}" for entry in existing_lessons)
        sections.append(
            "EXISTING LESSONS (already saved — do not repeat or reword these):\n" + rendered
        )
    sections.append("TRACES:\n" + trace_batch)
    return "\n\n".join(sections)


async def _extract_batch_from_traces(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    traces: list,
) -> list[str]:
    """LLM batch pass over an already-selected trace window. Raises on infrastructure errors."""
    if not traces:
        return []

    batch = build_trace_batch(traces)
    if not batch.strip():
        return []

    system_prompt = read_query_prompt(BATCH_PROMPT_FILE)
    if not system_prompt:
        raise RuntimeError("Batch agent-context extraction: system prompt not found")

    existing = await _existing_agent_lessons(session_manager, user_id, session_id)
    result = await LLMGateway.acreate_structured_output(
        text_input=build_extraction_input(batch, existing),
        system_prompt=system_prompt,
        response_model=AgentContextExtraction,
    )
    candidates = list(result.lessons)[:MAX_BATCH_LESSONS]
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


async def extract_batch_agent_context(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    last_n: int = BATCH_TRACE_LIMIT,
) -> list[str]:
    """LLM batch pass: read recent traces, propose new agent lessons, and store them.

    The prompt is shown the session's existing agent lessons so the model only proposes
    genuinely new ones, and the output is capped at ``MAX_BATCH_LESSONS`` — together these bound
    store growth across repeated improve() runs. Exact-content dedup in the applier is the final
    backstop. Fail-open -> [].
    """
    try:
        traces = await session_manager.get_agent_trace_session(
            user_id=user_id, session_id=session_id, last_n=last_n
        )
        return await _extract_batch_from_traces(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            traces=traces,
        )
    except Exception as error:
        logger.warning("Batch agent-context extraction failed open: %s", error)
        return []


async def extract_pending_agent_context(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    min_new_traces: int = TRACE_EXTRACTION_INTERVAL,
    overlap: int = TRACE_EXTRACTION_OVERLAP,
    max_window: int = BATCH_TRACE_LIMIT,
) -> list[str]:
    """Extract agent lessons from unprocessed traces using one shared watermark policy.

    ``min_new_traces`` controls the trigger:
    - trace-write path uses the interval default, so extraction happens periodically;
    - improve/session-end callers can pass ``1`` to flush any pending traces before distillation.

    The LLM receives only the latest ``pending + overlap`` trace steps, capped by ``max_window``.
    The watermark advances only after the extraction attempt completes without raising.
    Fail-open -> [].
    """
    if min_new_traces <= 0:
        return []
    try:
        plan = await _plan_pending_extraction(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            min_new_traces=min_new_traces,
            overlap=overlap,
            max_window=max_window,
        )
        if plan is None:
            return []

        traces = await session_manager.get_agent_trace_session(
            user_id=user_id,
            session_id=session_id,
            last_n=plan.window_size,
        )
        touched = await _extract_batch_from_traces(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            traces=traces,
        )
        await _save_processed_trace_count(
            session_manager,
            user_id,
            session_id,
            processed_trace_count=plan.total_trace_count,
        )
        return touched
    except Exception as error:
        logger.warning("Pending agent-context extraction failed open: %s", error)
        return []
