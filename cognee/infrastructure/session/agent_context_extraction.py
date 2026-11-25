"""Turn agent trace evidence into agent-profile session-context lessons.

Flow:

1. LIVE   (cheap, no LLM) — runs after each trace is stored; turns an errored step into a
   failure_lessons candidate built straight from its error text.
2. BATCH  (LLM) — proposes reasoning-heavy sections (tool_rules, success_patterns,
   workflow_state, environment_facts, and richer failure_lessons) over bounded pending trace
   windows. It runs in two places: periodically from the trace-write path (every
   TRACE_EXTRACTION_INTERVAL traces, as a background task, at most one per session, started
   by SessionManager after a trace is stored), and again at improve/session end to flush any
   pending tail. The trace write, and a capture's receipt, never wait for the LLM call.

Both feed the shared deterministic applier (apply_candidate_updates), so lessons are
confidence-gated and de-duplicated the same way QA lessons are. Fail-open: extraction never
raises into the trace write path — a failure just means no lesson this time, and no new
attempt before the next interval.
"""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.pipeline_stage import caller_scope
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
from cognee.infrastructure.session.session_persist_watermark import StateRowWatermark
from cognee.modules.agent_memory.sanitization import sanitize_value
from cognee.modules.preflight import llm_available
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
# Count watermark over trace steps already turned into lessons: the shared
# state-row implementation every improve() stage uses (session_persist_watermark).
TRACE_EXTRACTION_WATERMARK = StateRowWatermark(
    state_id=TRACE_EXTRACTION_STATE_ID,
    state_kind=TRACE_EXTRACTION_STATE_KIND,
    field="processed_trace_count",
)
# The trace count at which the last window was started, written before its LLM call.
# Automatic runs count new traces from here, so a window that never finished (the
# call failed or was cancelled) is retried after the next interval, not on every
# trace write. The watermark above says what was extracted; this says when to try.
TRACE_EXTRACTION_ATTEMPT = StateRowWatermark(
    state_id=TRACE_EXTRACTION_STATE_ID,
    state_kind=TRACE_EXTRACTION_STATE_KIND,
    field="attempted_trace_count",
)

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
    """The traces to extract, from the processed watermark to the total."""

    total_trace_count: int
    processed_count: int


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
    return [
        CandidateFailureLessonUpdate(
            content=content,
            confidence=LIVE_FAILURE_CONFIDENCE,
            evidence=content,
        )
    ]


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
        logger.warning("Live agent-context extraction failed open: %s", error, exc_info=True)
        return []


def _trace_windows(
    processed: int, total: int, overlap: int, max_window: int
) -> list[tuple[int, int, int]]:
    """``(context_start, start, end)`` of each window over traces ``processed..total``,
    oldest first. A window holds at most ``max_window`` traces: up to ``overlap``
    already-seen ones for context, then the new ones it seals."""
    step = max(1, max_window - overlap)
    return [
        (max(0, start - overlap), start, min(start + step, total))
        for start in range(processed, total, step)
    ]


async def _effective_count(
    watermark: StateRowWatermark, rows, session_manager, user_id: str, session_id: str, total: int
) -> int:
    """A count above the trace total is stale (the traces expired); reset it in the store,
    or it would count again once the new traces reach it."""
    stored = watermark.count_from_rows(rows)
    effective = StateRowWatermark.resolve_effective(stored, total, session_id=session_id)
    if effective != stored:
        await watermark.write_count(session_manager, user_id, session_id, count=effective)
    return effective


async def _plan_pending_extraction(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    min_new_traces: int,
    overlap: int,
    max_window: int,
    automatic: bool,
) -> TraceExtractionPlan | None:
    """Load trace-count state and decide whether a new extraction window is due.

    New traces are counted from the last finished window, or for an automatic run
    from the last started one. The window always covers every trace not yet
    extracted, so the traces of a window that never finished are not lost.
    """
    total_trace_count = await session_manager.get_agent_trace_count(
        user_id=user_id, session_id=session_id
    )
    rows = await session_manager.get_session_context_entries(user_id=user_id, session_id=session_id)
    processed_count, attempted_count = [
        await _effective_count(
            watermark, rows, session_manager, user_id, session_id, total_trace_count
        )
        for watermark in (TRACE_EXTRACTION_WATERMARK, TRACE_EXTRACTION_ATTEMPT)
    ]
    counted_from = max(processed_count, attempted_count) if automatic else processed_count
    if total_trace_count - counted_from < min_new_traces:
        return None
    return TraceExtractionPlan(total_trace_count=total_trace_count, processed_count=processed_count)


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
    with caller_scope("_extract_batch_from_traces"):
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
    except Exception:
        # No blanket catch-to-[] here: an LLM outage must propagate so the
        # caller's watermark never advances over traces that were never
        # extracted (the same failed-vs-empty rule the distillation watermark
        # follows). Callers own the fail-open policy.
        logger.warning("Batch agent-context extraction failed", exc_info=True)
        raise


async def extract_pending_agent_context(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    min_new_traces: int = TRACE_EXTRACTION_INTERVAL,
    overlap: int = TRACE_EXTRACTION_OVERLAP,
    max_window: int = BATCH_TRACE_LIMIT,
    automatic: bool = True,
    admission: Callable[[], Awaitable[str | None]] | None = None,
) -> list[str]:
    """Extract agent lessons from unprocessed traces using one shared watermark policy.

    ``min_new_traces`` controls the trigger:
    - trace-write path uses the interval default, so extraction happens periodically;
    - improve/session-end callers can pass ``1`` to flush any pending traces before distillation.

    ``automatic`` counts new traces from the last started window, so a failing call is
    retried once per interval; ``False`` (improve) counts from the last finished one.
    ``admission`` is asked once per due window, before the trace read and the LLM call;
    a skip reason ends the run and counts as the window's attempt.

    Pending traces are sent in windows of at most ``max_window`` steps (``overlap`` of
    them already seen), oldest first, the watermark sealed after each window.
    The start of a window is recorded before its LLM call; the watermark advances only
    after the extraction attempt completes without raising. One window per session runs
    at a time (in-process lock): a concurrent caller waits, then plans from the new
    watermark.

    Failures PROPAGATE — the callers own the fail-open policy: the trace-write
    path wraps this in its own try (a failed extraction must never lose the
    already-stored trace row), and the improve stage catches per session and
    reports ``errored``. A blanket catch-to-[] here made that stage's errored
    branch unreachable and hid every outage as "nothing pending".
    """
    if not llm_available():
        return []
    if min_new_traces <= 0:
        return []

    # The trace write's background batch and improve's extract_agent_context stage
    # both run this: plan, call the LLM and advance the watermark under one lock per
    # session, so the second caller reads the advanced watermark instead of sending
    # the same window again, and a slow window cannot write its older count over the
    # count the other sealed. In-process only, like the coding extraction's lock.
    # Imported here, like the coding extraction does: the locks package must not
    # load with the session infrastructure.
    from cognee.infrastructure.locks import session_lock

    async def plan_window() -> TraceExtractionPlan | None:
        return await _plan_pending_extraction(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            min_new_traces=min_new_traces,
            overlap=overlap,
            max_window=max_window,
            automatic=automatic,
        )

    # The host check is the caller's code with no bound on how long it takes, and
    # improve() waits on this lock while it holds the dataset lock: ask it before
    # the lock, once a window is due, and plan again under the lock.
    skip_reason = None
    if admission is not None:
        if await plan_window() is None:
            return []
        skip_reason = await admission()

    async with session_lock(f"{user_id}:{session_id}", "agent_context_extraction"):
        plan = await plan_window()
        if plan is None:
            return []
        await TRACE_EXTRACTION_ATTEMPT.write_count(
            session_manager, user_id, session_id, count=plan.total_trace_count
        )
        if skip_reason:
            logger.info(
                "Agent-context extraction skipped by the host admission check (%s)", skip_reason
            )
            return []
        total = plan.total_trace_count
        # Read by position, not from the tail: trace writes take no lock, so traces
        # appended after the plan would shift a tail read (last_n) forward.
        first = max(0, plan.processed_count - overlap)
        traces = await session_manager.get_agent_trace_session(
            user_id=user_id, session_id=session_id, start=first, end=total
        )
        touched: list[str] = []
        # A backlog larger than one window (calls that failed for several intervals)
        # is walked oldest first, sealing each window, so no trace is skipped.
        for context_start, start, end in _trace_windows(
            plan.processed_count, total, overlap, max_window
        ):
            touched += await _extract_batch_from_traces(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                traces=traces[context_start - first : end - first],
            )
            await TRACE_EXTRACTION_WATERMARK.write_count(
                session_manager, user_id, session_id, count=end
            )
        return touched
