"""Opt-in capture of every LLM call cognee makes (part of SDK-529).

``capture_llm_call()`` turns one call into an ``LLMCallRecord`` and hands it to
a background writer. It is a no-op unless ``LLM_CALL_CAPTURE_ENABLED=true``, and
it never raises: capture is observability, so a failure is logged and the LLM
call it describes is unaffected. The stage, caller, run and chunk the call belongs
to are read from the current context, so callers pass only what they know about the call;
a task that makes one LLM call per chunk wraps each in ``chunk_scope(chunk.id)``.

``stage`` is set by ``pipeline_stage()`` in graph extraction (``extraction``),
LLM summaries (``summarization``) and answer generation (``query``). ``caller``
is set by ``caller_scope()`` in:

- ``search_in_datasets_context``: every LLM call a search or recall makes
- ``improve.<stage name>``: each improve stage, e.g. ``improve.distill_sessions``
- ``analyze_turn_for_session_context``: session turn analysis
- ``test_llm_connection``: the startup connection check
- ``generate_agent_trace_feedback``: agent trace step summaries
- ``_extract_batch_from_traces``: agent-context lessons from traces
- ``detect_pii._llm_assess``: LLM-assisted PII detection when presorting files

Calls made elsewhere are still captured, without a ``caller``. With tracing on,
the default ``litellm_native`` adapter also writes both to its LLM spans, as
``cognee.pipeline.stage`` and ``cognee.llm.caller``.

Records are local JSONL files (see ``writer``); shipping them anywhere else is
a deployment's choice, not this module's.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from cognee.shared.logging_utils import get_logger

from .config import LLMCallCaptureConfig, get_llm_call_capture_config
from .record import SCHEMA_VERSION, LLMCallRecord, build_llm_call_record
from .writer import JsonlWriter

logger = get_logger("llm_call_capture")

_writer: JsonlWriter | None = None

current_chunk_id: ContextVar[UUID | None] = ContextVar("llm_call_chunk_id", default=None)


@contextmanager
def chunk_scope(chunk_id: UUID) -> Iterator[None]:
    """Tag the LLM calls made inside this block with the chunk they are about.

    Set it inside the per-chunk coroutine, not around an ``asyncio.gather`` of
    them: each gathered coroutine runs in its own copy of the context, so the
    chunks never see each other's id.
    """
    token = current_chunk_id.set(chunk_id)
    try:
        yield
    finally:
        current_chunk_id.reset(token)


def _get_writer(directory: str) -> JsonlWriter:
    """The process's writer, replaced after a fork so parent and child never share a file."""
    global _writer
    if _writer is None or _writer.pid != os.getpid():
        _writer = JsonlWriter(directory)
    return _writer


def capture_enabled() -> bool:
    """Whether capture is on. Never raises.

    A capture that is on but misconfigured reads as on, so ``capture_llm_call``
    runs and logs why it cannot record.
    """
    try:
        return get_llm_call_capture_config().enabled
    except ValueError:
        return True


def _as_text(output: Any) -> str | None:
    if output is None or isinstance(output, str):
        return output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    return str(output)


def _record_json(record: LLMCallRecord) -> str:
    """The record as one JSON line, without ``chunk_id`` or ``caller`` when not supplied."""
    omitted = {name for name in ("chunk_id", "caller") if getattr(record, name) is None}
    return record.model_dump_json(exclude=omitted or None)


def capture_llm_call(
    *,
    started_at: datetime,
    duration_ms: float,
    model: str | None,
    framework: str,
    response_type: str,
    system_prompt: str,
    text_input: str,
    output: Any,
    status: str,
    error_class: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
) -> None:
    """Record one LLM call when capture is on. Never raises.

    ``output`` is the call's result (text, or a pydantic model, recorded as its
    JSON), or None when the call raised (``status="error"`` with its ``error_class``).
    """
    try:
        config = get_llm_call_capture_config()
        if not config.enabled:
            return

        from cognee.context_global_variables import current_caller, current_pipeline_stage
        from cognee.modules.operations.usage_accumulator import get_parent_run_id

        record = build_llm_call_record(
            include_content=config.include_content,
            system_prompt=system_prompt,
            text_input=text_input,
            output=_as_text(output),
            started_at=started_at,
            duration_ms=duration_ms,
            stage=current_pipeline_stage.get(),
            caller=current_caller.get(),
            run_id=get_parent_run_id(),
            chunk_id=current_chunk_id.get(),
            model=model,
            framework=framework,
            response_type=response_type,
            status=status,
            error_class=error_class,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        _get_writer(config.directory).put(_record_json(record))
    except Exception:
        logger.warning("LLM call capture failed; the call itself is unaffected", exc_info=True)


def flush_llm_call_capture(timeout: float = 5.0) -> bool:
    """Wait until every captured record is on disk (scripts and tests).

    True once they are; False on timeout or after a write error lost records.
    """
    return True if _writer is None else _writer.flush(timeout)


__all__ = [
    "SCHEMA_VERSION",
    "LLMCallCaptureConfig",
    "LLMCallRecord",
    "capture_enabled",
    "capture_llm_call",
    "chunk_scope",
    "current_chunk_id",
    "flush_llm_call_capture",
    "get_llm_call_capture_config",
]
