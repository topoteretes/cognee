"""Opt-in capture of every LLM call cognee makes (part of SDK-529).

``capture_llm_call()`` turns one call into an ``LLMCallRecord`` and hands it to
a background writer. It is a no-op unless ``LLM_CALL_CAPTURE_ENABLED=true``, and
it never raises: capture is observability, so a failure is logged and the LLM
call it describes is unaffected. The stage and run the call belongs to are read
from the current context, so callers pass only what they know about the call.

Records are local JSONL files (see ``writer``); shipping them anywhere else is
a deployment's choice, not this module's.
"""

import os
from datetime import datetime

from cognee.shared.logging_utils import get_logger

from .config import LLMCallCaptureConfig, get_llm_call_capture_config
from .record import SCHEMA_VERSION, LLMCallRecord, build_llm_call_record
from .writer import JsonlWriter

logger = get_logger("llm_call_capture")

_writer: JsonlWriter | None = None


def _get_writer(directory: str) -> JsonlWriter:
    """The process's writer, replaced after a fork so parent and child never share a file."""
    global _writer
    if _writer is None or _writer.pid != os.getpid():
        _writer = JsonlWriter(directory)
    return _writer


def capture_llm_call(
    *,
    started_at: datetime,
    duration_ms: float,
    model: str | None,
    framework: str,
    response_type: str,
    system_prompt: str,
    text_input: str,
    output: str | None,
    status: str,
    error_class: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
) -> None:
    """Record one LLM call when capture is on. Never raises.

    ``output`` is the call's result as text, or None when the call raised
    (``status="error"`` with its ``error_class``).
    """
    try:
        config = get_llm_call_capture_config()
        if not config.enabled:
            return

        from cognee.context_global_variables import current_pipeline_stage
        from cognee.modules.operations.usage_accumulator import get_parent_run_id

        record = build_llm_call_record(
            include_content=config.include_content,
            system_prompt=system_prompt,
            text_input=text_input,
            output=output,
            started_at=started_at,
            duration_ms=duration_ms,
            stage=current_pipeline_stage.get(),
            run_id=get_parent_run_id(),
            model=model,
            framework=framework,
            response_type=response_type,
            status=status,
            error_class=error_class,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        _get_writer(config.directory).put(record.model_dump_json())
    except Exception:
        logger.warning("LLM call capture failed; the call itself is unaffected", exc_info=True)


def flush_llm_call_capture(timeout: float = 5.0) -> bool:
    """Wait until every captured record is on disk (scripts and tests). True if nothing is pending."""
    return True if _writer is None else _writer.flush(timeout)


__all__ = [
    "SCHEMA_VERSION",
    "LLMCallCaptureConfig",
    "LLMCallRecord",
    "capture_llm_call",
    "flush_llm_call_capture",
    "get_llm_call_capture_config",
]
