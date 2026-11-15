"""The record written for one LLM call.

Adding a field is backward compatible; removing or renaming one, or changing
its meaning, bumps ``SCHEMA_VERSION`` so readers can tell the shapes apart.
"""

import hashlib
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class LLMCallRecord(BaseModel):
    """One LLM call cognee made: what was asked, of which model, from where, with what result.

    ``system_prompt``, ``text_input`` and ``output`` are the only fields that can
    hold user content. They are set only when the record was built with content
    included (``content_included``); the lengths and the system prompt hash are
    always set, so scrubbed records can still be grouped by prompt and size.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    event_id: UUID = Field(default_factory=uuid4)
    started_at: datetime
    duration_ms: float

    stage: str | None
    run_id: UUID | None
    chunk_id: UUID | None = None

    model: str | None
    framework: str
    response_type: str
    status: Literal["ok", "error"]
    error_class: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    system_prompt_sha256: str
    system_prompt_chars: int
    text_input_chars: int
    output_chars: int | None

    content_included: bool
    system_prompt: str | None = None
    text_input: str | None = None
    output: str | None = None


def build_llm_call_record(
    *,
    include_content: bool,
    system_prompt: str,
    text_input: str,
    output: str | None,
    **fields,
) -> LLMCallRecord:
    """Build a record, keeping the three text fields only when ``include_content`` is set.

    ``output`` is None for a call that raised. Every other field is passed through.
    """
    return LLMCallRecord(
        system_prompt_sha256=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        system_prompt_chars=len(system_prompt),
        text_input_chars=len(text_input),
        output_chars=None if output is None else len(output),
        content_included=include_content,
        system_prompt=system_prompt if include_content else None,
        text_input=text_input if include_content else None,
        output=output if include_content else None,
        **fields,
    )
