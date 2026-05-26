"""Discriminated-union memory entries for remember()/recall().

Typed payloads let callers pass rich structured data to
``cognee.remember()`` — Q&A turns, agent trace steps, feedback
attachments, and skill-run scores — in addition to the legacy
"blob of text/files" shape.
Each entry carries a literal ``type`` discriminator so the remember
dispatch can route to the right ``SessionManager`` method.

Raw data (str / bytes / file-like / list of the above) continues to
flow through the permanent add+cognify path unchanged.
"""

from typing import Any, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class QAEntry(BaseModel):
    """A Q&A turn stored in the session cache.

    Represents a user question + assistant answer with optional
    retrieval context. Dispatched to ``SessionManager.add_qa``.
    """

    type: Literal["qa"] = "qa"
    question: str
    answer: str
    context: str = ""
    feedback_text: Optional[str] = None
    feedback_score: Optional[int] = None
    used_graph_element_ids: Optional[dict] = None


class TraceEntry(BaseModel):
    """One step of an agent trace.

    Structured representation of a tool/function call — origin,
    outcome, parameters, return value. Dispatched to
    ``SessionManager.add_agent_trace_step``.
    """

    type: Literal["trace"] = "trace"
    origin_function: str
    status: Literal["success", "error"] = "success"
    method_params: Optional[dict] = None
    method_return_value: Optional[Any] = None
    memory_query: str = ""
    memory_context: str = ""
    error_message: str = ""
    generate_feedback_with_llm: bool = False


class FeedbackEntry(BaseModel):
    """Feedback attached to an existing QA entry.

    Semantically an update rather than a new memory — carried through
    remember() for API minimalism. Dispatched to
    ``SessionManager.add_feedback``.
    """

    type: Literal["feedback"] = "feedback"
    qa_id: str
    feedback_text: Optional[str] = None
    feedback_score: Optional[int] = None


class SkillRunEntry(BaseModel):
    """A persisted execution record for a skill.

    This is graph-backed rather than session-cache-backed. It lets agents
    report explicit skill quality signals through ``cognee.remember()``
    without adding another public API surface.
    """

    type: Literal["skill_run"] = "skill_run"
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    selected_skill_id: str
    task_text: str = ""
    result_summary: str = ""
    success_score: Optional[float] = None
    feedback: float = 0.0
    error_type: str = ""
    error_message: str = ""
    started_at_ms: int = 0
    latency_ms: int = 0
    candidate_skill_ids: list[str] = Field(default_factory=list)
    task_pattern_id: str = ""
    router_version: str = ""
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    node_set: str = "skills"

    @field_validator("success_score")
    @classmethod
    def _validate_success_score(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("success_score must be in range [0.0, 1.0]")
        return value

    @field_validator("feedback")
    @classmethod
    def _validate_feedback(cls, value: float) -> float:
        if not -1.0 <= value <= 1.0:
            raise ValueError("feedback must be in range [-1.0, 1.0]")
        return value

    @field_validator("started_at_ms", "latency_ms")
    @classmethod
    def _validate_non_negative_ms(cls, value: int) -> int:
        if value < 0:
            raise ValueError("timestamp and latency fields must be non-negative")
        return value


MemoryEntry = Union[QAEntry, TraceEntry, FeedbackEntry, SkillRunEntry]


# Tuple used at runtime for isinstance checks; Union itself isn't
# a valid isinstance target on older Python versions.
MEMORY_ENTRY_TYPES = (QAEntry, TraceEntry, FeedbackEntry, SkillRunEntry)


RecallScope = Literal["auto", "graph", "session", "trace", "graph_context", "all"]


_VALID_SCOPES = {"auto", "graph", "session", "trace", "graph_context", "all"}


def normalize_scope(scope: Optional[Union[str, list[str]]]) -> list[str]:
    """Normalize the recall ``scope`` parameter to a concrete source list.

    Accepts ``None``, a single string, or a list of strings. Returns a
    deduplicated list of concrete sources (``graph``, ``session``,
    ``trace``, ``graph_context``). ``None`` and ``"auto"`` expand later
    based on whether a session_id is present; this function just
    canonicalizes the input.

    Raises ``ValueError`` on unknown scope names.
    """
    if scope is None:
        return ["auto"]
    if isinstance(scope, str):
        scopes = [scope]
    else:
        scopes = list(scope)

    unknown = [s for s in scopes if s not in _VALID_SCOPES]
    if unknown:
        raise ValueError(
            f"Unknown recall scope(s): {unknown}. Valid values: {sorted(_VALID_SCOPES)}"
        )

    if "all" in scopes:
        return ["graph", "session", "trace", "graph_context"]

    # Dedupe while preserving order
    seen: set = set()
    out: list[str] = []
    for s in scopes:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
