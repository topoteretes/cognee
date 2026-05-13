"""Discriminated-union memory entries for remember()/recall().

Typed payloads let callers pass rich structured data to
``cognee.remember()`` — Q&A turns, agent trace steps, feedback
attachments — in addition to the legacy "blob of text/files" shape.
Each entry carries a literal ``type`` discriminator so the remember
dispatch can route to the right ``SessionManager`` method.

Raw data (str / bytes / file-like / list of the above) continues to
flow through the permanent add+cognify path unchanged.
"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel


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


MemoryEntry = Union[QAEntry, TraceEntry, FeedbackEntry]


# Tuple used at runtime for isinstance checks; Union itself isn't
# a valid isinstance target on older Python versions.
MEMORY_ENTRY_TYPES = (QAEntry, TraceEntry, FeedbackEntry)


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
