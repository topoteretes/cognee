from typing import Annotated, Literal

from pydantic import BaseModel, Field

from cognee.infrastructure.databases.cache import SessionAgentTraceEntry, SessionQAEntry
from cognee.modules.recall.types.SearchResultItem import SearchResultItem


class ResponseQAEntry(SessionQAEntry):
    source: Literal["session"]


class ResponseAgentTraceEntry(SessionAgentTraceEntry):
    source: Literal["trace"]


class ResponseSessionContextEntry(BaseModel):
    source: Literal["session_context"]
    content: str
    context_profile: str


class ResponseGraphEntry(SearchResultItem):
    source: Literal["graph"]


class ResponseCodeEntry(SearchResultItem):
    """One deterministic code-graph fact from the recall "code" scope.

    Same normalized shape as graph entries (kind CODE, payload under
    ``raw``) — only the source discriminator differs, so callers can
    route code facts separately from semantic graph results.
    """

    source: Literal["code"]


class ResponseToolEntry(BaseModel):
    """One tool invocation's result from the recall "tools" scope.

    Generic across tools: ``tool_name`` discriminates the tool (only
    ``text_to_sql`` in v1) and ``structured`` carries the tool-specific
    payload, so adding a tool never changes this union. Secrets (connection
    strings) never appear here.
    """

    source: Literal["tools"]
    tool_name: str
    question: str
    text: str
    success: bool = True
    error: str | None = None
    structured: dict | None = None


class ResponseSkillEntry(BaseModel):
    """One skill surfaced by the deterministic skill gate.

    Metadata-only: ``skill`` carries the projected Skill fields and never the
    procedure body — progressive disclosure keeps bodies behind the
    ``load_skill`` tool or ``GET /skills/{skill_id}``. ``text`` is a
    renderable "name: description" line; ``score`` is the raw vector distance
    (lower is better) when available.
    """

    source: Literal["skills"]
    text: str
    skill: dict
    score: float | None = None


class ResponseMarkerEntry(BaseModel):
    """System-generated marker (not data), e.g. "memory still warming up".

    ``text`` carries a human-readable message so generic consumers that fall
    back to text rendering display something sensible.

    ``status`` values: ``memory_warming_up`` (nothing searchable yet, no
    results), ``build_failed`` (nothing searchable, the last build errored),
    and ``memory_<state>`` while the memory is searchable but its graph is not
    built yet -- ``memory_cached`` (keyword search over the cached documents
    answered), ``memory_indexed`` / ``memory_extracting`` (chunk search
    answered), ``memory_caching`` / ``memory_indexing`` (a stage is still
    running), ``memory_indexing_failed`` / ``memory_extraction_failed`` (an
    earlier stage's results still answer). Those markers are appended after
    the results; the two cold ones are returned alone.
    """

    source: Literal["system"]
    status: str
    text: str
    datapoint_count: int
    threshold: int
    # Populated for failure states: the root cause of the last errored run,
    # so clients can show WHY memory has fewer answers than expected instead
    # of an unexplained result.
    error_class: str | None = None
    error_message: str | None = None


RecallResponse = Annotated[
    ResponseQAEntry
    | ResponseAgentTraceEntry
    | ResponseSessionContextEntry
    | ResponseGraphEntry
    | ResponseCodeEntry
    | ResponseToolEntry
    | ResponseSkillEntry
    | ResponseMarkerEntry,
    Field(discriminator="source"),
]
