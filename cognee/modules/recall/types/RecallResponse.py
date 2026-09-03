from typing import Annotated, Literal, Optional

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
    error: Optional[str] = None
    structured: Optional[dict] = None


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
    score: Optional[float] = None


class ResponseMarkerEntry(BaseModel):
    """System-generated marker (not data), e.g. "memory still warming up".

    ``text`` carries a human-readable message so generic consumers that fall
    back to text rendering display something sensible.
    """

    source: Literal["system"]
    status: str
    text: str
    datapoint_count: int
    threshold: int
    # Populated when status == "build_failed": the root cause of the last
    # errored build, so clients can show WHY memory has no answers instead
    # of an unexplained empty result.
    error_class: Optional[str] = None
    error_message: Optional[str] = None


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
