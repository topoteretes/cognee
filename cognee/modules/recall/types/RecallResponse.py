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


RecallResponse = Annotated[
    ResponseQAEntry
    | ResponseAgentTraceEntry
    | ResponseSessionContextEntry
    | ResponseGraphEntry
    | ResponseToolEntry,
    Field(discriminator="source"),
]
