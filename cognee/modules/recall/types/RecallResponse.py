from typing import Annotated, Literal

from pydantic import BaseModel, Field

from cognee.infrastructure.databases.cache import SessionAgentTraceEntry, SessionQAEntry
from cognee.modules.recall.types.SearchResultItem import SearchResultItem


class ResponseQAEntry(SessionQAEntry):
    source: Literal["session"]
    score: float | None = None


class ResponseAgentTraceEntry(SessionAgentTraceEntry):
    source: Literal["trace"]
    score: float | None = None


class ResponseGraphContextEntry(BaseModel):
    source: Literal["graph_context"]
    content: str


class ResponseSessionContextEntry(BaseModel):
    source: Literal["session_context"]
    content: str
    context_profile: str


class ResponseGraphEntry(SearchResultItem):
    source: Literal["graph"]


RecallResponse = Annotated[
    ResponseQAEntry
    | ResponseAgentTraceEntry
    | ResponseGraphContextEntry
    | ResponseSessionContextEntry
    | ResponseGraphEntry,
    Field(discriminator="source"),
]
