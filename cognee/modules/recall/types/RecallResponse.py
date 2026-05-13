from typing import Annotated, Literal

from pydantic import BaseModel, Field

from cognee.infrastructure.databases.cache import SessionAgentTraceEntry, SessionQAEntry
from cognee.modules.recall.types.SearchResultItem import SearchResultItem


class ResponseQAEntry(SessionQAEntry):
    source: Literal["session"]


class ResponseAgentTraceEntry(SessionAgentTraceEntry):
    source: Literal["trace"]


class ResponseGraphContextEntry(BaseModel):
    source: Literal["graph_context"]
    content: str


class ResponseGraphEntry(SearchResultItem):
    source: Literal["graph"]


RecallResponse = Annotated[
    ResponseQAEntry | ResponseAgentTraceEntry | ResponseGraphContextEntry | ResponseGraphEntry,
    Field(discriminator="source"),
]
