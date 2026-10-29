"""Public conflict inspection and resolution results."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from cognee.api.DTO import OutDTO


class ConflictFact(OutDTO):
    id: UUID
    text: str
    source_id: str
    target_id: str
    relationship: str
    status: str
    sources: list[dict] = Field(default_factory=list)


class ConflictDetail(OutDTO):
    id: UUID
    dataset_id: UUID
    about_id: str
    attribute: str
    kind: Literal["time_varying", "fixed"]
    status: Literal["resolved", "unresolved"]
    text: str
    facts: list[ConflictFact] = Field(default_factory=list)
    resolution_source: Literal["review", "user"] = "review"
    selected_fact_id: UUID | None = None
    selected_fact: str | None = None
    resolved_by: UUID | None = None
    resolved_at: str | None = None
    resolution_reason: str | None = None
    write_pending: bool = False
