from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class EvidenceLocator(BaseModel):
    dataset_id: UUID
    kind: Literal["segment", "graph_node", "graph_edge"]
    artifact_id: UUID
    version: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    source_node_id: UUID | None = None
    target_node_id: UUID | None = None
    relationship_name: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def edge_address(self):
        if self.kind == "graph_edge" and not (
            self.source_node_id and self.target_node_id and self.relationship_name
        ):
            raise ValueError("Graph edges require both endpoints and a relationship name")
        return self


class EvidenceItem(BaseModel):
    reference: EvidenceLocator
    status: Literal["available", "changed", "unavailable", "unsupported", "error"]
    citation: str | None = None
    title: str | None = None
    source: dict = Field(default_factory=dict)
    # Hash of the full projected artifact, not of the bounded display excerpt.
    current_version: str | None = None
    requested_version: str | None = None
    text: str | None = None
    truncated: bool = False
    rank: int | None = None
    role: str | None = None
    assertion_id: str | None = None


class EvidenceBatch(BaseModel):
    references: list[EvidenceLocator] = Field(min_length=1, max_length=5)


class EvidenceIndex(BaseModel):
    items: list[EvidenceItem] = Field(max_length=20)
    omitted: int = 0
    unaddressable_results: int = 0
