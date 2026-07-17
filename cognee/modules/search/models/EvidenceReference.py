from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


EvidenceKind = Literal["segment", "graph_node", "graph_edge", "triplet"]
EvidenceRole = Literal["used_as_context", "supports_assertion"]


class EvidenceReference(BaseModel):
    """Stable identifier for an artifact that was included in completion context.

    Evidence references intentionally contain identifiers and small locators only.
    Source text remains in the vector or graph store rather than being duplicated in
    every search response.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    kind: EvidenceKind
    artifact_id: str
    role: EvidenceRole = "used_as_context"

    dataset_id: Optional[str] = None
    source_ref_key: Optional[str] = None

    data_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    document_name: Optional[str] = None

    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    relationship_name: Optional[str] = None
    assertion_id: Optional[str] = None
    label: Optional[str] = None

    rank: Optional[int] = None
    score: Optional[float] = None
