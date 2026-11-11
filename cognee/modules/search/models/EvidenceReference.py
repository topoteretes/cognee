from typing import Literal

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

    dataset_id: str | None = None
    source_ref_key: str | None = None

    data_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    document_name: str | None = None
    # The cited document's title (``Data.label``) and link (``system_metadata["url"]``),
    # when it has them; ``document_name`` is the stored file's name.
    title: str | None = None
    url: str | None = None

    source_node_id: str | None = None
    target_node_id: str | None = None
    relationship_name: str | None = None
    assertion_id: str | None = None
    label: str | None = None

    rank: int | None = None
    score: float | None = None
