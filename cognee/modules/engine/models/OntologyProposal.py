from __future__ import annotations

from pydantic import Field

from cognee.infrastructure.engine import DataPoint

PROPOSAL_KIND_MAPPING = "mapping"
PROPOSAL_KIND_DEFINITION_CONFLICT = "definition_conflict"
PROPOSAL_KIND_ONTOLOGY_EXTENSION = "ontology_extension"
PROPOSAL_KINDS = (
    PROPOSAL_KIND_MAPPING,
    PROPOSAL_KIND_DEFINITION_CONFLICT,
    PROPOSAL_KIND_ONTOLOGY_EXTENSION,
)

PROPOSAL_STATUS_PROPOSED = "proposed"
PROPOSAL_STATUS_APPLIED = "applied"
PROPOSAL_STATUS_REJECTED = "rejected"


class OntologyProposal(DataPoint):
    """A reviewable, graph-only proposal to change the governed model.

    The "LLM (or heuristic) proposes, a human ratifies" loop in one shape. ``kind``
    says what applying it does:

    * ``mapping`` — link ``subject`` (a schema table or column node) to the ontology
      concept ``concept_name`` with a ``realizes`` edge.
    * ``definition_conflict`` — two nodes (``subject`` and ``counterpart``) define the
      same term differently; applying records the ``resolution`` on the ``contradicts``
      edge between them and marks it reviewed.
    * ``ontology_extension`` — an extracted ``EntityType`` that appears often with no
      ontology match; applying marks the node ``ontology_valid`` so it counts as part
      of the model until the ontology file catches up.

    Nothing in the graph changes until ``apply`` runs; ``ratified_by`` records who did.
    """

    proposal_id: str
    kind: str
    dataset_scope: list[str] = Field(default_factory=list)

    subject_id: str = ""
    subject_name: str = ""
    subject_kind: str = ""  # SchemaTable | SchemaColumn | TableType | EntityType | Entity
    concept_name: str = ""
    concept_uri: str | None = None
    concept_category: str = ""  # classes | properties
    counterpart_id: str = ""
    counterpart_name: str = ""

    evidence: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0
    occurrences: int = 0
    model_name: str = ""
    proposed_by: str = "heuristic"  # "llm" | "heuristic"

    status: str = PROPOSAL_STATUS_PROPOSED
    ratified_by: str | None = None
    ratified_at: str | None = None
    resolution: str = ""

    metadata: dict = Field(
        default={
            "index_fields": ["subject_name", "rationale"],
            "identity_fields": ["proposal_id"],
        }
    )
