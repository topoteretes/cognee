"""Ratify or reject an ``OntologyProposal`` — the "human ratifies" half of the loop.

Applying writes the governed change with ``ontology_valid=True`` and a ``ratified_by``
stamp, then marks the proposal ``applied``; rejecting only marks it, and because
proposal ids are deterministic the same finding is never drafted again. Callers run
inside the dataset's database context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from cognee.modules.engine.models import EntityType, OntologyProperty, OntologyProposal
from cognee.modules.engine.models.OntologyProposal import (
    PROPOSAL_KIND_DEFINITION_CONFLICT,
    PROPOSAL_KIND_MAPPING,
    PROPOSAL_KIND_ONTOLOGY_EXTENSION,
    PROPOSAL_STATUS_APPLIED,
    PROPOSAL_STATUS_PROPOSED,
    PROPOSAL_STATUS_REJECTED,
)
from cognee.modules.engine.utils import generate_node_name
from cognee.modules.ontology.schema_alignment import REALIZES_RELATIONSHIP
from cognee.shared.logging_utils import get_logger

from .generate import CONTRADICTS_RELATIONSHIP
from .store import save_proposals

logger = get_logger("ontology_proposals")


class ProposalNotApplicableError(ValueError):
    """The proposal is not in ``proposed`` status or its kind is unknown."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp(proposal: OntologyProposal, ratified_by: str, **extra: Any) -> dict[str, Any]:
    return {
        "ontology_valid": True,
        "ratified_by": ratified_by,
        "ratified_at": _now(),
        "proposal_id": proposal.proposal_id,
        **extra,
    }


def _concept_node(proposal: OntologyProposal) -> EntityType | OntologyProperty:
    normalized = generate_node_name(proposal.concept_name)
    if proposal.concept_category == "properties":
        return OntologyProperty(
            id=OntologyProperty.id_for(proposal.concept_name),
            name=normalized,
            description=normalized,
            ontology_valid=True,
            ontology_uri=proposal.concept_uri,
        )
    return EntityType(
        id=EntityType.id_for(proposal.concept_name),
        name=normalized,
        description=normalized,
        ontology_valid=True,
        ontology_uri=proposal.concept_uri,
    )


def _as_uuid(node_id: str) -> UUID | str:
    """Graph writers and the rollback ledger both take UUID endpoints; keep a raw string
    only for ids that are not UUIDs (never the case for cognee-written nodes)."""
    try:
        return UUID(str(node_id))
    except (ValueError, AttributeError, TypeError):
        return node_id


def _edge(source_id: str, target_id: str, relationship_name: str, props: dict[str, Any]):
    return (
        _as_uuid(source_id),
        _as_uuid(target_id),
        relationship_name,
        {
            "source_node_id": str(source_id),
            "target_node_id": str(target_id),
            "relationship_name": relationship_name,
            **props,
        },
    )


async def _existing_edge_props(graph_engine: Any, proposal: OntologyProposal) -> dict[str, Any]:
    """Current properties of the ``contradicts`` edge, so applying merges instead of clobbering."""
    try:
        _, edges = await graph_engine.get_neighborhood(
            [proposal.subject_id], depth=1, edge_types=[CONTRADICTS_RELATIONSHIP]
        )
    except Exception as error:
        logger.debug("Could not read the contradicts edge: %s", error, exc_info=True)
        return {}
    for edge in edges or []:
        if (
            isinstance(edge, (list, tuple))
            and len(edge) >= 3
            and str(edge[0]) == proposal.subject_id
            and str(edge[1]) == proposal.counterpart_id
            and str(edge[2]) == CONTRADICTS_RELATIONSHIP
        ):
            props = edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}
            return dict(props)
    return {}


async def apply_proposal(
    proposal: OntologyProposal,
    *,
    ratified_by: str,
    user: Any,
    dataset: Any,
    resolution: str = "",
    graph_engine: Any = None,
) -> OntologyProposal:
    """Write the change the proposal describes and mark it applied."""
    if proposal.status != PROPOSAL_STATUS_PROPOSED:
        raise ProposalNotApplicableError(
            f"Proposal {proposal.proposal_id} is already {proposal.status}."
        )
    if graph_engine is None:
        from cognee.infrastructure.databases.graph import get_graph_engine

        graph_engine = await get_graph_engine()

    nodes: list[Any] = []
    edges: list[Any] = []
    if proposal.kind == PROPOSAL_KIND_MAPPING:
        concept = _concept_node(proposal)
        nodes.append(concept)
        edges.append(
            _edge(
                proposal.subject_id,
                str(concept.id),
                REALIZES_RELATIONSHIP,
                _stamp(proposal, ratified_by, authoritative=False, owner=None),
            )
        )
    elif proposal.kind == PROPOSAL_KIND_DEFINITION_CONFLICT:
        props = await _existing_edge_props(graph_engine, proposal)
        props.update(_stamp(proposal, ratified_by, resolution=resolution or "reviewed"))
        edges.append(
            _edge(proposal.subject_id, proposal.counterpart_id, CONTRADICTS_RELATIONSHIP, props)
        )
    elif proposal.kind == PROPOSAL_KIND_ONTOLOGY_EXTENSION:
        updated = await graph_engine.update_node(proposal.subject_id, _stamp(proposal, ratified_by))
        if not updated:
            raise ProposalNotApplicableError(
                f"EntityType {proposal.subject_id} no longer exists; nothing to extend."
            )
    else:
        raise ProposalNotApplicableError(f"Unknown proposal kind {proposal.kind!r}.")

    proposal.status = PROPOSAL_STATUS_APPLIED
    proposal.ratified_by = ratified_by
    proposal.ratified_at = _now()
    proposal.resolution = resolution or proposal.resolution
    await save_proposals([*nodes, proposal], user=user, dataset=dataset, custom_edges=edges or None)
    logger.info(
        "Applied %s proposal %s (%s) ratified by %s",
        proposal.kind,
        proposal.proposal_id,
        proposal.subject_name,
        ratified_by,
    )
    return proposal


async def reject_proposal(
    proposal: OntologyProposal,
    *,
    ratified_by: str,
    user: Any,
    dataset: Any,
    resolution: str = "",
) -> OntologyProposal:
    """Mark the proposal rejected; nothing else in the graph changes."""
    if proposal.status != PROPOSAL_STATUS_PROPOSED:
        raise ProposalNotApplicableError(
            f"Proposal {proposal.proposal_id} is already {proposal.status}."
        )
    proposal.status = PROPOSAL_STATUS_REJECTED
    proposal.ratified_by = ratified_by
    proposal.ratified_at = _now()
    proposal.resolution = resolution
    await save_proposals([proposal], user=user, dataset=dataset)
    return proposal
