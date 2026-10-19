"""Read and write ``OntologyProposal`` nodes.

Proposals are graph-only DataPoints (no vector index beyond their ``index_fields``),
scoped to a dataset through ``dataset_scope`` exactly like ``SkillImprovementProposal``,
and stored through ``add_data_points`` so provenance stamping is the same as for any
other node. Callers run inside the dataset's database context.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.modules.engine.models import OntologyProposal
from cognee.shared.logging_utils import get_logger

logger = get_logger("ontology_proposals")

PROPOSAL_PIPELINE_NAME = "ontology_proposals_pipeline"


def proposal_id_for(kind: str, dataset_id: UUID | str, *parts: str) -> str:
    """Deterministic id so the same finding is proposed once per dataset, ever."""
    key = ":".join([kind, str(dataset_id), *(str(part).strip().lower() for part in parts)])
    return str(uuid5(NAMESPACE_URL, f"cognee:ontology-proposal:{key}"))


def storage_context(user: Any, dataset: Any, key: str):
    """A ``PipelineContext`` for ``add_data_points``; ``None`` when there is no dataset."""
    from cognee.modules.pipelines.models.PipelineContext import PipelineContext

    dataset_id = getattr(dataset, "id", None)
    if user is None or dataset is None or dataset_id is None:
        return None
    return PipelineContext(
        user=user,
        dataset=dataset,
        data_item=SimpleNamespace(id=uuid5(NAMESPACE_URL, f"cognee:ontology-proposal:{key}")),
        pipeline_name=PROPOSAL_PIPELINE_NAME,
    )


async def save_proposals(
    proposals: list[Any],
    *,
    user: Any,
    dataset: Any,
    custom_edges: list | None = None,
) -> None:
    """Write proposal nodes (plus any concept nodes and edges an applied proposal produced)."""
    if not proposals and not custom_edges:
        return
    from cognee.tasks.storage.add_data_points import add_data_points

    key = ",".join(getattr(proposal, "proposal_id", str(proposal.id)) for proposal in proposals)
    # Proposals themselves need no vector index; a concept node an applied mapping
    # creates does (grounding and the hybrid entity lane find classes by name).
    only_proposals = all(isinstance(point, OntologyProposal) for point in proposals)
    await add_data_points(
        list(proposals),
        custom_edges=custom_edges,
        ctx=storage_context(user, dataset, key or "edges"),
        graph_only=only_proposals,
    )


def coerce_proposal(raw: Any) -> OntologyProposal | None:
    """Turn a raw graph node (``(id, props)`` tuple or dict) into a proposal, or ``None``."""
    if isinstance(raw, OntologyProposal):
        return raw
    node_id = None
    if isinstance(raw, (list, tuple)) and len(raw) > 1:
        node_id, raw = raw[0], raw[1]
    data = raw.model_dump() if hasattr(raw, "model_dump") else raw
    if not isinstance(data, dict):
        return None
    if data.get("type") not in (None, OntologyProposal.__name__):
        return None
    data = {key: value for key, value in data.items() if key != "metadata"}
    if node_id is not None and "id" not in data:
        data["id"] = node_id
    try:
        return OntologyProposal.model_validate(data)
    except Exception:
        logger.debug("Skipping node that is not a valid OntologyProposal", exc_info=True)
        return None


async def load_proposals(
    dataset_id: UUID | str,
    *,
    kind: str | None = None,
    status: str | None = None,
) -> list[OntologyProposal]:
    """Every proposal in the dataset's scope, optionally filtered by kind and status."""
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()
    raw_nodes: list[Any] = []
    get_by_type = getattr(graph_engine, "get_nodes_by_type", None)
    if get_by_type is not None:
        try:
            raw_nodes = await get_by_type(node_type=OntologyProposal)
        except Exception as error:
            logger.warning("Proposal lookup by type failed: %s", error, exc_info=True)
            raw_nodes = []
    if not raw_nodes:
        try:
            raw_nodes, _ = await graph_engine.get_filtered_graph_data(
                [{"type": [OntologyProposal.__name__]}]
            )
        except Exception as error:
            logger.warning("Proposal lookup failed: %s", error, exc_info=True)
            return []

    proposals = []
    for raw in raw_nodes:
        proposal = coerce_proposal(raw)
        if proposal is None or str(dataset_id) not in (proposal.dataset_scope or []):
            continue
        if kind is not None and proposal.kind != kind:
            continue
        if status is not None and proposal.status != status:
            continue
        proposals.append(proposal)
    proposals.sort(key=lambda proposal: (-proposal.confidence, proposal.subject_name))
    return proposals


async def find_proposal(proposal_id: str, dataset_id: UUID | str) -> OntologyProposal | None:
    for proposal in await load_proposals(dataset_id):
        if proposal.proposal_id == proposal_id:
            return proposal
    return None
