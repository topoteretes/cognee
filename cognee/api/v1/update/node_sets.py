"""Metadata-only update: move one document to other node sets, without reprocessing.

``update(data_id, None, dataset_id, node_set=[...])`` lands here once the
dataset write permission is checked and the id resolved. The replacement is
reconciled in graph and vector storage from the document's in-graph ownership
(``reassign_document_node_sets``), then published to the relational row. No
LLM call, no loading or chunking, no embedding, and never the full rebuild: a
precondition that does not hold raises before anything is written.
"""

from time import perf_counter
from uuid import UUID

from cognee.api.v1.update.result import UpdateResult
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.locks import dataset_lock
from cognee.modules.cognify.routing import CognifyRoute, cognify_route_for
from cognee.modules.data.methods import get_authorized_dataset, get_data, publish_node_set
from cognee.modules.data.methods.publish_updated_data import has_node_set
from cognee.modules.engine.models.node_set import (
    NodeSet,
    parse_node_set_names,
    validate_node_set_names,
)
from cognee.modules.graph.exceptions import NodeSetReassignmentNotSupportedError
from cognee.modules.graph.methods.reassign_document_node_sets import (
    apply_node_set_reassignment,
    plan_node_set_reassignment,
)
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("update.node_sets")


def normalize_node_set(node_set: list[str]) -> list[str]:
    """The validated replacement, one entry per NodeSet (first spelling wins).

    Names that normalize to one NodeSet id are one set, so repeating one never
    duplicates a membership.
    """
    validate_node_set_names(node_set)
    unique: dict = {}
    for name in node_set:
        unique.setdefault(NodeSet.id_for(name), name)
    return list(unique.values())


async def _require_capabilities(graph, vector) -> bool:
    """Refuse storage that cannot patch membership in place; True when there is any.

    The boundary is the chunk-level update's: graph provenance plus partial
    node patches, and payload-only vector updates. An empty graph has nothing
    to reconcile; a non-empty one without the provenance marker is a legacy
    graph whose ownership cannot be read.
    """
    if not getattr(graph, "supports_incremental_chunk_updates", False):
        raise NodeSetReassignmentNotSupportedError(
            f"graph backend {type(graph).__name__} does not support in-place node-set updates"
        )
    if not getattr(vector, "supports_payload_update", False):
        raise NodeSetReassignmentNotSupportedError(
            f"vector backend {type(vector).__name__} does not support payload-only updates"
        )
    if await stores_provenance_in_graph(graph):
        return True
    if await graph.is_empty():
        return False
    raise NodeSetReassignmentNotSupportedError(
        "the graph does not store ownership provenance in-graph (a legacy graph), so the "
        "document's memberships cannot be identified"
    )


async def update_node_sets(
    data_id: UUID,
    dataset_id: UUID,
    user: User,
    node_set: list[str],
) -> dict:
    """Replace the complete node-set list of one document; ``[]`` clears it.

    ``data_id`` is the resolved id of a document in the dataset. Dataset write
    permission is enough: nothing is removed from memory but memberships. The
    stores are the dataset owner's, as for every other dataset write.
    Idempotent: a repeat finds storage consistent and answers ``unchanged``. A
    call that failed part-way is completed by repeating it.
    """
    started = perf_counter()
    node_set = normalize_node_set(node_set)
    dataset = await get_authorized_dataset(user, dataset_id, "write")
    if dataset is None:
        raise PermissionDeniedError(message="Not authorized to update this dataset")

    async with (
        dataset_lock(dataset.id),
        set_database_global_context_variables(dataset.id, dataset.owner_id),
    ):
        data = await get_data(user.id, data_id, dataset.id, verify_owner=False)
        if data is None:
            from cognee.api.v1.exceptions import UpdateTargetNotFoundError

            raise UpdateTargetNotFoundError(data_id=data_id, dataset_id=dataset_id)
        route = cognify_route_for(data)
        if route is not CognifyRoute.STANDARD:
            raise NodeSetReassignmentNotSupportedError(
                f"document {data.id} is on the {route.value} cognify route, whose graph has no "
                "document chunks to reassign"
            )

        graph = await get_graph_engine()
        vector = await get_vector_engine_async()
        stored_node_set = parse_node_set_names(data.node_set)
        plan = None
        if await _require_capabilities(graph, vector):
            plan = await plan_node_set_reassignment(
                graph, vector, dataset.id, data.id, node_set, stored_node_set
            )

        changed = not has_node_set(data, node_set) or (plan is not None and not plan.is_empty)
        if changed:
            # Graph and vector first: the relational row is what a retry and a
            # later cognify read the assignment from, so it flips last.
            if plan is not None:
                await apply_node_set_reassignment(graph, vector, plan)
            await publish_node_set(data.id, node_set)
            logger.info(
                "node sets of document %s replaced: %s -> %s", data.id, stored_node_set, node_set
            )

    return UpdateResult(
        status="metadata_updated" if changed else "unchanged",
        data_id=data.id,
        dataset_id=dataset.id,
        duration_seconds=round(perf_counter() - started, 3),
    ).model_dump()
