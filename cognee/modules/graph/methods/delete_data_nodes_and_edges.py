from uuid import UUID

from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine
from cognee.modules.graph.legacy.has_edges_in_legacy_ledger import has_edges_in_legacy_ledger
from cognee.modules.graph.legacy.has_nodes_in_legacy_ledger import has_nodes_in_legacy_ledger
from cognee.modules.graph.methods import (
    delete_data_related_edges,
    delete_data_related_nodes,
    get_data_related_nodes,
    get_data_related_edges,
    get_global_data_related_nodes,
    get_global_data_related_edges,
    get_orphaned_nodeset_labels_for_dataset,
    get_shared_slugs_losing_dataset_anchor,
)
from cognee.modules.graph.methods.delete_from_graph_and_vector import (
    delete_from_graph_and_vector,
)
from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset
from cognee.modules.users.methods.get_user import get_user
from cognee.shared.logging_utils import get_logger


logger = get_logger("delete_data_nodes_and_edges")


async def delete_data_nodes_and_edges(dataset_id: UUID, data_id: UUID, user_id: UUID) -> None:
    user = await get_user(user_id)

    # Check if user has delete permissions for the dataset before proceeding with deletion of related graph/vector nodes and edges.
    dataset = await get_authorized_dataset(user, dataset_id, "delete")
    dataset_id = dataset.id

    if backend_access_control_enabled():
        affected_nodes = await get_data_related_nodes(dataset_id, data_id)
        affected_edges = await get_data_related_edges(dataset_id, data_id) if affected_nodes else []
    else:
        # Pass dataset_id so shared data (same data_id linked to multiple
        # datasets in a single global DB) isn't hard-deleted when only one
        # of its dataset links is removed. Slugs co-owned by another
        # (dataset_id, data_id) pair are excluded from the delete set.
        affected_nodes = await get_global_data_related_nodes(data_id, dataset_id=dataset_id)
        affected_edges = (
            await get_global_data_related_edges(data_id, dataset_id=dataset_id)
            if affected_nodes
            else []
        )

    # Snapshot shared slugs that lose their (dataset_id, *) anchor but will
    # stay alive in the graph/vector stores because another dataset owns
    # them, plus the NodeSet labels this dataset is fully losing. Both must
    # be computed BEFORE relational ledger cleanup since the queries read
    # the surviving-owner rows that `delete_data_related_*` is about to
    # remove. Only relevant in the single-DB deployment — the multi-user
    # path isolates datasets into their own databases so no cross-dataset
    # slug sharing can occur.
    shared_slugs_to_detag: list = []
    orphaned_nodeset_labels: list = []
    if not backend_access_control_enabled():
        shared_slugs_to_detag = await get_shared_slugs_losing_dataset_anchor(dataset_id, data_id)
        if shared_slugs_to_detag:
            orphaned_nodeset_labels = await get_orphaned_nodeset_labels_for_dataset(
                dataset_id, data_id
            )

    if affected_nodes:
        is_legacy_node = await has_nodes_in_legacy_ledger(affected_nodes)
        is_legacy_edge = await has_edges_in_legacy_ledger(affected_edges)

        await delete_from_graph_and_vector(
            affected_nodes, affected_edges, is_legacy_node, is_legacy_edge
        )

    # Always clean up relational ownership records, even when no unique
    # graph/vector nodes needed deletion (e.g. shared nodes across data items).
    await delete_data_related_nodes(data_id, dataset_id)
    await delete_data_related_edges(data_id, dataset_id)

    # Reconcile `belongs_to_set` on surviving shared nodes: strip every
    # NodeSet label this dataset is losing from the surviving shared slugs.
    # The NodeSet(s) themselves may still exist (cross-dataset anchors), so
    # the unscoped detag in `delete_from_graph_and_vector` won't touch them
    # — we have to scope by node_ids and pass the orphaned NodeSet labels
    # explicitly. `belongs_to_set` stores NodeSet names (not dataset names),
    # so the tag list must be sourced from `Node.label` on NodeSet ledger
    # rows that are losing their (dataset_id, *) anchor.
    if shared_slugs_to_detag and orphaned_nodeset_labels:
        slug_ids = [str(slug) for slug in shared_slugs_to_detag]
        try:
            graph_engine = await get_graph_engine()
            await graph_engine.remove_belongs_to_set_tags(
                orphaned_nodeset_labels, node_ids=slug_ids
            )
        except Exception as e:
            logger.warning(
                "Shared-slug graph detag failed for dataset %s (non-fatal): %s",
                dataset_id,
                e,
            )
        try:
            vector_engine = get_vector_engine()
            await vector_engine.remove_belongs_to_set_tags(
                orphaned_nodeset_labels, node_ids=slug_ids
            )
        except Exception as e:
            logger.warning(
                "Shared-slug vector detag failed for dataset %s (non-fatal): %s",
                dataset_id,
                e,
            )
