from uuid import UUID

from cognee.context_global_variables import is_multi_user_support_possible
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
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
    get_shared_slugs_losing_dataset_anchor,
)
from cognee.modules.graph.methods.delete_from_graph_and_vector import (
    delete_from_graph_and_vector,
)
from cognee.shared.logging_utils import get_logger


logger = get_logger("delete_data_nodes_and_edges")


async def delete_data_nodes_and_edges(dataset_id: UUID, data_id: UUID, user_id: UUID) -> None:
    if is_multi_user_support_possible():
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
    # them. These must be computed BEFORE relational ledger cleanup since
    # the query reads the surviving-owner rows that `delete_data_related_*`
    # is about to remove. Only relevant in the single-DB deployment — the
    # multi-user path isolates datasets into their own databases so no
    # cross-dataset slug sharing can occur.
    shared_slugs_to_detag: list = []
    dataset_name = None
    if not is_multi_user_support_possible():
        shared_slugs_to_detag = await get_shared_slugs_losing_dataset_anchor(
            dataset_id, data_id
        )
        if shared_slugs_to_detag:
            dataset_name = await _get_dataset_name(dataset_id)

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

    # Reconcile `belongs_to_set` on surviving shared nodes: strip the
    # removed dataset's label from nodes that just lost their last
    # (dataset_id, *) anchor. The dataset's NodeSet may still exist for
    # other data items in the dataset, so the unscoped detag path isn't
    # safe — pass the specific node ids.
    if shared_slugs_to_detag and dataset_name:
        slug_ids = [str(slug) for slug in shared_slugs_to_detag]
        try:
            graph_engine = await get_graph_engine()
            await graph_engine.remove_belongs_to_set_tags([dataset_name], node_ids=slug_ids)
        except Exception as e:
            logger.warning(
                "Shared-slug graph detag failed for dataset %s (non-fatal): %s",
                dataset_id,
                e,
            )
        try:
            vector_engine = get_vector_engine()
            await vector_engine.remove_belongs_to_set_tags([dataset_name], node_ids=slug_ids)
        except Exception as e:
            logger.warning(
                "Shared-slug vector detag failed for dataset %s (non-fatal): %s",
                dataset_id,
                e,
            )


async def _get_dataset_name(dataset_id: UUID) -> str:
    """Return the dataset's `name`, used as the `belongs_to_set` label.

    Looked up directly from the relational `datasets` table instead of
    through the permission-aware `get_dataset(user_id, dataset_id)` helper
    — the caller has already passed the authorization check, and we need
    the name regardless of ownership so the stale-tag cleanup runs.
    """
    from cognee.modules.data.models import Dataset

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        dataset = await session.get(Dataset, dataset_id)
        return dataset.name if dataset else None
