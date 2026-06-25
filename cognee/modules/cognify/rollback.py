from typing import Any, Optional
from uuid import UUID

from sqlalchemy import and_, delete, distinct, select
from sqlalchemy.orm import aliased, attributes as orm_attributes

from cognee.context_global_variables import multi_user_support_possible
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.provenance import get_data_id_from_source_ref_key
from cognee.infrastructure.databases.provenance.markers import is_graph_native_graph
from cognee.modules.data.models import Data
from cognee.modules.graph.legacy.has_edges_in_legacy_ledger import has_edges_in_legacy_ledger
from cognee.modules.graph.legacy.has_nodes_in_legacy_ledger import has_nodes_in_legacy_ledger
from cognee.modules.graph.methods.delete_from_graph_and_vector import delete_from_graph_and_vector
from cognee.modules.graph.models import Edge, Node
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify.rollback")


def _to_uuid(value: Any) -> Optional[UUID]:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _extract_data_ids(data_ingestion_info: Any) -> set[UUID]:
    if not isinstance(data_ingestion_info, list):
        return set()

    data_ids: set[UUID] = set()
    for entry in data_ingestion_info:
        if not isinstance(entry, dict):
            continue
        maybe_data_id = _to_uuid(entry.get("data_id"))
        if maybe_data_id:
            data_ids.add(maybe_data_id)
    return data_ids


async def _graph_native_affected_data_ids(graph_engine, pipeline_run_id: str) -> set[UUID]:
    """Data ids whose ownership the run introduced, read from graph provenance.

    Must be called *before* the rollback removes the run's source refs. The run's
    source refs (per Part 0, the refs it newly attached) carry the dataset/data
    pair, so the data ids fall straight out of the source-ref helper — this is the
    set whose per-data cognify status the rollback must clear, and it works even
    when ``data_ingestion_info`` is absent (e.g. startup recovery).
    """
    refs_by_node = await graph_engine.find_node_source_refs_by_pipeline_run(pipeline_run_id)
    refs_by_edge = await graph_engine.find_edge_source_refs_by_pipeline_run(pipeline_run_id)

    data_ids: set[UUID] = set()
    for refs in list(refs_by_node.values()) + list(refs_by_edge.values()):
        for source_ref_key in refs:
            data_ids.add(get_data_id_from_source_ref_key(source_ref_key))
    return data_ids


async def _reset_pipeline_status(session, target_data_ids: set, dataset_id: Any) -> None:
    """Clear the cognify_pipeline status for the rolled-back run's data ids."""
    if not target_data_ids:
        return

    dataset_id_str = str(dataset_id)
    data_records = (
        (await session.execute(select(Data).where(Data.id.in_(list(target_data_ids)))))
        .scalars()
        .all()
    )

    for data_record in data_records:
        if not data_record.pipeline_status:
            continue
        if (
            "cognify_pipeline" in data_record.pipeline_status
            and dataset_id_str in data_record.pipeline_status["cognify_pipeline"]
        ):
            del data_record.pipeline_status["cognify_pipeline"][dataset_id_str]
            orm_attributes.flag_modified(data_record, "pipeline_status")


async def cognify_rollback_handler(
    pipeline_run_id: UUID,
    dataset: Any,
    user: Any = None,
    data_ingestion_info: Any = None,
    **kwargs: Any,
) -> None:
    dataset_id = getattr(dataset, "id", None)
    if not dataset_id or not pipeline_run_id:
        logger.warning(
            "Rollback skipped due to missing dataset_id or pipeline_run_id "
            "(dataset_id=%s, pipeline_run_id=%s).",
            dataset_id,
            pipeline_run_id,
        )
        return

    user_id = getattr(user, "id", None)
    db_engine = get_relational_engine()

    # Graph-native graphs carry provenance in the graph (no relational ledger
    # rows). Roll back through the unified boundary, which removes the refs the
    # run attached and hard-deletes any artifact left unowned. We still reset
    # the cognify pipeline status for the run's data ids so re-cognify works.
    unified = await get_unified_engine()
    if unified.supports_graph_native_delete():
        graph_engine = unified.graph
        if await is_graph_native_graph(graph_engine):
            # Read the run's affected data ids from graph provenance BEFORE the
            # rollback removes the run's source refs. Supplement with any
            # data_ingestion_info the caller passed (startup recovery passes
            # none, so the graph read is what keeps status reset correct there).
            target_data_ids = await _graph_native_affected_data_ids(
                graph_engine, str(pipeline_run_id)
            )
            target_data_ids |= _extract_data_ids(data_ingestion_info)

            await unified.rollback_by_pipeline_run_id(str(pipeline_run_id))

            async with db_engine.get_async_session() as session:
                await _reset_pipeline_status(session, target_data_ids, dataset_id)
                await session.commit()

            logger.info(
                "Graph-native cognify rollback completed for run %s (dataset=%s, user=%s).",
                pipeline_run_id,
                dataset_id,
                user_id,
            )
            return

    async with db_engine.get_async_session() as session:
        target_nodes = (
            (
                await session.execute(
                    select(Node).where(
                        and_(
                            Node.pipeline_run_id == pipeline_run_id,
                            Node.dataset_id == dataset_id,
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        target_edges = (
            (
                await session.execute(
                    select(Edge).where(
                        and_(
                            Edge.pipeline_run_id == pipeline_run_id,
                            Edge.dataset_id == dataset_id,
                        )
                    )
                )
            )
            .scalars()
            .all()
        )

        target_data_ids = (
            {node.data_id for node in target_nodes}
            | {edge.data_id for edge in target_edges}
            | _extract_data_ids(data_ingestion_info)
        )

        unique_nodes = []
        if target_nodes:
            target_node_ids = [node.id for node in target_nodes]
            target_node_slugs = list({node.slug for node in target_nodes})
            node_alias = aliased(Node)
            shared_node_slugs_query = (
                select(distinct(node_alias.slug))
                .where(node_alias.slug.in_(target_node_slugs))
                .where(node_alias.id.not_in(target_node_ids))
            )
            if multi_user_support_possible():
                shared_node_slugs_query = shared_node_slugs_query.where(
                    node_alias.dataset_id == dataset_id
                )

            shared_node_slugs = set(
                (await session.execute(shared_node_slugs_query)).scalars().all()
            )
            unique_nodes = [node for node in target_nodes if node.slug not in shared_node_slugs]

        unique_edges = []
        if target_edges:
            target_edge_ids = [edge.id for edge in target_edges]
            target_edge_slugs = list({edge.slug for edge in target_edges})
            edge_alias = aliased(Edge)
            shared_edge_slugs_query = (
                select(distinct(edge_alias.slug))
                .where(edge_alias.slug.in_(target_edge_slugs))
                .where(edge_alias.id.not_in(target_edge_ids))
            )
            if multi_user_support_possible():
                shared_edge_slugs_query = shared_edge_slugs_query.where(
                    edge_alias.dataset_id == dataset_id
                )

            shared_edge_slugs = set(
                (await session.execute(shared_edge_slugs_query)).scalars().all()
            )
            unique_edges = [edge for edge in target_edges if edge.slug not in shared_edge_slugs]

    # Important ordering for robust retries:
    # 1) Delete graph/vector artifacts first
    # 2) Delete relational ownership rows and reset pipeline_status second
    # If graph/vector deletion fails, relational rows remain as rollback metadata.

    if unique_nodes:
        is_legacy_node = await has_nodes_in_legacy_ledger(unique_nodes)
    else:
        is_legacy_node = []

    if unique_edges:
        is_legacy_edge = await has_edges_in_legacy_ledger(unique_edges)
    else:
        is_legacy_edge = []

    if unique_nodes or unique_edges:
        await delete_from_graph_and_vector(
            unique_nodes, unique_edges, is_legacy_node, is_legacy_edge
        )

    async with db_engine.get_async_session() as session:
        if target_nodes:
            await session.execute(
                delete(Node).where(
                    and_(
                        Node.pipeline_run_id == pipeline_run_id,
                        Node.dataset_id == dataset_id,
                    )
                )
            )
        if target_edges:
            await session.execute(
                delete(Edge).where(
                    and_(
                        Edge.pipeline_run_id == pipeline_run_id,
                        Edge.dataset_id == dataset_id,
                    )
                )
            )

        await _reset_pipeline_status(session, target_data_ids, dataset_id)

        await session.commit()

    logger.info(
        "Cognify rollback completed for run %s (dataset=%s, user=%s, rows=%d nodes/%d edges).",
        pipeline_run_id,
        dataset_id,
        user_id,
        len(target_nodes),
        len(target_edges),
    )
