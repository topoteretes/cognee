from typing import Any, Iterable, Optional
from uuid import UUID

from sqlalchemy import and_, delete, exists, select
from sqlalchemy.orm import aliased, attributes as orm_attributes

from cognee.context_global_variables import multi_user_support_possible
from cognee.infrastructure.databases.relational import get_relational_engine
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


def _deduplicate_by_id(rows: Iterable[Any]) -> list[Any]:
    seen_ids = set()
    deduplicated = []
    for row in rows:
        if row.id in seen_ids:
            continue
        seen_ids.add(row.id)
        deduplicated.append(row)
    return deduplicated


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

        target_node_ids = [node.id for node in target_nodes]
        target_edge_ids = [edge.id for edge in target_edges]

        unique_nodes = []
        if target_nodes:
            node_alias = aliased(Node)
            for node in target_nodes:
                shared_query = (
                    select(node_alias.id)
                    .where(node_alias.slug == node.slug)
                    .where(node_alias.id.not_in(target_node_ids))
                )
                if multi_user_support_possible():
                    shared_query = shared_query.where(node_alias.dataset_id == dataset_id)

                shared_exists = (
                    await session.execute(select(exists(shared_query).label("has_shared")))
                ).scalar()

                if not shared_exists:
                    unique_nodes.append(node)

        unique_edges = []
        if target_edges:
            edge_alias = aliased(Edge)
            for edge in target_edges:
                shared_query = (
                    select(edge_alias.id)
                    .where(edge_alias.slug == edge.slug)
                    .where(edge_alias.id.not_in(target_edge_ids))
                )
                if multi_user_support_possible():
                    shared_query = shared_query.where(edge_alias.dataset_id == dataset_id)

                shared_exists = (
                    await session.execute(select(exists(shared_query).label("has_shared")))
                ).scalar()

                if not shared_exists:
                    unique_edges.append(edge)

        unique_nodes = _deduplicate_by_id(unique_nodes)
        unique_edges = _deduplicate_by_id(unique_edges)

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

        dataset_id_str = str(dataset_id)
        if target_data_ids:
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

        await session.commit()

    logger.info(
        "Cognify rollback completed for run %s (dataset=%s, user=%s, rows=%d nodes/%d edges).",
        pipeline_run_id,
        dataset_id,
        user_id,
        len(target_nodes),
        len(target_edges),
    )
