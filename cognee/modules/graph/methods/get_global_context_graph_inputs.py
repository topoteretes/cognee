from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.relational import with_async_session
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.graph.models import Edge, Node

# Graph node types and edge relationship names traversed to build the graph
# bucketing input. On graph-provenance graphs both edges live in relationship_name
# (the relational ledger splits "contains" into label, but the graph does not).
_SUMMARY_TYPE = "TextSummary"
_CHUNK_TYPE = "DocumentChunk"
_ENTITY_TYPE = "Entity"
_MADE_FROM = "made_from"
_CONTAINS = "contains"


@dataclass
class SummaryEntityLoadResult:
    entities_by_summary_id: dict[str, set[str]]
    summarized_chunk_count: int
    summary_ids_with_made_from: set[str]
    missing_made_from_summary_ids: set[str]
    entity_link_count: int


@dataclass
class DatasetEntityCounts:
    chunk_count: int
    entity_chunk_counts: dict[str, int]


@dataclass
class DatasetGraphEntityInput:
    summary_entities: SummaryEntityLoadResult
    entity_counts: DatasetEntityCounts


def coerce_graph_uuid(value: str | UUID, field_name: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{field_name} must be a UUID for graph bucketing: {value!r}.") from error


def coerce_graph_uuid_set(values: Iterable[str | UUID], field_name: str) -> set[UUID]:
    return {coerce_graph_uuid(value, field_name) for value in values}


async def _resolve_graph_provenance_engine():
    """Return the graph engine if this graph stores provenance in the graph
    itself (so its relational Node/Edge ledger is empty), else None."""
    unified = await get_unified_engine()
    if not unified.supports_graph_provenance_delete():
        return None
    graph_engine = unified.graph
    if await stores_provenance_in_graph(graph_engine):
        return graph_engine
    return None


async def _graph_provenance_dataset_subgraph(
    graph_engine,
    dataset_uuid: UUID,
) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
    """Load this dataset's nodes + edges from the graph.

    Scopes by source-ref provenance (works whether or not the graph is isolated
    per dataset), then keeps edges whose endpoints both belong to the dataset.
    """
    node_refs = await graph_engine.find_node_source_refs_by_dataset(str(dataset_uuid))
    dataset_node_ids = set(node_refs)
    all_nodes, all_edges = await graph_engine.get_graph_data()
    nodes_by_id = {
        str(node_id): props for node_id, props in all_nodes if str(node_id) in dataset_node_ids
    }
    edges = [
        (str(source_id), str(target_id), relationship_name)
        for source_id, target_id, relationship_name, _props in all_edges
        if str(source_id) in dataset_node_ids and str(target_id) in dataset_node_ids
    ]
    return nodes_by_id, edges


def _graph_provenance_entity_input(
    nodes_by_id: dict[str, dict],
    edges: list[tuple[str, str, str]],
    expected_summary_uuids: set[UUID],
) -> DatasetGraphEntityInput:
    """Rebuild summary→chunk→entity rows from graph edges, then reuse the same
    result builders as the relational path."""
    type_of = {node_id: props.get("type") for node_id, props in nodes_by_id.items()}
    expected_str = {str(summary_id) for summary_id in expected_summary_uuids}

    summary_chunk_pairs = [
        (source_id, target_id)
        for source_id, target_id, relationship_name in edges
        if relationship_name == _MADE_FROM
        and source_id in expected_str
        and type_of.get(source_id) == _SUMMARY_TYPE
        and type_of.get(target_id) == _CHUNK_TYPE
    ]
    chunk_ids = {target_id for _, target_id in summary_chunk_pairs}
    chunk_entity_pairs = [
        (source_id, target_id)
        for source_id, target_id, relationship_name in edges
        if relationship_name == _CONTAINS
        and source_id in chunk_ids
        and type_of.get(target_id) == _ENTITY_TYPE
    ]

    summary_chunk_rows = [
        (
            coerce_graph_uuid(source_id, "summary node id"),
            coerce_graph_uuid(target_id, "chunk node id"),
        )
        for source_id, target_id in summary_chunk_pairs
    ]
    chunk_entity_rows = [
        (
            coerce_graph_uuid(source_id, "chunk node id"),
            coerce_graph_uuid(target_id, "entity node id"),
        )
        for source_id, target_id in chunk_entity_pairs
    ]

    return DatasetGraphEntityInput(
        summary_entities=_build_summary_entity_load_result(
            expected_summary_uuids,
            summary_chunk_rows,
            chunk_entity_rows,
        ),
        entity_counts=_build_dataset_entity_counts(summary_chunk_rows, chunk_entity_rows),
    )


async def get_dataset_text_summary_ids(dataset_id: str | UUID) -> set[str]:
    graph_engine = await _resolve_graph_provenance_engine()
    if graph_engine is not None:
        dataset_uuid = coerce_graph_uuid(dataset_id, "dataset_id")
        nodes_by_id, _edges = await _graph_provenance_dataset_subgraph(graph_engine, dataset_uuid)
        return {
            node_id for node_id, props in nodes_by_id.items() if props.get("type") == _SUMMARY_TYPE
        }
    return await _relational_dataset_text_summary_ids(dataset_id)


@with_async_session
async def _relational_dataset_text_summary_ids(
    dataset_id: str | UUID,
    session: AsyncSession,
) -> set[str]:
    dataset_uuid = coerce_graph_uuid(dataset_id, "dataset_id")
    result = await session.execute(
        select(Node.slug).where(
            and_(
                Node.dataset_id == dataset_uuid,
                Node.type == "TextSummary",
            )
        )
    )
    return {str(row[0]) for row in result.all()}


@with_async_session
async def load_summary_entities_for_dataset(
    dataset_id: str | UUID,
    expected_summary_ids: Iterable[str | UUID],
    session: AsyncSession,
) -> SummaryEntityLoadResult:
    graph_input = await _load_dataset_graph_entity_input(
        dataset_id,
        expected_summary_ids,
        session,
    )
    return graph_input.summary_entities


@with_async_session
async def get_dataset_chunk_entity_counts(
    dataset_id: str | UUID,
    expected_summary_ids: Iterable[str | UUID],
    session: AsyncSession,
) -> DatasetEntityCounts:
    graph_input = await _load_dataset_graph_entity_input(
        dataset_id,
        expected_summary_ids,
        session,
    )
    return graph_input.entity_counts


@with_async_session
async def load_dataset_graph_entity_input(
    dataset_id: str | UUID,
    expected_summary_ids: Iterable[str | UUID],
    session: AsyncSession,
) -> DatasetGraphEntityInput:
    return await _load_dataset_graph_entity_input(dataset_id, expected_summary_ids, session)


async def _load_dataset_graph_entity_input(
    dataset_id: str | UUID,
    expected_summary_ids: Iterable[str | UUID],
    session: AsyncSession,
) -> DatasetGraphEntityInput:
    dataset_uuid = coerce_graph_uuid(dataset_id, "dataset_id")
    expected_summary_uuids = coerce_graph_uuid_set(expected_summary_ids, "expected_summary_ids")
    if not expected_summary_uuids:
        return DatasetGraphEntityInput(
            summary_entities=_build_summary_entity_load_result(set(), [], []),
            entity_counts=DatasetEntityCounts(chunk_count=0, entity_chunk_counts={}),
        )

    graph_engine = await _resolve_graph_provenance_engine()
    if graph_engine is not None:
        nodes_by_id, edges = await _graph_provenance_dataset_subgraph(graph_engine, dataset_uuid)
        return _graph_provenance_entity_input(nodes_by_id, edges, expected_summary_uuids)

    summary_chunk_rows = await _load_summary_chunk_rows(
        dataset_uuid, expected_summary_uuids, session
    )
    chunk_ids = {chunk_id for _, chunk_id in summary_chunk_rows}
    chunk_entity_rows = await _load_chunk_entity_rows(dataset_uuid, chunk_ids, session)

    return DatasetGraphEntityInput(
        summary_entities=_build_summary_entity_load_result(
            expected_summary_uuids,
            summary_chunk_rows,
            chunk_entity_rows,
        ),
        entity_counts=_build_dataset_entity_counts(summary_chunk_rows, chunk_entity_rows),
    )


def _build_dataset_entity_counts(
    summary_chunk_rows: list[tuple[UUID, UUID]],
    chunk_entity_rows: list[tuple[UUID, UUID]],
) -> DatasetEntityCounts:
    chunk_ids = {chunk_id for _, chunk_id in summary_chunk_rows}
    entity_chunk_ids: dict[UUID, set[UUID]] = {}
    for chunk_id, entity_id in chunk_entity_rows:
        entity_chunk_ids.setdefault(entity_id, set()).add(chunk_id)

    return DatasetEntityCounts(
        chunk_count=len(chunk_ids),
        entity_chunk_counts={
            str(entity_id): len(entity_chunk_ids_for_entity)
            for entity_id, entity_chunk_ids_for_entity in entity_chunk_ids.items()
        },
    )


def _build_summary_entity_load_result(
    expected_summary_ids: set[UUID],
    summary_chunk_rows: list[tuple[UUID, UUID]],
    chunk_entity_rows: list[tuple[UUID, UUID]],
) -> SummaryEntityLoadResult:
    entities_by_summary_id = {str(summary_id): set() for summary_id in expected_summary_ids}
    summary_chunk_ids = _group_summary_chunk_ids(summary_chunk_rows)
    chunk_entity_ids = _group_chunk_entity_ids(chunk_entity_rows)

    for summary_id, summary_chunk_ids_for_summary in summary_chunk_ids.items():
        summary_entities = entities_by_summary_id[str(summary_id)]
        for chunk_id in summary_chunk_ids_for_summary:
            summary_entities.update(
                str(entity_id) for entity_id in chunk_entity_ids.get(chunk_id, set())
            )

    return SummaryEntityLoadResult(
        entities_by_summary_id=entities_by_summary_id,
        summarized_chunk_count=len(_flatten_chunk_ids(summary_chunk_ids)),
        summary_ids_with_made_from={str(summary_id) for summary_id in summary_chunk_ids},
        missing_made_from_summary_ids={
            str(summary_id) for summary_id in expected_summary_ids - set(summary_chunk_ids)
        },
        entity_link_count=len(chunk_entity_rows),
    )


async def _load_summary_chunk_rows(
    dataset_id: UUID,
    expected_summary_ids: set[UUID],
    session: AsyncSession,
) -> list[tuple[UUID, UUID]]:
    if not expected_summary_ids:
        return []

    result = await session.execute(_summary_chunk_statement(dataset_id, expected_summary_ids))
    return [(row[0], row[1]) for row in result.all()]


async def _load_chunk_entity_rows(
    dataset_id: UUID,
    chunk_ids: set[UUID],
    session: AsyncSession,
) -> list[tuple[UUID, UUID]]:
    if not chunk_ids:
        return []

    result = await session.execute(_chunk_entity_statement(dataset_id, chunk_ids))
    return [(row[0], row[1]) for row in result.all()]


def _summary_chunk_statement(dataset_id: UUID, expected_summary_ids: set[UUID]):
    summary_node = aliased(Node)
    chunk_node = aliased(Node)
    made_from_edge = aliased(Edge)

    return (
        select(summary_node.slug, chunk_node.slug)
        .select_from(summary_node)
        .join(
            made_from_edge,
            and_(
                summary_node.slug == made_from_edge.source_node_id,
                made_from_edge.dataset_id == dataset_id,
                made_from_edge.relationship_name == "made_from",
            ),
        )
        .join(
            chunk_node,
            and_(
                chunk_node.slug == made_from_edge.destination_node_id,
                chunk_node.dataset_id == dataset_id,
                chunk_node.type == "DocumentChunk",
            ),
        )
        .where(
            and_(
                summary_node.dataset_id == dataset_id,
                summary_node.type == "TextSummary",
                summary_node.slug.in_(expected_summary_ids),
            )
        )
        .distinct()
    )


def _chunk_entity_statement(dataset_id: UUID, chunk_ids: set[UUID]):
    chunk_node = aliased(Node)
    entity_node = aliased(Node)
    contains_edge = aliased(Edge)

    return (
        select(chunk_node.slug, entity_node.slug)
        .select_from(chunk_node)
        .join(
            contains_edge,
            and_(
                chunk_node.slug == contains_edge.source_node_id,
                contains_edge.dataset_id == dataset_id,
                contains_edge.label == "contains",
            ),
        )
        .join(
            entity_node,
            and_(
                entity_node.slug == contains_edge.destination_node_id,
                entity_node.dataset_id == dataset_id,
                entity_node.type == "Entity",
            ),
        )
        .where(
            and_(
                chunk_node.dataset_id == dataset_id,
                chunk_node.type == "DocumentChunk",
                chunk_node.slug.in_(chunk_ids),
            )
        )
        .distinct()
    )


def _group_summary_chunk_ids(rows: list[tuple[UUID, UUID]]) -> dict[UUID, set[UUID]]:
    summary_chunk_ids: dict[UUID, set[UUID]] = {}
    for summary_id, chunk_id in rows:
        summary_chunk_ids.setdefault(summary_id, set()).add(chunk_id)
    return summary_chunk_ids


def _flatten_chunk_ids(summary_chunk_ids: dict[UUID, set[UUID]]) -> set[UUID]:
    return {
        chunk_id
        for chunk_ids_for_summary in summary_chunk_ids.values()
        for chunk_id in chunk_ids_for_summary
    }


def _group_chunk_entity_ids(rows: list[tuple[UUID, UUID]]) -> dict[UUID, set[UUID]]:
    chunk_entity_ids: dict[UUID, set[UUID]] = {}
    for chunk_id, entity_id in rows:
        chunk_entity_ids.setdefault(chunk_id, set()).add(entity_id)
    return chunk_entity_ids
