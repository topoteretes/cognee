from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import with_async_session
from cognee.modules.graph.models import Edge, Node


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


@with_async_session
async def get_dataset_text_summary_ids(
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
