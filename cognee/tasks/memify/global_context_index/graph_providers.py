from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.graph.methods.get_global_context_graph_inputs import (
    DatasetEntityCounts,
    SummaryEntityLoadResult,
    load_dataset_graph_entity_input,
)

from .idf import compute_idf_from_counts


@dataclass
class GlobalContextGraphInput:
    entities_by_summary_id: dict[str, set[str]]
    idf_weights: dict[str, float]
    summary_entities: SummaryEntityLoadResult
    entity_counts: DatasetEntityCounts


async def load_global_context_graph_input(
    dataset_id: str | UUID,
    expected_summary_ids: Iterable[str | UUID],
    session: AsyncSession | None = None,
) -> GlobalContextGraphInput:
    expected_summary_id_list = list(expected_summary_ids)
    if session is None:
        graph_entity_input = await load_dataset_graph_entity_input(
            dataset_id,
            expected_summary_id_list,
        )
    else:
        graph_entity_input = await load_dataset_graph_entity_input(
            dataset_id,
            expected_summary_id_list,
            session=session,
        )

    return GlobalContextGraphInput(
        entities_by_summary_id=graph_entity_input.summary_entities.entities_by_summary_id,
        idf_weights=compute_idf_from_counts(
            graph_entity_input.entity_counts.chunk_count,
            graph_entity_input.entity_counts.entity_chunk_counts,
        ),
        summary_entities=graph_entity_input.summary_entities,
        entity_counts=graph_entity_input.entity_counts,
    )
