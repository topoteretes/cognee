from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.graph.methods.get_global_context_graph_inputs import (
    load_dataset_graph_entity_input,
)

from .scoring import compute_idf_from_counts


async def load_graph_bucketing_inputs(
    dataset_id: str | UUID,
    expected_summary_ids: Iterable[str | UUID],
    session: AsyncSession | None = None,
) -> tuple[dict[str, set[str]], dict[str, float]]:
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

    validate_graph_bucketing_inputs(
        graph_entity_input.summary_entities.missing_made_from_summary_ids
    )
    return (
        graph_entity_input.summary_entities.entities_by_summary_id,
        compute_idf_from_counts(
            graph_entity_input.entity_counts.chunk_count,
            graph_entity_input.entity_counts.entity_chunk_counts,
        ),
    )


def validate_graph_bucketing_inputs(missing_made_from_summary_ids: set[str]) -> None:
    if not missing_made_from_summary_ids:
        return

    sample = ", ".join(sorted(missing_made_from_summary_ids)[:5])
    suffix = "..." if len(missing_made_from_summary_ids) > 5 else ""
    raise ValueError(
        'bucketing_strategy="graph" requires every TextSummary to have a made_from '
        "chunk edge. Missing made_from for "
        f"{len(missing_made_from_summary_ids)} summary id(s): {sample}{suffix}"
    )
