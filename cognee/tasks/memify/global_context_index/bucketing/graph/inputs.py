from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine_async
from cognee.modules.graph.methods.get_global_context_graph_inputs import (
    load_dataset_graph_entity_input,
)

from .scoring import compute_idf_from_counts


@dataclass(frozen=True)
class GraphBucketingInputs:
    entities_by_summary_id: dict[str, set[str]]
    idf_weights: dict[str, float]
    entity_type_by_entity_id: dict[str, str]
    type_idf_weights: dict[str, float]
    entity_relations: list[tuple[str, str, str]]
    edge_type_embeddings: dict[str, list[float]]


async def load_graph_bucketing_inputs(
    dataset_id: str | UUID,
    expected_summary_ids: Iterable[str | UUID],
    session: AsyncSession | None = None,
) -> GraphBucketingInputs:
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

    edge_type_embeddings = await _embed_relationship_names(graph_entity_input.entity_relations)

    return GraphBucketingInputs(
        entities_by_summary_id=graph_entity_input.summary_entities.entities_by_summary_id,
        idf_weights=compute_idf_from_counts(
            graph_entity_input.entity_counts.chunk_count,
            graph_entity_input.entity_counts.entity_chunk_counts,
        ),
        entity_type_by_entity_id=graph_entity_input.entity_types.entity_type_by_entity_id,
        type_idf_weights=compute_idf_from_counts(
            graph_entity_input.entity_counts.chunk_count,
            graph_entity_input.entity_types.entity_type_chunk_counts,
        ),
        entity_relations=graph_entity_input.entity_relations,
        edge_type_embeddings=edge_type_embeddings,
    )


async def _embed_relationship_names(
    entity_relations: list[tuple[str, str, str]],
) -> dict[str, list[float]]:
    """
    Embed each distinct relationship name once, so relationship_match() can
    compare two relationship names by embedding distance without re-embedding
    on every comparison. Re-embeds fresh rather than reusing stored EdgeType
    vectors, since not every vector adapter can return a stored vector back
    (PGVector's retrieve() drops it; only LanceDB's does not) -- re-embedding a
    handful of short strings once per build is cheap and adapter-agnostic.
    """
    relationship_names = sorted(
        {relationship_name for _, _, relationship_name in entity_relations}
    )
    if not relationship_names:
        return {}

    vector_engine = await get_vector_engine_async()
    vectors = await vector_engine.embedding_engine.embed_text(relationship_names)
    return dict(zip(relationship_names, vectors))


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
