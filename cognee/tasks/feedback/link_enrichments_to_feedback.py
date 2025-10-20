from __future__ import annotations

from typing import List, Tuple
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.tasks.storage import index_graph_edges
from cognee.shared.logging_utils import get_logger

from .models import FeedbackEnrichment


logger = get_logger("link_enrichments_to_feedback")


def _create_edge_tuple(
    source_id: UUID, target_id: UUID, relationship_name: str
) -> Tuple[UUID, UUID, str, dict]:
    """Create an edge tuple with proper properties structure."""
    return (
        source_id,
        target_id,
        relationship_name,
        {
            "relationship_name": relationship_name,
            "source_node_id": source_id,
            "target_node_id": target_id,
            "ontology_valid": False,
        },
    )


async def link_enrichments_to_feedback(
    enrichments: List[FeedbackEnrichment],
) -> List[FeedbackEnrichment]:
    """Manually create edges from enrichments to original feedback/interaction nodes."""
    if not enrichments:
        logger.info("No enrichments provided; returning empty list")
        return []

    relationships = []

    for enrichment in enrichments:
        enrichment_id = enrichment.id
        feedback_id = enrichment.feedback_id
        interaction_id = enrichment.interaction_id

        if enrichment_id and feedback_id:
            enriches_feedback_edge = _create_edge_tuple(
                enrichment_id, feedback_id, "enriches_feedback"
            )
            relationships.append(enriches_feedback_edge)

        if enrichment_id and interaction_id:
            improves_interaction_edge = _create_edge_tuple(
                enrichment_id, interaction_id, "improves_interaction"
            )
            relationships.append(improves_interaction_edge)

    if relationships:
        graph_engine = await get_graph_engine()
        await graph_engine.add_edges(relationships)
        await index_graph_edges(relationships)
        logger.info("Linking enrichments to feedback", edge_count=len(relationships))

    logger.info("Linked enrichments", enrichment_count=len(enrichments))
    return enrichments
