"""Task for connecting sentiment nodes to their originating interactions."""

from __future__ import annotations

from typing import Iterable, List

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger

from cognee.tasks.storage.index_graph_edges import index_graph_edges
from .models import InteractionSentiment


logger = get_logger("sentiment.link_sentiment_to_interactions")


async def link_sentiment_to_interactions(
    sentiments: Iterable[InteractionSentiment],
) -> List[InteractionSentiment]:
    """Create graph edges between interactions and their sentiment annotations."""

    sentiments = list(sentiments or [])
    if not sentiments:
        return []

    graph_engine = await get_graph_engine()

    edges = []
    for sentiment in sentiments:
        source_id = str(sentiment.interaction_id)
        target_id = str(sentiment.id)

        edges.append(
            (
                source_id,
                target_id,
                "has_sentiment",
                {
                    "relationship_name": "has_sentiment",
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                    "sentiment": sentiment.sentiment.value,
                    "confidence": sentiment.confidence,
                },
            )
        )

    await graph_engine.add_edges(edges)
    await index_graph_edges(edges)

    logger.debug("Linked %d sentiments to interactions", len(edges))

    return sentiments
