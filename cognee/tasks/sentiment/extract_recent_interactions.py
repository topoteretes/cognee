"""Task for retrieving the most recent saved user interactions."""

from __future__ import annotations

from typing import Any, List
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger

from .models import InteractionSnapshot


logger = get_logger("sentiment.extract_recent_interactions")


async def extract_recent_interactions(_: Any, last_k: int = 20) -> List[InteractionSnapshot]:
    """Fetch the most recent Cognee user interactions for downstream sentiment analysis."""

    if last_k <= 0:
        logger.debug("last_k <= 0 provided; skipping interaction extraction")
        return []

    graph_engine = await get_graph_engine()
    interaction_ids = await graph_engine.get_last_user_interaction_ids(limit=last_k)

    if not interaction_ids:
        logger.debug("No recent interactions found for sentiment extraction")
        return []

    nodes = await graph_engine.get_nodes(interaction_ids)
    nodes_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}

    interactions: List[InteractionSnapshot] = []

    for interaction_id in interaction_ids:
        raw = nodes_by_id.get(str(interaction_id))
        if not raw:
            continue

        question = (raw.get("question") or "").strip()
        answer = (raw.get("answer") or "").strip()
        context = (raw.get("context") or "").strip()

        if not question and not answer:
            continue

        try:
            interaction_uuid = UUID(str(raw.get("id")))
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping interaction with invalid UUID: %s", exc)
            continue

        interactions.append(
            InteractionSnapshot(
                interaction_id=interaction_uuid,
                question=question,
                answer=answer,
                context=context,
            )
        )

    return interactions
