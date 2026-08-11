"""
Memory decay pipeline for reducing feedback_weight of unaccessed memories.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from uuid import UUID
import os

from sqlalchemy import select, or_

import cognee
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, DatasetData
from cognee.modules.graph.methods import get_global_data_related_nodes, get_global_data_related_edges
from cognee.shared.logging_utils import get_logger

logger = get_logger("decay_memory")

async def decay_memory(
    elapsed_hours: float = 24.0,
    half_life_days: float = 7.0,
    prune_threshold: float = 0.05,
    dry_run: bool = False,
    user_id: Optional[UUID] = None,
) -> Dict[str, Any]:
    """
    Decay feedback_weights for memories not accessed recently, pruning documents if they fall below threshold.
    """
    if os.getenv("ENABLE_LAST_ACCESSED", "false").lower() != "true":
        logger.warning("Decay skipped: ENABLE_LAST_ACCESSED is not enabled.")
        return {"status": "skipped", "reason": "ENABLE_LAST_ACCESSED not enabled"}

    cutoff_date = datetime.now(timezone.utc) - timedelta(hours=elapsed_hours)
    db_engine = get_relational_engine()
    graph_engine = await get_graph_engine()
    
    decay_factor = (0.5) ** (elapsed_hours / (half_life_days * 24.0))

    async with db_engine.get_async_session() as session:
        query = (
            select(Data, DatasetData)
            .join(DatasetData, Data.id == DatasetData.data_id)
            .where(or_(Data.last_accessed < cutoff_date, Data.last_accessed.is_(None)))
        )
        if user_id:
            from cognee.modules.data.models import Dataset
            query = query.join(Dataset, DatasetData.dataset_id == Dataset.id).where(
                Dataset.owner_id == user_id
            )
        result = await session.execute(query)
        unaccessed_data = result.all()

    processed_count = 0
    decayed_count = 0
    pruned_count = 0

    from cognee.modules.users.methods import get_default_user
    user = await get_default_user() if user_id is None else None

    # Process each unaccessed document
    async with db_engine.get_async_session() as session:
        for data, dataset_data in unaccessed_data:
            processed_count += 1
            # Get nodes and edges related to this data
            nodes = await get_global_data_related_nodes(data.id, session=session, dataset_id=dataset_data.dataset_id)
            edges = await get_global_data_related_edges(data.id, session=session, dataset_id=dataset_data.dataset_id)
            
            node_ids = [str(n.slug) for n in nodes]
            edge_ids = [str(e.slug) for e in edges]
            
            if not node_ids and not edge_ids:
                continue

            # Fetch current weights
            node_weights = await graph_engine.get_node_feedback_weights(node_ids) if node_ids else {}
            edge_weights = await graph_engine.get_edge_feedback_weights(edge_ids) if edge_ids else {}

            # Calculate new weights
            updated_node_weights = {nid: max(0.0, float(w) * decay_factor) for nid, w in node_weights.items()}
            updated_edge_weights = {eid: max(0.0, float(w) * decay_factor) for eid, w in edge_weights.items()}
            
            max_node_weight = max(updated_node_weights.values()) if updated_node_weights else 0.0
            max_edge_weight = max(updated_edge_weights.values()) if updated_edge_weights else 0.0
            max_weight = max(max_node_weight, max_edge_weight)

            if max_weight > 0 and max_weight < prune_threshold:
                if not dry_run:
                    await cognee.delete(data_id=data.id, dataset_id=dataset_data.dataset_id, mode="hard", user=user)
                pruned_count += 1
                logger.info(f"Pruned data {data.id} from dataset {dataset_data.dataset_id} (max weight {max_weight:.4f} < {prune_threshold})")
            else:
                if not dry_run:
                    if updated_node_weights:
                        await graph_engine.set_node_feedback_weights(updated_node_weights)
                    if updated_edge_weights:
                        await graph_engine.set_edge_feedback_weights(updated_edge_weights)
                decayed_count += 1
                logger.info(f"Decayed data {data.id} in dataset {dataset_data.dataset_id}")

    return {
        "status": "completed",
        "processed_count": processed_count,
        "decayed_count": decayed_count,
        "pruned_count": pruned_count,
        "decay_factor": decay_factor,
        "dry_run": dry_run
    }
