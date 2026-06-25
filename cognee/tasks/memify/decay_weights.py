from datetime import datetime, timezone
from typing import Any, Dict
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.shared.logging_utils import get_logger

logger = get_logger("decay_weights")


async def decay_weights(
    data: Any, decay_rate: float = 0.95, prune_threshold: float = 0.1
) -> Dict[str, int]:
    """
    Decay feedback_weight and frequency_weight on all Entity nodes and relationships.
    Time elapsed since the last update (updated_at) determines the decay factor.
    Nodes falling below the prune_threshold for both weights are pruned from the graph.
    """
    graph_engine = await get_graph_engine()

    try:
        nodes, edges = await graph_engine.get_filtered_graph_data([{"type": ["Entity"]}])
    except Exception as e:
        logger.warning("Decay weights: failed to fetch graph data: %s", e)
        return {"nodes_processed": 0, "edges_processed": 0, "nodes_deleted": 0}

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    node_weight_updates = {}
    node_frequency_updates = {}
    nodes_to_delete = []

    for node_id, props in nodes:
        feedback_weight = props.get("feedback_weight", 0.5)
        frequency_weight = props.get("frequency_weight", 0.0)

        # Extract updated_at or default to now
        updated_at = props.get("updated_at", now_ms)
        if isinstance(updated_at, str):
            try:
                # If timestamp string, parse or convert
                updated_at = int(updated_at)
            except ValueError:
                updated_at = now_ms

        elapsed_ms = now_ms - int(updated_at)
        elapsed_days = max(0.0, elapsed_ms / (1000 * 60 * 60 * 24))

        decay_factor = decay_rate**elapsed_days

        new_feedback = float(feedback_weight) * decay_factor
        new_frequency = float(frequency_weight) * decay_factor

        if new_feedback < prune_threshold and new_frequency < prune_threshold:
            nodes_to_delete.append(node_id)
        else:
            node_weight_updates[node_id] = new_feedback
            node_frequency_updates[node_id] = new_frequency

    # Apply node weight updates
    try:
        if node_weight_updates:
            await graph_engine.set_node_feedback_weights(node_weight_updates)
        if node_frequency_updates:
            await graph_engine.set_node_frequency_weights(node_frequency_updates)
    except NotImplementedError:
        logger.debug("Node weight update methods not implemented on current graph adapter.")
    except Exception as e:
        logger.warning("Failed to update node weights: %s", e)

    # Process edges
    edge_weight_updates = {}
    edge_frequency_updates = {}

    for edge in edges:
        if len(edge) < 4:
            continue
        edge_props = edge[3]
        if not isinstance(edge_props, dict):
            continue

        edge_object_id = edge_props.get("edge_object_id")
        if not edge_object_id:
            continue

        feedback_weight = edge_props.get("feedback_weight", 0.5)
        frequency_weight = edge_props.get("frequency_weight", 0.0)

        updated_at = edge_props.get("updated_at", now_ms)
        if isinstance(updated_at, str):
            try:
                updated_at = int(updated_at)
            except ValueError:
                updated_at = now_ms

        elapsed_ms = now_ms - int(updated_at)
        elapsed_days = max(0.0, elapsed_ms / (1000 * 60 * 60 * 24))

        decay_factor = decay_rate**elapsed_days

        new_feedback = float(feedback_weight) * decay_factor
        new_frequency = float(frequency_weight) * decay_factor

        edge_weight_updates[edge_object_id] = new_feedback
        edge_frequency_updates[edge_object_id] = new_frequency

    # Apply edge weight updates
    try:
        if edge_weight_updates:
            await graph_engine.set_edge_feedback_weights(edge_weight_updates)
        if edge_frequency_updates:
            await graph_engine.set_edge_frequency_weights(edge_frequency_updates)
    except NotImplementedError:
        logger.debug("Edge weight update methods not implemented on current graph adapter.")
    except Exception as e:
        logger.warning("Failed to update edge weights: %s", e)

    # Delete low-weight nodes
    if nodes_to_delete:
        try:
            await graph_engine.delete_nodes(nodes_to_delete)
            logger.info("Decay weights: pruned %d nodes from the graph", len(nodes_to_delete))
        except Exception as e:
            logger.warning("Failed to delete nodes during pruning: %s", e)

    return {
        "nodes_processed": len(nodes),
        "edges_processed": len(edges),
        "nodes_deleted": len(nodes_to_delete) if nodes_to_delete else 0,
    }
