from typing import List, Dict, Any
from collections import defaultdict
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge
from cognee.shared.logging_utils import get_logger

logger = get_logger("WeightPropagationTask")
DEFAULT_WEIGHT = 0.5

async def propagate_importance_weights(data: List[CogneeGraph], task_config: Dict[str, Any] = None) -> List[
    CogneeGraph]:
    """
    Propagates and fuses importance weights from initial nodes (e.g., DocumentChunks)
    to their neighboring nodes and edges in the graph memory fragment using an
    Average Aggregation strategy.

    Args:
        data: A list containing the CogneeGraph memory fragment to be processed.
        task_config: Configuration dictionary (optional).

    Returns:
        The updated CogneeGraph memory fragment list.
    """
    if not data or not isinstance(data[0], CogneeGraph):
        logger.warning("No CogneeGraph memory fragment provided for weight propagation.")
        return data

    memory_fragment: CogneeGraph = data[0]

    logger.info("Starting importance weight propagation and fusion.")

    node_weight_contributions = defaultdict(list)

    all_nodes: List[Node] = list(memory_fragment.nodes.values())

    source_nodes: List[Node] = [
        node for node in all_nodes
        if node.attributes.get("importance_weight") is not None and 0.0 <= node.attributes.get("importance_weight",
                                                                                               -1) <= 1.0
    ]

    for source_node in source_nodes:
        initial_weight = source_node.attributes["importance_weight"]

        node_weight_contributions[source_node.id].append(initial_weight)

        for neighbor in source_node.get_skeleton_neighbours():
            node_weight_contributions[neighbor.id].append(initial_weight)

    updated_node_count = 0
    for node_id, weights in node_weight_contributions.items():
        if weights:
            avg_weight = sum(weights) / len(weights)
            target_node = memory_fragment.get_node(node_id)

            if target_node:
                target_node.add_attribute("importance_weight", round(avg_weight, 4))
                updated_node_count += 1
            else:
                logger.error(f"Target Node ID {node_id} unexpectedly not found in fragment during weight update.")

    logger.info(f"Propagation Phase 1 completed: Updated {updated_node_count} nodes via Average Aggregation.")

    all_edges: List[Edge] = memory_fragment.get_edges()

    updated_edge_count = 0
    for edge in all_edges:
        node1_weight = edge.node1.attributes.get("importance_weight", DEFAULT_WEIGHT)
        node2_weight = edge.node2.attributes.get("importance_weight", DEFAULT_WEIGHT)

        edge_weight = (node1_weight + node2_weight) / 2
        edge.add_attribute("importance_weight", round(edge_weight, 4))
        updated_edge_count += 1

    logger.info(f"Propagation Phase 2 completed: Updated {updated_edge_count} edges.")

    return data


class PropagateImportanceWeights(Task):
    """
    Cognee Task wrapper for propagating importance weights across the graph.
    """

    def __init__(self, **kwargs):
        super().__init__(propagate_importance_weights, **kwargs)

    async def __call__(self, data: List[CogneeGraph], task_config: Dict[str, Any] = None) -> List[CogneeGraph]:
        return await self.func(data, task_config=task_config)