from uuid import UUID
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.context_global_variables import set_database_global_context_variables

logger = get_logger("GraphDataFormatter")


async def get_formatted_graph_data(dataset_id: UUID, user_id: UUID):
    logger.info(f"Starting graph data formatting for dataset_id={dataset_id}, user_id={user_id}")

    import time

    start_time = time.time()

    try:
        logger.debug("Setting database global context variables")
        await set_database_global_context_variables(dataset_id, user_id)

        logger.debug("Getting graph engine")
        graph_client = await get_graph_engine()

        logger.debug("Retrieving raw graph data from database")
        (nodes, edges) = await graph_client.get_graph_data()

        logger.info(f"Retrieved raw graph data: {len(nodes)} nodes, {len(edges)} edges")

        if len(nodes) == 0:
            logger.warning("No nodes found in graph data")
            return {"nodes": [], "edges": []}

        # Format nodes
        logger.debug("Starting node formatting")
        formatted_nodes = []
        processed_nodes = 0
        failed_nodes = 0

        for node in nodes:
            try:
                node_id = str(node[0])
                node_props = node[1]

                # Determine node label
                if "name" in node_props and node_props["name"] != "":
                    label = node_props["name"]
                else:
                    label = f"{node_props.get('type', 'Unknown')}_{node_id}"

                # Extract additional properties (excluding system fields)
                properties = {
                    key: value
                    for key, value in node_props.items()
                    if key not in ["id", "type", "name", "created_at", "updated_at"]
                    and value is not None
                }

                formatted_node = {
                    "id": node_id,
                    "label": label,
                    "type": node_props.get("type", "Unknown"),
                    "properties": properties,
                }
                formatted_nodes.append(formatted_node)
                processed_nodes += 1

            except Exception as e:
                failed_nodes += 1
                logger.warning(f"Failed to format node {node}: {str(e)}")

        logger.info(
            f"Node formatting completed: {processed_nodes} processed, {failed_nodes} failed"
        )

        # Format edges
        logger.debug("Starting edge formatting")
        formatted_edges = []
        processed_edges = 0
        failed_edges = 0

        for edge in edges:
            try:
                formatted_edge = {
                    "source": str(edge[0]),
                    "target": str(edge[1]),
                    "label": str(edge[2]),
                }
                formatted_edges.append(formatted_edge)
                processed_edges += 1

            except Exception as e:
                failed_edges += 1
                logger.warning(f"Failed to format edge {edge}: {str(e)}")

        logger.info(
            f"Edge formatting completed: {processed_edges} processed, {failed_edges} failed"
        )

        # Prepare final result
        result = {
            "nodes": formatted_nodes,
            "edges": formatted_edges,
        }

        formatting_time = time.time() - start_time
        logger.info(f"Graph data formatting completed in {formatting_time:.2f} seconds")
        logger.info(f"Final result: {len(formatted_nodes)} nodes, {len(formatted_edges)} edges")

        # Log graph statistics
        if len(formatted_nodes) > 0:
            node_types = {}
            for node in formatted_nodes:
                node_type = node.get("type", "Unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            logger.debug(f"Node types distribution: {node_types}")

        if len(formatted_edges) > 0:
            edge_labels = {}
            for edge in formatted_edges:
                edge_label = edge.get("label", "Unknown")
                edge_labels[edge_label] = edge_labels.get(edge_label, 0) + 1
            logger.debug(f"Edge labels distribution: {edge_labels}")

        return result

    except Exception as e:
        logger.error(f"Error during graph data formatting: {str(e)}")
        raise
