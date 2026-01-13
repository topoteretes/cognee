# cognee/tasks/memify/extract_usage_frequency.py
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from cognee.shared.logging_utils import get_logger
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.pipelines.tasks.task import Task
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

logger = get_logger("extract_usage_frequency")


async def extract_usage_frequency(
    subgraphs: List[CogneeGraph],
    time_window: timedelta = timedelta(days=7),
    min_interaction_threshold: int = 1,
) -> Dict[str, Any]:
    """
    Extract usage frequency from CogneeUserInteraction nodes.

    When save_interaction=True in cognee.search(), the system creates:
    - CogneeUserInteraction nodes (representing the query/answer interaction)
    - used_graph_element_to_answer edges (connecting interactions to graph elements used)

    This function tallies how often each graph element is referenced via these edges,
    enabling frequency-based ranking in downstream retrievers.

    :param subgraphs: List of CogneeGraph instances containing interaction data
    :param time_window: Time window to consider for interactions (default: 7 days)
    :param min_interaction_threshold: Minimum interactions to track (default: 1)
    :return: Dictionary containing node frequencies, edge frequencies, and metadata
    """
    current_time = datetime.now()
    cutoff_time = current_time - time_window

    # Track frequencies for graph elements (nodes and edges)
    node_frequencies = {}
    edge_frequencies = {}
    relationship_type_frequencies = {}

    # Track interaction metadata
    interaction_count = 0
    interactions_in_window = 0

    logger.info(f"Extracting usage frequencies from {len(subgraphs)} subgraphs")
    logger.info(f"Time window: {time_window}, Cutoff: {cutoff_time.isoformat()}")

    for subgraph in subgraphs:
        # Find all CogneeUserInteraction nodes
        interaction_nodes = {}
        for node_id, node in subgraph.nodes.items():
            node_type = node.attributes.get("type") or node.attributes.get("node_type")

            if node_type == "CogneeUserInteraction":
                # Parse and validate timestamp
                timestamp_value = node.attributes.get("timestamp") or node.attributes.get(
                    "created_at"
                )
                if timestamp_value is not None:
                    try:
                        # Handle various timestamp formats
                        interaction_time = None

                        if isinstance(timestamp_value, datetime):
                            # Already a Python datetime
                            interaction_time = timestamp_value
                        elif isinstance(timestamp_value, (int, float)):
                            # Unix timestamp (assume milliseconds if > 10 digits)
                            if timestamp_value > 10000000000:
                                # Milliseconds since epoch
                                interaction_time = datetime.fromtimestamp(timestamp_value / 1000.0)
                            else:
                                # Seconds since epoch
                                interaction_time = datetime.fromtimestamp(timestamp_value)
                        elif isinstance(timestamp_value, str):
                            # Try different string formats
                            if timestamp_value.isdigit():
                                # Numeric string - treat as Unix timestamp
                                ts_int = int(timestamp_value)
                                if ts_int > 10000000000:
                                    interaction_time = datetime.fromtimestamp(ts_int / 1000.0)
                                else:
                                    interaction_time = datetime.fromtimestamp(ts_int)
                            else:
                                # ISO format string
                                interaction_time = datetime.fromisoformat(timestamp_value)
                        elif hasattr(timestamp_value, "to_native"):
                            # Neo4j datetime object - convert to Python datetime
                            interaction_time = timestamp_value.to_native()
                        elif hasattr(timestamp_value, "year") and hasattr(timestamp_value, "month"):
                            # Datetime-like object - extract components
                            try:
                                interaction_time = datetime(
                                    year=timestamp_value.year,
                                    month=timestamp_value.month,
                                    day=timestamp_value.day,
                                    hour=getattr(timestamp_value, "hour", 0),
                                    minute=getattr(timestamp_value, "minute", 0),
                                    second=getattr(timestamp_value, "second", 0),
                                    microsecond=getattr(timestamp_value, "microsecond", 0),
                                )
                            except (AttributeError, ValueError):
                                pass

                        if interaction_time is None:
                            # Last resort: try converting to string and parsing
                            str_value = str(timestamp_value)
                            if str_value.isdigit():
                                ts_int = int(str_value)
                                if ts_int > 10000000000:
                                    interaction_time = datetime.fromtimestamp(ts_int / 1000.0)
                                else:
                                    interaction_time = datetime.fromtimestamp(ts_int)
                            else:
                                interaction_time = datetime.fromisoformat(str_value)

                        if interaction_time is None:
                            raise ValueError(f"Could not parse timestamp: {timestamp_value}")

                        # Make sure it's timezone-naive for comparison
                        if interaction_time.tzinfo is not None:
                            interaction_time = interaction_time.replace(tzinfo=None)

                        interaction_nodes[node_id] = {
                            "node": node,
                            "timestamp": interaction_time,
                            "in_window": interaction_time >= cutoff_time,
                        }
                        interaction_count += 1
                        if interaction_time >= cutoff_time:
                            interactions_in_window += 1
                    except (ValueError, TypeError, AttributeError, OSError) as e:
                        logger.warning(
                            f"Failed to parse timestamp for interaction node {node_id}: {e}"
                        )
                        logger.debug(
                            f"Timestamp value type: {type(timestamp_value)}, value: {timestamp_value}"
                        )

        # Process edges to find graph elements used in interactions
        for edge in subgraph.edges:
            relationship_type = edge.attributes.get("relationship_type")

            # Look for 'used_graph_element_to_answer' edges
            if relationship_type == "used_graph_element_to_answer":
                # node1 should be the CogneeUserInteraction, node2 is the graph element
                source_id = str(edge.node1.id)
                target_id = str(edge.node2.id)

                # Check if source is an interaction node in our time window
                if source_id in interaction_nodes:
                    interaction_data = interaction_nodes[source_id]

                    if interaction_data["in_window"]:
                        # Count the graph element (target node) being used
                        node_frequencies[target_id] = node_frequencies.get(target_id, 0) + 1

                        # Also track what type of element it is for analytics
                        target_node = subgraph.get_node(target_id)
                        if target_node:
                            element_type = target_node.attributes.get(
                                "type"
                            ) or target_node.attributes.get("node_type")
                            if element_type:
                                relationship_type_frequencies[element_type] = (
                                    relationship_type_frequencies.get(element_type, 0) + 1
                                )

            # Also track general edge usage patterns
            elif relationship_type and relationship_type != "used_graph_element_to_answer":
                # Check if either endpoint is referenced in a recent interaction
                source_id = str(edge.node1.id)
                target_id = str(edge.node2.id)

                # If this edge connects to any frequently accessed nodes, track the edge type
                if source_id in node_frequencies or target_id in node_frequencies:
                    edge_key = f"{relationship_type}:{source_id}:{target_id}"
                    edge_frequencies[edge_key] = edge_frequencies.get(edge_key, 0) + 1

    # Filter frequencies above threshold
    filtered_node_frequencies = {
        node_id: freq
        for node_id, freq in node_frequencies.items()
        if freq >= min_interaction_threshold
    }

    filtered_edge_frequencies = {
        edge_key: freq
        for edge_key, freq in edge_frequencies.items()
        if freq >= min_interaction_threshold
    }

    logger.info(
        f"Processed {interactions_in_window}/{interaction_count} interactions in time window"
    )
    logger.info(
        f"Found {len(filtered_node_frequencies)} nodes and {len(filtered_edge_frequencies)} edges "
        f"above threshold (min: {min_interaction_threshold})"
    )
    logger.info(f"Element type distribution: {relationship_type_frequencies}")

    return {
        "node_frequencies": filtered_node_frequencies,
        "edge_frequencies": filtered_edge_frequencies,
        "element_type_frequencies": relationship_type_frequencies,
        "total_interactions": interaction_count,
        "interactions_in_window": interactions_in_window,
        "time_window_days": time_window.days,
        "last_processed_timestamp": current_time.isoformat(),
        "cutoff_timestamp": cutoff_time.isoformat(),
    }


async def add_frequency_weights(
    graph_adapter: GraphDBInterface, usage_frequencies: Dict[str, Any]
) -> None:
    """
    Add frequency weights to graph nodes and edges using the graph adapter.

    Uses direct Cypher queries for Neo4j adapter compatibility.
    Writes frequency_weight properties back to the graph for use in:
    - Ranking frequently referenced entities higher during retrieval
    - Adjusting scoring for completion strategies
    - Exposing usage metrics in dashboards or audits

    :param graph_adapter: Graph database adapter interface
    :param usage_frequencies: Calculated usage frequencies from extract_usage_frequency
    """
    node_frequencies = usage_frequencies.get("node_frequencies", {})
    edge_frequencies = usage_frequencies.get("edge_frequencies", {})

    logger.info(f"Adding frequency weights to {len(node_frequencies)} nodes")

    # Check adapter type and use appropriate method
    adapter_type = type(graph_adapter).__name__
    logger.info(f"Using adapter: {adapter_type}")

    nodes_updated = 0
    nodes_failed = 0

    # Determine which method to use based on adapter type
    use_neo4j_cypher = adapter_type == "Neo4jAdapter" and hasattr(graph_adapter, "query")
    use_kuzu_query = adapter_type == "KuzuAdapter" and hasattr(graph_adapter, "query")
    use_get_update = hasattr(graph_adapter, "get_node_by_id") and hasattr(
        graph_adapter, "update_node_properties"
    )

    # Method 1: Neo4j Cypher with SET (creates properties on the fly)
    if use_neo4j_cypher:
        try:
            logger.info("Using Neo4j Cypher SET method")
            last_updated = usage_frequencies.get("last_processed_timestamp")

            for node_id, frequency in node_frequencies.items():
                try:
                    query = """
                    MATCH (n)
                    WHERE n.id = $node_id
                    SET n.frequency_weight = $frequency,
                        n.frequency_updated_at = $updated_at
                    RETURN n.id as id
                    """

                    result = await graph_adapter.query(
                        query,
                        params={
                            "node_id": node_id,
                            "frequency": frequency,
                            "updated_at": last_updated,
                        },
                    )

                    if result and len(result) > 0:
                        nodes_updated += 1
                    else:
                        logger.warning(f"Node {node_id} not found or not updated")
                        nodes_failed += 1

                except Exception as e:
                    logger.error(f"Error updating node {node_id}: {e}")
                    nodes_failed += 1

            logger.info(f"Node update complete: {nodes_updated} succeeded, {nodes_failed} failed")

        except Exception as e:
            logger.error(f"Neo4j Cypher update failed: {e}")
            use_neo4j_cypher = False

    # Method 2: Kuzu - use get_node + add_node (updates via re-adding with same ID)
    elif (
        use_kuzu_query and hasattr(graph_adapter, "get_node") and hasattr(graph_adapter, "add_node")
    ):
        logger.info("Using Kuzu get_node + add_node method")
        last_updated = usage_frequencies.get("last_processed_timestamp")

        for node_id, frequency in node_frequencies.items():
            try:
                # Get the existing node (returns a dict)
                existing_node_dict = await graph_adapter.get_node(node_id)

                if existing_node_dict:
                    # Update the dict with new properties
                    existing_node_dict["frequency_weight"] = frequency
                    existing_node_dict["frequency_updated_at"] = last_updated

                    # Kuzu's add_node likely just takes the dict directly, not a Node object
                    # Try passing the dict directly first
                    try:
                        await graph_adapter.add_node(existing_node_dict)
                        nodes_updated += 1
                    except Exception as dict_error:
                        # If dict doesn't work, try creating a Node object
                        logger.debug(f"Dict add failed, trying Node object: {dict_error}")

                        try:
                            from cognee.infrastructure.engine import Node

                            # Try different Node constructor patterns
                            try:
                                # Pattern 1: Just properties
                                node_obj = Node(existing_node_dict)
                            except Exception:
                                # Pattern 2: Type and properties
                                node_obj = Node(
                                    type=existing_node_dict.get("type", "Unknown"),
                                    **existing_node_dict,
                                )

                            await graph_adapter.add_node(node_obj)
                            nodes_updated += 1
                        except Exception as node_error:
                            logger.error(f"Both dict and Node object failed: {node_error}")
                            nodes_failed += 1
                else:
                    logger.warning(f"Node {node_id} not found in graph")
                    nodes_failed += 1

            except Exception as e:
                logger.error(f"Error updating node {node_id}: {e}")
                nodes_failed += 1

        logger.info(f"Node update complete: {nodes_updated} succeeded, {nodes_failed} failed")

    # Method 3: Generic get_node_by_id + update_node_properties
    elif use_get_update:
        logger.info("Using get/update method for adapter")
        for node_id, frequency in node_frequencies.items():
            try:
                # Get current node data
                node_data = await graph_adapter.get_node_by_id(node_id)

                if node_data:
                    # Tweak the properties dict - add frequency_weight
                    if isinstance(node_data, dict):
                        properties = node_data.get("properties", {})
                    else:
                        properties = getattr(node_data, "properties", {}) or {}

                    # Update with frequency weight
                    properties["frequency_weight"] = frequency
                    properties["frequency_updated_at"] = usage_frequencies.get(
                        "last_processed_timestamp"
                    )

                    # Write back via adapter
                    await graph_adapter.update_node_properties(node_id, properties)
                    nodes_updated += 1
                else:
                    logger.warning(f"Node {node_id} not found in graph")
                    nodes_failed += 1

            except Exception as e:
                logger.error(f"Error updating node {node_id}: {e}")
                nodes_failed += 1

        logger.info(f"Node update complete: {nodes_updated} succeeded, {nodes_failed} failed")
        for node_id, frequency in node_frequencies.items():
            try:
                # Get current node data
                node_data = await graph_adapter.get_node_by_id(node_id)

                if node_data:
                    # Tweak the properties dict - add frequency_weight
                    if isinstance(node_data, dict):
                        properties = node_data.get("properties", {})
                    else:
                        properties = getattr(node_data, "properties", {}) or {}

                    # Update with frequency weight
                    properties["frequency_weight"] = frequency
                    properties["frequency_updated_at"] = usage_frequencies.get(
                        "last_processed_timestamp"
                    )

                    # Write back via adapter
                    await graph_adapter.update_node_properties(node_id, properties)
                    nodes_updated += 1
                else:
                    logger.warning(f"Node {node_id} not found in graph")
                    nodes_failed += 1

            except Exception as e:
                logger.error(f"Error updating node {node_id}: {e}")
                nodes_failed += 1

    # If no method is available
    if not use_neo4j_cypher and not use_kuzu_query and not use_get_update:
        logger.error(f"Adapter {adapter_type} does not support required update methods")
        logger.error(
            "Required: either 'query' method or both 'get_node_by_id' and 'update_node_properties'"
        )
        return

    # Update edge frequencies
    # Note: Edge property updates are backend-specific
    if edge_frequencies:
        logger.info(f"Processing {len(edge_frequencies)} edge frequency entries")

        edges_updated = 0
        edges_failed = 0

        for edge_key, frequency in edge_frequencies.items():
            try:
                # Parse edge key: "relationship_type:source_id:target_id"
                parts = edge_key.split(":", 2)
                if len(parts) == 3:
                    relationship_type, source_id, target_id = parts

                    # Try to update edge if adapter supports it
                    if hasattr(graph_adapter, "update_edge_properties"):
                        edge_properties = {
                            "frequency_weight": frequency,
                            "frequency_updated_at": usage_frequencies.get(
                                "last_processed_timestamp"
                            ),
                        }

                        await graph_adapter.update_edge_properties(
                            source_id, target_id, relationship_type, edge_properties
                        )
                        edges_updated += 1
                    else:
                        # Fallback: store in metadata or log
                        logger.debug(
                            f"Adapter doesn't support update_edge_properties for "
                            f"{relationship_type} ({source_id} -> {target_id})"
                        )

            except Exception as e:
                logger.error(f"Error updating edge {edge_key}: {e}")
                edges_failed += 1

        if edges_updated > 0:
            logger.info(f"Edge update complete: {edges_updated} succeeded, {edges_failed} failed")
        else:
            logger.info(
                "Edge frequency updates skipped (adapter may not support edge property updates)"
            )

    # Store aggregate statistics as metadata if supported
    if hasattr(graph_adapter, "set_metadata"):
        try:
            metadata = {
                "element_type_frequencies": usage_frequencies.get("element_type_frequencies", {}),
                "total_interactions": usage_frequencies.get("total_interactions", 0),
                "interactions_in_window": usage_frequencies.get("interactions_in_window", 0),
                "last_frequency_update": usage_frequencies.get("last_processed_timestamp"),
            }
            await graph_adapter.set_metadata("usage_frequency_stats", metadata)
            logger.info("Stored usage frequency statistics as metadata")
        except Exception as e:
            logger.warning(f"Could not store usage statistics as metadata: {e}")


async def create_usage_frequency_pipeline(
    graph_adapter: GraphDBInterface,
    time_window: timedelta = timedelta(days=7),
    min_interaction_threshold: int = 1,
    batch_size: int = 100,
) -> tuple:
    """
    Create memify pipeline entry for usage frequency tracking.

    This follows the same pattern as feedback enrichment flows, allowing
    the frequency update to run end-to-end in a custom memify pipeline.

    Use case example:
        extraction_tasks, enrichment_tasks = await create_usage_frequency_pipeline(
            graph_adapter=my_adapter,
            time_window=timedelta(days=30),
            min_interaction_threshold=2
        )

        # Run in memify pipeline
        pipeline = Pipeline(extraction_tasks + enrichment_tasks)
        results = await pipeline.run()

    :param graph_adapter: Graph database adapter
    :param time_window: Time window for counting interactions (default: 7 days)
    :param min_interaction_threshold: Minimum interactions to track (default: 1)
    :param batch_size: Batch size for processing (default: 100)
    :return: Tuple of (extraction_tasks, enrichment_tasks)
    """
    logger.info("Creating usage frequency pipeline")
    logger.info(f"Config: time_window={time_window}, threshold={min_interaction_threshold}")

    extraction_tasks = [
        Task(
            extract_usage_frequency,
            time_window=time_window,
            min_interaction_threshold=min_interaction_threshold,
        )
    ]

    enrichment_tasks = [
        Task(
            add_frequency_weights,
            graph_adapter=graph_adapter,
            task_config={"batch_size": batch_size},
        )
    ]

    return extraction_tasks, enrichment_tasks


async def run_usage_frequency_update(
    graph_adapter: GraphDBInterface,
    subgraphs: List[CogneeGraph],
    time_window: timedelta = timedelta(days=7),
    min_interaction_threshold: int = 1,
) -> Dict[str, Any]:
    """
    Convenience function to run the complete usage frequency update pipeline.

    This is the main entry point for updating frequency weights on graph elements
    based on CogneeUserInteraction data from cognee.search(save_interaction=True).

    Example usage:
        # After running searches with save_interaction=True
        from cognee.tasks.memify.extract_usage_frequency import run_usage_frequency_update

        # Get the graph with interactions
        graph = await get_cognee_graph_with_interactions()

        # Update frequency weights
        stats = await run_usage_frequency_update(
            graph_adapter=graph_adapter,
            subgraphs=[graph],
            time_window=timedelta(days=30),  # Last 30 days
            min_interaction_threshold=2       # At least 2 uses
        )

        print(f"Updated {len(stats['node_frequencies'])} nodes")

    :param graph_adapter: Graph database adapter
    :param subgraphs: List of CogneeGraph instances with interaction data
    :param time_window: Time window for counting interactions
    :param min_interaction_threshold: Minimum interactions to track
    :return: Usage frequency statistics
    """
    logger.info("Starting usage frequency update")

    try:
        # Extract frequencies from interaction data
        usage_frequencies = await extract_usage_frequency(
            subgraphs=subgraphs,
            time_window=time_window,
            min_interaction_threshold=min_interaction_threshold,
        )

        # Add frequency weights back to the graph
        await add_frequency_weights(
            graph_adapter=graph_adapter, usage_frequencies=usage_frequencies
        )

        logger.info("Usage frequency update completed successfully")
        logger.info(
            f"Summary: {usage_frequencies['interactions_in_window']} interactions processed, "
            f"{len(usage_frequencies['node_frequencies'])} nodes weighted"
        )

        return usage_frequencies

    except Exception as e:
        logger.error(f"Error during usage frequency update: {str(e)}")
        raise


async def get_most_frequent_elements(
    graph_adapter: GraphDBInterface, top_n: int = 10, element_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the most frequently accessed graph elements.

    Useful for analytics dashboards and understanding user behavior.

    :param graph_adapter: Graph database adapter
    :param top_n: Number of top elements to return
    :param element_type: Optional filter by element type
    :return: List of elements with their frequency weights
    """
    logger.info(f"Retrieving top {top_n} most frequent elements")

    # This would need to be implemented based on the specific graph adapter's query capabilities
    # Pseudocode:
    # results = await graph_adapter.query_nodes_by_property(
    #     property_name='frequency_weight',
    #     order_by='DESC',
    #     limit=top_n,
    #     filters={'type': element_type} if element_type else None
    # )

    logger.warning("get_most_frequent_elements needs adapter-specific implementation")
    return []
