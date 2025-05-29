from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.GraphMetrics import GraphMetrics
from cognee.shared.logging_utils import get_logger

logger = get_logger()

async def calculate_graph_metrics(data=None, task_config=None):
    """Calculate metrics for the graph and store them in the database."""
    try:
        # Get the graph engine
        graph_engine = await get_graph_engine()
        
        # Calculate metrics
        metrics = await graph_engine.get_graph_metrics(include_optional=True)
        
        # Create metrics record
        graph_metrics = GraphMetrics(
            num_nodes=metrics["num_nodes"],
            num_edges=metrics["num_edges"],
            mean_degree=metrics["mean_degree"],
            edge_density=metrics["edge_density"],
            num_connected_components=metrics["num_connected_components"],
            sizes_of_connected_components=metrics["sizes_of_connected_components"],
            num_selfloops=metrics["num_selfloops"],
            diameter=metrics["diameter"],
            avg_shortest_path_length=metrics["avg_shortest_path_length"],
            avg_clustering=metrics["avg_clustering"]
        )
        
        # Store in database
        relational_engine = get_relational_engine()
        async with relational_engine.get_async_session() as session:
            session.add(graph_metrics)
            await session.commit()
            
        logger.info("Successfully calculated and stored graph metrics")
        return data
        
    except Exception as e:
        logger.error(f"Failed to calculate graph metrics: {e}")
        raise