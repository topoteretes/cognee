from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.relational import get_relational_engine
from sqlalchemy import select
from sqlalchemy.sql import func
from cognee.modules.data.models import Data
from cognee.modules.data.models import GraphMetrics
import uuid
from cognee.infrastructure.databases.graph import get_graph_engine


async def fetch_token_count(db_engine) -> int:
    """
    Fetches and sums token counts from the database.

    Returns:
        int: The total number of tokens across all documents.
    """

    async with db_engine.get_async_session() as session:
        token_count_sum = await session.execute(select(func.sum(Data.token_count)))
        token_count_sum = token_count_sum.scalar()

    return token_count_sum


async def store_descriptive_metrics(data_points: list[DataPoint], include_optional: bool):
    db_engine = get_relational_engine()
    graph_engine = await get_graph_engine()
    graph_metrics = await graph_engine.get_graph_metrics(include_optional)

    async with db_engine.get_async_session() as session:
        metrics = GraphMetrics(
            id=uuid.uuid4(),
            num_tokens=await fetch_token_count(db_engine),
            num_nodes=graph_metrics["num_nodes"],
            num_edges=graph_metrics["num_edges"],
            mean_degree=graph_metrics["mean_degree"],
            edge_density=graph_metrics["edge_density"],
            num_connected_components=graph_metrics["num_connected_components"],
            sizes_of_connected_components=graph_metrics["sizes_of_connected_components"],
            num_selfloops=graph_metrics["num_selfloops"],
            diameter=graph_metrics["diameter"],
            avg_shortest_path_length=graph_metrics["avg_shortest_path_length"],
            avg_clustering=graph_metrics["avg_clustering"],
        )

        session.add(metrics)
        await session.commit()

    return data_points
