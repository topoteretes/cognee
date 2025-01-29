from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.processing.document_types import Document
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


async def calculate_graph_metrics(graph_data):
    nodes, edges = graph_data
    graph_metrics = {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
    }
    return graph_metrics


async def store_descriptive_metrics(data_points: list[DataPoint]):
    db_engine = get_relational_engine()
    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()

    token_count_sum = await fetch_token_count(db_engine)
    graph_metrics = await calculate_graph_metrics(graph_data)

    table_name = "graph_metrics_table"
    metrics_dict = {"id": uuid.uuid4(), "num_tokens": token_count_sum} | graph_metrics

    await db_engine.insert_data(table_name, metrics_dict)
    return data_points
