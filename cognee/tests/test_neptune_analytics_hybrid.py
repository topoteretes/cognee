import os
from dotenv import load_dotenv
import asyncio
import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.data.processing.document_types import TextDocument
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
    NeptuneAnalyticsAdapter,
)

# Set up Amazon credentials in .env file and get the values from environment variables
load_dotenv()
graph_id = os.getenv("GRAPH_ID", "")

# get the default embedder
embedding_engine = get_embedding_engine()
na_graph = NeptuneAnalyticsAdapter(graph_id)
na_vector = NeptuneAnalyticsAdapter(graph_id, embedding_engine)

collection = "test_collection"

logger = get_logger("test_neptune_analytics_hybrid")


def setup_data():
    # Define nodes data before the main function
    # These nodes were defined using openAI from the following prompt:
    #
    # Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads
    #     that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It
    #     complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load
    #     the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's
    #     stored in Amazon S3.

    document = TextDocument(
        name="text.txt",
        raw_data_location="git/cognee/examples/database_examples/data_storage/data/text.txt",
        external_metadata="{}",
        mime_type="text/plain",
    )
    document_chunk = DocumentChunk(
        text="Neptune Analytics is an ideal choice for investigatory, exploratory, or data-science workloads \n    that require fast iteration for data, analytical and algorithmic processing, or vector search on graph data. It \n    complements Amazon Neptune Database, a popular managed graph database. To perform intensive analysis, you can load \n    the data from a Neptune Database graph or snapshot into Neptune Analytics. You can also load graph data that's \n    stored in Amazon S3.\n    ",
        chunk_size=187,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
    )

    graph_database = EntityType(name="graph database", description="graph database")
    neptune_analytics_entity = Entity(
        name="neptune analytics",
        description="A memory-optimized graph database engine for analytics that processes large amounts of graph data quickly.",
    )
    neptune_database_entity = Entity(
        name="amazon neptune database",
        description="A popular managed graph database that complements Neptune Analytics.",
    )

    storage = EntityType(name="storage", description="storage")
    storage_entity = Entity(
        name="amazon s3",
        description="A storage service provided by Amazon Web Services that allows storing graph data.",
    )

    nodes_data = [
        document,
        document_chunk,
        graph_database,
        neptune_analytics_entity,
        neptune_database_entity,
        storage,
        storage_entity,
    ]

    edges_data = [
        (
            str(document_chunk.id),
            str(storage_entity.id),
            "contains",
        ),
        (
            str(storage_entity.id),
            str(storage.id),
            "is_a",
        ),
        (
            str(document_chunk.id),
            str(neptune_database_entity.id),
            "contains",
        ),
        (
            str(neptune_database_entity.id),
            str(graph_database.id),
            "is_a",
        ),
        (
            str(document_chunk.id),
            str(document.id),
            "is_part_of",
        ),
        (
            str(document_chunk.id),
            str(neptune_analytics_entity.id),
            "contains",
        ),
        (
            str(neptune_analytics_entity.id),
            str(graph_database.id),
            "is_a",
        ),
    ]
    return nodes_data, edges_data


async def test_add_graph_then_vector_data():
    logger.info("------test_add_graph_then_vector_data-------")
    (nodes, edges) = setup_data()
    await na_graph.add_nodes(nodes)
    await na_graph.add_edges(edges)
    await na_vector.create_data_points(collection, nodes)

    node_ids = [str(node.id) for node in nodes]
    retrieved_data_points = await na_vector.retrieve(collection, node_ids)
    retrieved_nodes = await na_graph.get_nodes(node_ids)

    assert len(retrieved_data_points) == len(retrieved_nodes) == len(node_ids)

    # delete all nodes and edges and vectors:
    await na_graph.delete_graph()
    await na_vector.prune()

    (nodes, edges) = await na_graph.get_graph_data()
    assert len(nodes) == 0
    assert len(edges) == 0
    logger.info("------PASSED-------")


async def test_add_vector_then_node_data():
    logger.info("------test_add_vector_then_node_data-------")
    (nodes, edges) = setup_data()
    await na_vector.create_data_points(collection, nodes)
    await na_graph.add_nodes(nodes)
    await na_graph.add_edges(edges)

    node_ids = [str(node.id) for node in nodes]
    retrieved_data_points = await na_vector.retrieve(collection, node_ids)
    retrieved_nodes = await na_graph.get_nodes(node_ids)

    assert len(retrieved_data_points) == len(retrieved_nodes) == len(node_ids)

    # delete all nodes and edges and vectors:
    await na_vector.prune()
    await na_graph.delete_graph()

    (nodes, edges) = await na_graph.get_graph_data()
    assert len(nodes) == 0
    assert len(edges) == 0
    logger.info("------PASSED-------")


def main():
    """
    Example script uses neptune analytics for the graph and vector (hybrid) store with small sample data
    This example demonstrates how to add nodes and vectors to Neptune Analytics, and ensures that
    the nodes do not conflict
    """
    asyncio.run(test_add_graph_then_vector_data())
    asyncio.run(test_add_vector_then_node_data())


if __name__ == "__main__":
    main()
