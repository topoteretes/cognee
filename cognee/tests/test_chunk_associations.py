import pytest

import cognee
from cognee.shared.logging_utils import get_logger

logger = get_logger()


@pytest.mark.asyncio
async def test_chunk_associations():
    """
    Integration test for chunk associations.
    Tests: add data → cognify → memify with chunk associations
    """
    import time

    # Clean up at start
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "test_chunk_associations"

    # Add sample data with clearly related topics
    sample_texts = [
        "Python is a popular programming language created by Guido van Rossum. It is widely used for web development, data science, and automation.",
        "Python programming language supports multiple paradigms including object-oriented and functional programming. It has a large standard library.",
        "Machine learning models can be built using Python libraries like TensorFlow and PyTorch for deep learning applications.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn from data.",
    ]

    start_add = time.time()
    for text in sample_texts:
        await cognee.add(text, dataset_name)
    logger.info(f"Add data: {time.time() - start_add:.2f}s")

    # Run cognify to create the knowledge graph
    start_cognify = time.time()
    await cognee.cognify([dataset_name])
    logger.info(f"Cognify: {time.time() - start_cognify:.2f}s")

    # Run chunk associations pipeline
    from cognee.memify_pipelines.chunk_associations_pipeline import chunk_associations_pipeline
    from cognee.modules.users.methods import get_default_user

    start_memify = time.time()
    default_user = await get_default_user()
    await chunk_associations_pipeline(
        user=default_user,
        dataset=dataset_name,
        similarity_threshold=0.90,
        max_candidates_per_chunk=4,
    )
    logger.info(f"Chunk associations pipeline: {time.time() - start_memify:.2f}s")

    # Verify associations were created by checking the graph
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph_engine = await get_graph_engine()

    # Query for association edges (Kuzu stores all as Node and EDGE)
    query = """
        MATCH (a:Node)-[r:EDGE]->(b:Node)
        WHERE r.relationship_name = 'associated_with'
        RETURN count(r) as association_count
    """
    results = await graph_engine.query(query)
    association_count = results[0][0] if results else 0

    logger.info(f"Found {association_count} chunk associations in the graph")

    # Query to get details of associations
    detail_query = """
        MATCH (a:Node)-[r:EDGE]->(b:Node)
        WHERE r.relationship_name = 'associated_with'
        RETURN r
        LIMIT 5
    """
    detail_results = await graph_engine.query(detail_query)

    for result in detail_results:
        edge = result[0] if isinstance(result, (list, tuple)) else result
        logger.info(f"Association edge: {edge}")

    # Assert that at least some associations were created
    assert association_count > 0, "No chunk associations were created!"

    logger.info("Chunk associations test completed successfully")

    # Clean up at end
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_chunk_associations(), debug=True)
