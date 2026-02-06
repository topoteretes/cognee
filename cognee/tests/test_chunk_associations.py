import pathlib
import pytest
import pytest_asyncio

import cognee
from cognee.low_level import setup
from cognee import memify
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.memify.extract_subgraph_chunks import extract_subgraph_chunks
from cognee.tasks.chunks.create_chunk_associations import create_chunk_associations
from cognee.infrastructure.databases.graph import get_graph_engine


@pytest_asyncio.fixture
async def clean_test_environment():
    base_dir = pathlib.Path(__file__).parent.parent
    system_directory_path = str(base_dir / ".cognee_system/test_chunk_associations")
    data_directory_path = str(base_dir / ".data_storage/test_chunk_associations")

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


def _get_association_edges(edges):
    return [edge for edge in edges if edge[2] == "associated_with"]


@pytest.mark.asyncio
async def test_chunk_associations_creates_edges_between_similar_chunks(clean_test_environment):
    dolphin_text_1 = "River dolphins are freshwater mammals found in South America and Asia."
    dolphin_text_2 = (
        "Scientists study dolphin behavior in Amazon rivers to understand their communication."
    )
    dolphin_text_3 = "The Amazon river dolphin, also known as boto, is pink in color."
    unrelated_text = "Python is a high-level programming language widely used for data science."

    await cognee.add(dolphin_text_1)
    await cognee.add(dolphin_text_2)
    await cognee.add(dolphin_text_3)
    await cognee.add(unrelated_text)

    await cognee.cognify()

    extraction_tasks = [Task(extract_subgraph_chunks)]
    enrichment_tasks = [
        Task(
            create_chunk_associations,
            similarity_threshold=0.7,
            min_chunk_length=10,
            top_k_candidates=10,
            task_config={"batch_size": 10},
        )
    ]

    await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
    )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    association_edges = _get_association_edges(edges)

    assert len(association_edges) > 0, "Should have created at least one association edge"

    for edge in association_edges:
        props = edge[3]
        weight = props.get("weight")
        if weight:
            assert 0.0 <= weight <= 1.0, f"Weight should be between 0 and 1, got {weight}"


@pytest.mark.asyncio
async def test_chunk_associations_respects_similarity_threshold(clean_test_environment):
    related_text_1 = "Machine learning models require large datasets for training."
    related_text_2 = "Deep learning is a subset of machine learning that uses neural networks."
    very_different_text = "The quick brown fox jumps over the lazy dog."

    await cognee.add(related_text_1)
    await cognee.add(related_text_2)
    await cognee.add(very_different_text)

    await cognee.cognify()

    extraction_tasks = [Task(extract_subgraph_chunks)]
    enrichment_tasks = [
        Task(
            create_chunk_associations,
            similarity_threshold=0.9,
            min_chunk_length=10,
            task_config={"batch_size": 10},
        )
    ]

    await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
    )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    association_edges = _get_association_edges(edges)

    for edge in association_edges:
        weight = edge[3].get("weight")
        if weight:
            assert weight >= 0.9, f"All associations should meet threshold of 0.9, got {weight}"


@pytest.mark.asyncio
async def test_chunk_associations_includes_metadata(clean_test_environment):
    chunk_1 = "Artificial intelligence is transforming healthcare diagnostics."
    chunk_2 = "AI systems can now detect diseases from medical images with high accuracy."

    await cognee.add(chunk_1)
    await cognee.add(chunk_2)

    await cognee.cognify()

    extraction_tasks = [Task(extract_subgraph_chunks)]
    enrichment_tasks = [
        Task(
            create_chunk_associations,
            similarity_threshold=0.6,
            task_config={"batch_size": 10},
        )
    ]

    await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
    )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    association_edges = _get_association_edges(edges)

    if len(association_edges) > 0:
        props = association_edges[0][3]
        assert "relationship_name" in props
        assert "source_node_id" in props
        assert "target_node_id" in props
        assert "ontology_valid" in props


@pytest.mark.asyncio
async def test_chunk_associations_with_cypher_query(clean_test_environment):
    text_1 = "Climate change is affecting global weather patterns and ecosystems."
    text_2 = "Rising temperatures contribute to extreme weather events worldwide."
    text_3 = "Renewable energy sources help reduce carbon emissions."

    await cognee.add(text_1)
    await cognee.add(text_2)
    await cognee.add(text_3)

    await cognee.cognify()

    extraction_tasks = [Task(extract_subgraph_chunks)]
    enrichment_tasks = [
        Task(
            create_chunk_associations,
            similarity_threshold=0.65,
            task_config={"batch_size": 10},
        )
    ]

    await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
    )

    search_results = await cognee.search(
        query_text="""
            MATCH (c1:Node)-[a:EDGE]->(c2:Node)
            WHERE c1.type = 'DocumentChunk' AND c2.type = 'DocumentChunk' AND a.relationship_name = 'associated_with'
            RETURN c1.properties, c2.properties, a.properties
            LIMIT 10
        """,
        query_type=cognee.SearchType.CYPHER,
    )

    assert search_results is not None


@pytest.mark.asyncio
async def test_chunk_associations_handles_single_chunk(clean_test_environment):
    single_text = "This is a single document chunk with no pairs to compare."

    await cognee.add(single_text)
    await cognee.cognify()

    extraction_tasks = [Task(extract_subgraph_chunks)]
    enrichment_tasks = [
        Task(
            create_chunk_associations,
            similarity_threshold=0.7,
            task_config={"batch_size": 10},
        )
    ]

    await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
    )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    association_edges = _get_association_edges(edges)

    assert len(association_edges) == 0, "Should not create associations with only one chunk"


@pytest.mark.asyncio
async def test_chunk_associations_configurable_parameters(clean_test_environment):
    text_1 = "Short"
    text_2 = "This is a longer text that should be processed based on min_chunk_length parameter."

    await cognee.add(text_1)
    await cognee.add(text_2)
    await cognee.cognify()

    extraction_tasks = [Task(extract_subgraph_chunks)]
    enrichment_tasks = [
        Task(
            create_chunk_associations,
            similarity_threshold=0.5,
            min_chunk_length=15,
            top_k_candidates=5,
            task_config={"batch_size": 10},
        )
    ]

    await memify(
        extraction_tasks=extraction_tasks,
        enrichment_tasks=enrichment_tasks,
    )

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    association_edges = _get_association_edges(edges)

    assert len(association_edges) == 0, (
        "Short text below min_chunk_length should not produce associations"
    )
