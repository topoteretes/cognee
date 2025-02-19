from cognee.tests.unit.interfaces.graph.get_graph_from_model_test import (
    Document,
    DocumentChunk,
    Entity,
    EntityType,
)
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
import json
from pathlib import Path


async def create_disconnected_test_graph():
    doc = Document(path="test/path")
    doc_chunk = DocumentChunk(part_of=doc, text="This is a chunk of text", contains=[])
    entity_type = EntityType(name="Person")
    entity = Entity(name="Alice", is_type=entity_type)
    entity2 = Entity(name="Alice2", is_type=entity_type)
    doc_chunk.contains.extend([entity, entity2])

    doc2 = Document(path="test/path2")
    doc_chunk2 = DocumentChunk(part_of=doc2, text="This is a chunk of text", contains=[])
    entity_type2 = EntityType(name="Person")
    entity3 = Entity(name="Bob", is_type=entity_type2)
    doc_chunk2.contains.extend([entity3])

    await add_data_points([doc_chunk, doc_chunk2])


async def create_connected_test_graph():
    doc = Document(path="test/path")
    doc_chunk = DocumentChunk(part_of=doc, text="This is a chunk of text", contains=[])
    entity_type = EntityType(name="Person")
    entity = Entity(name="Alice", is_type=entity_type)
    entity2 = Entity(name="Alice2", is_type=entity_type)
    # the following self-loop is intentional and serves the purpose of testing the self-loop counting functionality
    doc_chunk.contains.extend([entity, entity2, doc_chunk])

    await add_data_points([doc_chunk])


async def get_metrics(provider: str, include_optional=True):
    create_graph_engine.cache_clear()
    cognee.config.set_graph_database_provider(provider)
    graph_engine = await get_graph_engine()
    await graph_engine.delete_graph()
    if include_optional:
        await create_connected_test_graph()
    else:
        await create_disconnected_test_graph()
    graph_metrics = await graph_engine.get_graph_metrics(include_optional=include_optional)
    return graph_metrics


async def assert_metrics(provider, include_optional=True):
    metrics = await get_metrics(provider=provider, include_optional=include_optional)

    gt_path = Path(__file__).parent / "ground_truth_metrics.json"
    with open(gt_path, "r") as file:
        ground_truth_metrics = json.load(file)

    if include_optional:
        ground_truth_metrics = ground_truth_metrics["connected"]
    else:
        ground_truth_metrics = ground_truth_metrics["disconnected"]

    diff_keys = set(metrics.keys()).symmetric_difference(set(ground_truth_metrics.keys()))
    if diff_keys:
        raise AssertionError(f"Metrics dictionaries have different keys: {diff_keys}")

    for key, ground_truth_value in ground_truth_metrics.items():
        assert metrics[key] == ground_truth_value, (
            f"Expected {ground_truth_value} for '{key}' with {provider}, got {metrics[key]}"
        )
