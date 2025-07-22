import os
import asyncio
import cognee
import pathlib

from cognee.infrastructure.databases.graph import get_graph_engine
from collections import Counter
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def test_edge_ingestion():
    """
    Tests whether we ingest additional entity to entity edges
    """

    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_edge_ingestion")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_edge_ingestion")
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    basic_nested_edges = ["is_a", "is_part_of", "contains", "made_from"]

    entity_to_entity_edges = ["likes", "prefers", "watches"]

    text1 = "Dave watches Dexter Resurrection"
    text2 = "Ana likes apples"
    text3 = "Bob prefers Cognee over other solutions"

    await cognee.add([text1, text2, text3], dataset_name="edge_ingestion_test")

    user = await get_default_user()

    await cognee.cognify(["edge_ingestion_test"], user=user)

    graph_engine = await get_graph_engine()
    graph = await graph_engine.get_graph_data()

    edge_type_counts = Counter(edge_type[2] for edge_type in graph[1])

    "Tests the presence of basic nested edges"
    for basic_nested_edge in basic_nested_edges:
        assert edge_type_counts.get(basic_nested_edge, 0) >= 1, (
            f"Expected at least one {basic_nested_edge} edge, but found {edge_type_counts.get(basic_nested_edge, 0)}"
        )

    "Tests the presence of additional entity to entity edges"
    assert len(edge_type_counts) > 4, (
        f"Expected at least {5} edges (4 structural plus entity to entity edges), but found only {len(edge_type_counts)}"
    )

    "Tests the consistency of basic nested edges"
    assert edge_type_counts.get("made_from", 0) == edge_type_counts.get("is_part_of", 0), (
        f"Number of made_from and is_part_of edges are not matching, found {edge_type_counts.get('made_from', 0)} made from and {edge_type_counts.get('is_part_of', 0)} is_part_of."
    )

    "Tests whether we generate is_a for all entity that is contained by a chunk"
    assert edge_type_counts.get("contains", 0) == edge_type_counts.get("is_a", 0), (
        f"Number of contains and is_a edges are not matching, found {edge_type_counts.get('is_a', 0)} is_a and {edge_type_counts.get('is_part_of', 0)} contains."
    )

    found_edges = 0
    for entity_to_entity_edge in entity_to_entity_edges:
        if entity_to_entity_edge in edge_type_counts:
            found_edges = found_edges + 1

    "Tests the presence of extected entity to entity edges"
    assert found_edges >= 2, (
        f"Expected at least 2 entity to entity edges, but found only {found_edges}"
    )


if __name__ == "__main__":
    asyncio.run(test_edge_ingestion())
