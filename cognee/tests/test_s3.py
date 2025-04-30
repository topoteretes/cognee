import logging

import cognee
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from collections import Counter


logger = get_logger()


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    s3_input = "s3://samples3input"
    await cognee.add(s3_input)

    await cognee.cognify()

    graph_engine = await get_graph_engine()
    graph = await graph_engine.get_graph()

    type_counts = Counter(
        node_data["type"] for _, node_data in graph.nodes(data=True) if "type" in node_data
    )

    edge_type_counts = Counter(edge_type for _, _, edge_type in graph.edges(keys=True))

    logging.info(type_counts)

    # Assert there is exactly one PdfDocument.
    assert type_counts.get("PdfDocument", 0) == 1, (
        f"Expected exactly one PdfDocument, but found {type_counts.get('PdfDocument', 0)}"
    )

    # Assert there is exactly one TextDocument.
    assert type_counts.get("TextDocument", 0) == 1, (
        f"Expected exactly one TextDocument, but found {type_counts.get('TextDocument', 0)}"
    )

    # Assert there are at least two DocumentChunk nodes.
    assert type_counts.get("DocumentChunk", 0) >= 2, (
        f"Expected at least two DocumentChunk nodes, but found {type_counts.get('DocumentChunk', 0)}"
    )

    # Assert there is at least two TextSummary.
    assert type_counts.get("TextSummary", 0) >= 2, (
        f"Expected at least two TextSummary, but found {type_counts.get('TextSummary', 0)}"
    )

    # Assert there is at least one Entity.
    assert type_counts.get("Entity", 0) > 0, (
        f"Expected more than zero Entity nodes, but found {type_counts.get('Entity', 0)}"
    )

    # Assert there is at least one EntityType.
    assert type_counts.get("EntityType", 0) > 0, (
        f"Expected more than zero EntityType nodes, but found {type_counts.get('EntityType', 0)}"
    )

    # Assert that there are at least two 'is_part_of' edges.
    assert edge_type_counts.get("is_part_of", 0) >= 2, (
        f"Expected at least two 'is_part_of' edges, but found {edge_type_counts.get('is_part_of', 0)}"
    )

    # Assert that there are at least two 'made_from' edges.
    assert edge_type_counts.get("made_from", 0) >= 2, (
        f"Expected at least two 'made_from' edges, but found {edge_type_counts.get('made_from', 0)}"
    )

    # Assert that there is at least one 'is_a' edge.
    assert edge_type_counts.get("is_a", 0) >= 1, (
        f"Expected at least one 'is_a' edge, but found {edge_type_counts.get('is_a', 0)}"
    )

    # Assert that there is at least one 'contains' edge.
    assert edge_type_counts.get("contains", 0) >= 1, (
        f"Expected at least one 'contains' edge, but found {edge_type_counts.get('contains', 0)}"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
