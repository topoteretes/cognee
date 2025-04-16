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


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
