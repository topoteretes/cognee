"""Group remembered texts into node sets with the node_set argument and visualize the result.

Three texts are tagged AI/FinTech, AI and MedTech; the graph with its NodeSet groupings is written
to .artifacts/nodeset_grouping.html. Then the MedTech text is moved into the AI node set with a
metadata-only update (update(data_id, None, dataset_id, node_set=[...])): its existing memory is
regrouped without an LLM call, re-chunking or re-embedding, and a CHUNKS search filtered by node
set shows it under AI and no longer under MedTech.

Requires: LLM_API_KEY.
Run: uv run python examples/guides/nodeset_grouping_example.py
"""

import asyncio
import os

import cognee
from cognee import SearchType, visualize_graph
from cognee.shared.logging_utils import ERROR, setup_logging

text_a = """
    AI is revolutionizing financial services through intelligent fraud detection
    and automated customer service platforms.
    """

text_b = """
    Advances in AI are enabling smarter systems that learn and adapt over time.
    """

text_c = """
    MedTech startups have seen significant growth in recent years, driven by innovation
    in digital health and medical devices.
    """

node_set_a = ["AI", "FinTech"]
node_set_b = ["AI"]
node_set_c = ["MedTech"]


async def main():
    await cognee.forget(everything=True)

    await cognee.remember(text_a, node_set=node_set_a, self_improvement=False)
    await cognee.remember(text_b, node_set=node_set_b, self_improvement=False)
    await cognee.remember(text_c, node_set=node_set_c, self_improvement=False)

    visualization_path = os.path.join(
        os.path.dirname(__file__), ".artifacts", "nodeset_grouping.html"
    )
    await visualize_graph(visualization_path)

    # Move the MedTech text into the AI node set. data=None replaces only the node sets
    # (node_set=[] would clear them); repeating the call answers "unchanged".
    dataset = next(d for d in await cognee.datasets.list_datasets() if d.name == "main_dataset")
    medtech = next(
        row
        for row in await cognee.datasets.list_data(dataset.id)
        if "MedTech" in (row.node_set or "")
    )
    result = await cognee.update(
        data_id=medtech.id, data=None, dataset_id=dataset.id, node_set=["AI"]
    )
    print("Reassignment:", result["status"])  # metadata_updated

    for node_set in (["AI"], ["MedTech"]):
        chunks = await cognee.search(
            "MedTech startups", query_type=SearchType.CHUNKS, node_name=node_set
        )
        found = any("MedTech" in str(chunk) for chunk in chunks)
        print(f"MedTech text found under {node_set}: {found}")


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
