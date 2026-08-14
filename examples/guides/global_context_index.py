import asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine

DATASET = "global_context_index_demo"

INITIAL_FACTS = [
    "Alice hikes in the Alps every summer.",
    "Bob sails along the Adriatic coast every summer.",
    "Alice says hiking helps her disconnect from work.",
    "Bob says sailing is the best way to unwind after a busy winter.",
    "Last year Alice hiked a new trail near Lake Como.",
    "Last year Bob sailed to a small island he had never visited.",
]

ADDITIONAL_FACT = "This year, Bob decided to join Alice's hiking trip to the Alps instead of sailing."


async def print_index_structure(label):
    graph_engine = await get_graph_engine()
    nodes_data, _edges_data = await graph_engine.get_graph_data()

    root = None
    buckets = []
    text_summary_count = 0

    for node_id, properties in nodes_data:
        node_type = properties.get("type")
        if node_type == "TextSummary":
            text_summary_count += 1
        elif node_type == "GlobalContextSummary":
            if properties.get("is_root"):
                root = (node_id, properties.get("text", ""))
            else:
                buckets.append((node_id, properties.get("text", "")))

    print(f"\n{label}")
    print(f"Source summaries: {text_summary_count} TextSummary nodes")
    print(f"Buckets: {len(buckets)}")
    for bucket_id, text in buckets:
        print(f"  - [{bucket_id}] {text[:60]}...")
    if root:
        print(f"Root [{root[0]}]: {root[1][:100]}...")


async def main():
    await cognee.remember(
        INITIAL_FACTS,
        dataset_name=DATASET,
        self_improvement=False,
    )
    await cognee.improve(dataset=DATASET, build_global_context_index=True)
    await print_index_structure("Index structure after the initial build:")

    await cognee.remember(
        ADDITIONAL_FACT,
        dataset_name=DATASET,
        self_improvement=False,
    )
    await cognee.improve(dataset=DATASET, build_global_context_index=True)
    await print_index_structure("Index structure after adding one more fact:")


if __name__ == "__main__":
    asyncio.run(main())
