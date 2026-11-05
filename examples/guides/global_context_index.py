"""Build the global context index with improve(build_global_context_index=True), then extend it.

Six facts are remembered and indexed; the bucket summaries and root summary are printed, then one
more fact is remembered and improve() runs again. The index itself makes no LLM call: a bucket is
one document's top entities by tf-idf plus the relations among them, and the root is the dataset's
most widespread names. Re-running improve() over unchanged data rewrites nothing.

Requires: LLM_API_KEY for the cognify step that extracts the entities (not for the index itself).
Run: uv run python examples/guides/global_context_index.py
"""

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

ADDITIONAL_FACT = (
    "This year, Bob decided to join Alice's hiking trip to the Alps instead of sailing."
)


async def print_index_structure(label):
    """Print the index and return its texts, so two runs can be compared."""
    graph_engine = await get_graph_engine()
    nodes_data, _edges_data = await graph_engine.get_graph_data()

    root = None
    buckets = []

    for node_id, properties in nodes_data:
        if properties.get("type") != "GlobalContextSummary":
            continue
        if properties.get("is_root"):
            root = (node_id, properties.get("text", ""))
        else:
            buckets.append((node_id, properties.get("text", "")))

    print(f"\n{label}")
    print(f"Buckets: {len(buckets)}")
    for bucket_id, text in buckets:
        print(f"  - [{bucket_id}]\n{indent(text)}")
    if root:
        print(f"Root [{root[0]}]:\n{indent(root[1])}")

    return sorted(text for _, text in buckets), root[1] if root else None


def indent(text):
    return "\n".join(f"      {line}" for line in text.splitlines())


async def main():
    await cognee.remember(
        INITIAL_FACTS,
        dataset_name=DATASET,
        self_improvement=False,
    )
    await cognee.improve(dataset=DATASET, build_global_context_index=True)
    first = await print_index_structure("Index structure after the initial build:")

    # Nothing changed in between, so this run skips the build: the stage reports
    # already_completed and the index stays as it is.
    await cognee.improve(dataset=DATASET, build_global_context_index=True)
    second = await print_index_structure("Index structure after a second improve():")
    print(f"\nSecond run produced identical texts: {first == second}")

    await cognee.remember(
        ADDITIONAL_FACT,
        dataset_name=DATASET,
        self_improvement=False,
    )
    await cognee.improve(dataset=DATASET, build_global_context_index=True)
    await print_index_structure("Index structure after adding one more fact:")


if __name__ == "__main__":
    asyncio.run(main())
