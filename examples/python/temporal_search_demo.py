"""
Temporal Search Demo for Cognee

This example demonstrates how to use Cognee to ingest timestamped event data
and retrieve context using time-bounded temporal search (SearchType.TEMPORAL).
"""

import asyncio
from cognee import cognee, SearchType, prune


async def main():
    print("🧹 Cleaning up previous Cognee memory state...")
    await prune.prune_data()
    await prune.prune_system(metadata=True)

    # 1. Define sample event data with clear temporal context
    event_data = [
        "On 2026-06-01, the engineering team launched the new hybrid graph-vector engine.",
        "On 2026-06-15, Cognee reached 25,000 GitHub stars and released v1.0.",
        "On 2026-07-01, the WeMakeDevs AI hackathon officially kicked off with 5,000 developers.",
        "On 2026-07-05, the team submitted their final pull requests for the context evaluation.",
    ]

    print("📥 Ingesting temporal event data into Cognee...")
    for text in event_data:
        await cognee.add(text, dataset_name="hackathon_timeline")

    # 2. Cognify the dataset to build the graph and extract temporal relationships
    print("🧠 Cognifying data (building knowledge graph and embeddings)...")
    await cognee.cognify(dataset_name="hackathon_timeline")

    # 3. Perform a Temporal Search for events within a specific timeframe
    query = "What major events or launches happened between June and July 2026?"
    print(f"\n🔍 Executing TEMPORAL Search: '{query}'")
    
    results = await cognee.search(
        query_text=query,
        query_type=SearchType.TEMPORAL,
    )

    print("\n📊 Temporal Search Results:")
    for idx, result in enumerate(results, 1):
        print(f"  {idx}. {result}")

    print("\n✅ Temporal demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
