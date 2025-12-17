# cognee/examples/usage_frequency_example.py
import asyncio
import cognee
from cognee.api.v1.search import SearchType
from cognee.tasks.memify.extract_usage_frequency import usage_frequency_pipeline_entry

async def main():
    # Reset cognee state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Sample conversation
    conversation = [
        "Alice discusses machine learning",
        "Bob asks about neural networks",
        "Alice explains deep learning concepts",
        "Bob wants more details about neural networks"
    ]

    # Add conversation and cognify
    await cognee.add(conversation)
    await cognee.cognify()

    # Perform some searches to generate interactions
    for query in ["machine learning", "neural networks", "deep learning"]:
        await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text=query,
            save_interaction=True
        )

    # Run usage frequency tracking
    await cognee.memify(
        *usage_frequency_pipeline_entry(cognee.graph_adapter)
    )

    # Search and display frequency weights
    results = await cognee.search(
        query_text="Find nodes with frequency weights",
        query_type=SearchType.NODE_PROPERTIES,
        properties=["frequency_weight"]
    )

    print("Nodes with Frequency Weights:")
    for result in results[0]["search_result"][0]:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())