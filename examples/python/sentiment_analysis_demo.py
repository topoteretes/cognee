import cognee
import asyncio
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.tasks.sentiment_analysis.sentiment_analysis import run_sentiment_analysis

async def main():
    # Resetting cognee data
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # Adding sample content
    text = "Cognee turns documents into AI memory."
    await cognee.add(text)

    # Process with LLMs to build the knowledge graph
    print("Cognifying the content...")
    await cognee.cognify()
    print("Cognify complete.\n")

    # List of queries to test
    queries = [
        "What does Cognee do?",
        "How does Cognee store data?",
        "Are you even listening to what I am asking?"
    ]

    all_results = {}

    for q in queries:
        results = await cognee.search(
            query_text=q,
            save_interaction=True,  # Save interactions for analysis
        )
        all_results[q] = results  # Store results by query

    
    sentiment_data_points = await run_sentiment_analysis()
    print(sentiment_data_points)

if __name__ == '__main__':
    asyncio.run(main())
