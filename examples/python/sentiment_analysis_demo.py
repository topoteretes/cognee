import cognee
import asyncio

async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    text = "Cognee turns documents into AI memory."

    await cognee.add(text)
    await cognee.cognify()

    queries = [
        "What does Cognee do?",
        "How does Cognee store data?"
    ]

    all_results = {}

    for q in queries:
        results = await cognee.search(
            query_text=q,
            save_interaction=True,
        )
        all_results[q] = results

    # Print results
    for query, res in all_results.items():
        print(f"\nQuery: {query}")
        for r in res:
            print(f"  - {r}")

if __name__ == '__main__':
    asyncio.run(main())