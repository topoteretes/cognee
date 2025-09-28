import cognee
import asyncio
from cognee.tasks.sentiment_analysis.sentiment_analysis import run_sentiment_analysis
from cognee.modules.users.models import User  # If you want to get the logged-in user

async def main():

    # Create a clean slate for cognee -- reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    # Add sample content
    text = "Cognee turns documents into AI memory."
    await cognee.add(text)
    
    # Process with LLMs to build the knowledge graph
    await cognee.cognify()

    # List of queries
    queries = [
        "What does Cognee do?",
        "How does Cognee store data?"
        # "Explain Cognee's search functionality."
    ]

    all_results = {}

    for q in queries:
        results = await cognee.search(
            query_text=q,
            save_interaction=True,
        )
        all_results[q] = results  # store results by query

        # Print sentiment if needed (dummy previous Q&A for now)
        # prev_question = "Previous question placeholder"
        # prev_answer = "Previous answer placeholder"
        # sentiment = await run_sentiment_analysis(prev_question, prev_answer, q, user_id="demo-user-123")
        # print(f"Sentiment for '{q}': {sentiment}")

    # Print results
    # for query, res in all_results.items():
    #     print(f"\nQuery: {query}")
    #     for r in res:
    #         print(f"  - {r}")

if __name__ == '__main__':
    asyncio.run(main())



# results = await run_sentiment_analysis(
    #     prev_question="What is the warranty for Hp laptops?",
    #     prev_answer="Dell offers 3-year warranty.",
    #     current_question="I didnt asked for dell.",
    #     user=User,
    # )
    # print(results)