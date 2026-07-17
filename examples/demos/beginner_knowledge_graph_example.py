"""
Beginner-friendly Cognee example.

This script shows the smallest possible end-to-end workflow:
  1. Store a few sentences in memory (Cognee builds a knowledge graph from them)
  2. Ask a question and see the answer come back from the graph
  3. Inspect the graph directly so you can see WHAT got created, not just the answer

Run with:
    python beginner_knowledge_graph_example.py

Prerequisites:
    pip install cognee
    Set an LLM_API_KEY in a .env file (see the Installation guide for provider options)
"""

import asyncio
import cognee


async def main():
    # 1. Start from a clean slate.
    # This wipes any previous Cognee data so the example is reproducible.
    await cognee.forget(everything=True)

    # 2. Store a small, human-readable story in memory.
    # Cognee will read this text, break it into entities (people, places, things)
    # and relationships (who did what, who is connected to whom), and store
    # that structure as a knowledge graph -- not just as searchable text.
    text = (
        "Alice is a software engineer who works at Cognee. "
        "Cognee is a company that builds AI memory systems. "
        "Alice's manager is Bob, who leads the engineering team."
    )

    print("Storing the following text in memory:\n")
    print(text, "\n")

    await cognee.remember(text)

    # 3. Ask a question. Cognee automatically decides how to search the graph
    # to answer it -- you don't need to specify a search strategy.
    question = "Who does Alice work for, and who is her manager?"
    print(f"Question: {question}\n")

    answer = await cognee.recall(query_text=question)

    print("Answer:")
    for result in answer:
        print(f"  - {result.text}")

    # 4. See what the graph actually looks like under the hood.
    # `only_context=True` skips the final LLM answer and instead returns the
    # raw retrieved graph context -- this is useful for understanding HOW
    # Cognee arrived at its answer.
    print("\nRaw graph context used to answer the question:")
    context = await cognee.recall(query_text=question, only_context=True)
    for item in context:
        print(f"  - {item.text}")


if __name__ == "__main__":
    asyncio.run(main())