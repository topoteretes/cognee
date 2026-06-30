import asyncio
import cognee


async def session_one():
    """First session — agent learns and stores research findings."""
    print("\n========== SESSION 1: Agent learns new facts ==========\n")

    # Clear any previous memory for a clean demo
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("✓ Memory cleared (fresh start)\n")

    # Agent ingests research findings into the knowledge graph
    await cognee.remember("Transformer models use self-attention to process sequences in parallel.")
    await cognee.remember("GPT is a transformer-based model trained on next-token prediction.")
    await cognee.remember("BERT is a transformer model trained using masked language modeling.")
    await cognee.remember("Self-attention allows each token to attend to every other token in the sequence.")
    print("✓ Research findings stored in knowledge graph\n")

    # Query within same session
    results = await cognee.recall("How do transformers process sequences?")
    print("Query: 'How do transformers process sequences?'")
    for r in results:
        print(f"  → {r}")


async def session_two():
    """Second session — agent picks up exactly where it left off."""
    print("\n========== SESSION 2: Agent resumes with full memory ==========\n")
    print("(No re-ingestion. No context window tricks. Just persistent memory.)\n")

    # No remember() calls here — graph already exists from session 1
    results = await cognee.recall("What is the difference between GPT and BERT?")
    print("Query: 'What is the difference between GPT and BERT?'")
    for r in results:
        print(f"  → {r}")

    print()

    results = await cognee.recall("What is self-attention?")
    print("Query: 'What is self-attention?'")
    for r in results:
        print(f"  → {r}")

    # Enrich the graph with additional connections
    await cognee.improve()
    print("\n✓ Knowledge graph enriched with improve()\n")


async def main():
    print("=== Cognee: Persistent AI Memory Demo ===")
    print("Showing why stateless LLMs fall short and how Cognee fixes it.\n")

    await session_one()
    await session_two()

    print("\n========== WHAT JUST HAPPENED ==========")
    print("Session 1: Agent learned facts → graph was built")
    print("Session 2: Agent recalled across sessions → zero context loss")
    print("This is what cognee.remember() + cognee.recall() unlocks.\n")

    # Cleanup
    await cognee.forget(dataset="main_dataset")
    print("✓ Memory cleaned up with forget()\n")


if __name__ == "__main__":
    asyncio.run(main())