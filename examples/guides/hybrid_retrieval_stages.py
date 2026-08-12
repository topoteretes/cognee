"""See the three stages inside hybrid retrieval: retrieved objects, context, completion.

Most searches go straight from a query to an answer. This guide drives ``HybridRetriever``
manually so each stage is visible: what got retrieved (chunks, entities, facts), the context
assembled from it, and the completion generated from that context. Useful when you want to
understand or debug what the LLM actually saw.
"""

import asyncio

import cognee
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever

DATASET = "hybrid_retrieval_stages"

DOCUMENTS = [
    "Northstar Labs runs the Berlin office and the Lisbon office; each owns one project.",
    "The Berlin office owns RoutePulse, a project that predicts delivery delays for freight.",
    "The Lisbon office owns HarborLens, a project that monitors port congestion.",
    "RoutePulse uses traffic feeds, weather alerts, and customs delay reports.",
    "HarborLens uses vessel schedules, berth availability, and labor notices.",
]

QUERY = "Which office owns HarborLens, and what signals does HarborLens use?"


async def main():
    try:
        await cognee.forget(dataset=DATASET)
    except ValueError:
        pass  # First run — the dataset does not exist yet.

    await cognee.remember(DOCUMENTS, dataset_name=DATASET, self_improvement=False)

    retriever = HybridRetriever(chunks_top_k=4, entities_top_k=4, max_edges_per_entity=4)

    # Stage 1 — retrieval: raw objects fetched for the query.
    retrieved_objects = await retriever.get_retrieved_objects(query=QUERY)
    print("STAGE 1 — RETRIEVED OBJECTS")
    for chunk in retrieved_objects.get("chunks", []):
        payload = getattr(chunk, "payload", None) or {}
        print(f"  chunk: {str(payload.get('text', ''))[:120]}")
    for entity in retrieved_objects.get("entities", []):
        print(f"  entity: {entity.get('name') or entity.get('id')}")
    for fact in retrieved_objects.get("facts", []):
        print(f"  fact: {str(fact.get('text', ''))[:120]}")

    # Stage 2 — context: the text block assembled from those objects.
    context = await retriever.get_context_from_objects(
        query=QUERY, retrieved_objects=retrieved_objects
    )
    print("\nSTAGE 2 — CONTEXT")
    print(context or "[empty context]")

    # Stage 3 — completion: the answer generated from that context.
    completion = await retriever.get_completion_from_context(
        query=QUERY, retrieved_objects=retrieved_objects, context=context
    )
    print("\nSTAGE 3 — COMPLETION")
    for item in completion:
        print(item)


if __name__ == "__main__":
    asyncio.run(main())
