"""Teach retrieval a preference: truth-subspace reranking through the public API.

Learnings from a finished session (here: the user cares about coffee, not tea) are distilled
into a truth subspace by ``improve(build_truth_subspace=True)``; at query time the hybrid
retriever nudges ranking toward them. This guide runs the same ambiguous query twice — truth
weighting off, then on — and prints both retrieval contexts so the coffee chunks visibly rise.

For the mechanics underneath (centroid slots, epochs, rebuilds) see
``examples/advanced_guides/truth_centroid_slots_demo.py``.
"""

import asyncio

import cognee
from cognee import SearchType

DATASET = "truth_subspace_guide"

CORPUS = [
    "Espresso is brewed by forcing hot water through finely ground coffee under high pressure.",
    "A pour-over coffee drips a slow stream of hot water over a paper filter of ground coffee.",
    "Cold brew coffee steeps coarse coffee grounds in cold water for twelve hours or more.",
    "Green tea is brewed with water below boiling to avoid a bitter, astringent flavor.",
    "Black tea is steeped in fully boiling water for three to five minutes before serving.",
    "Matcha is a powdered green tea whisked into hot water with a bamboo whisk until frothy.",
]

# What a finished session learned about the user. build_truth_subspace reads its anchor
# lessons from the "session_learnings" node set.
LESSONS = [
    "The user is a dedicated coffee drinker who cares about espresso and pour-over technique.",
    "We learned the user wants coffee recommendations specifically, and is not interested in tea.",
]

QUERY = "How should I prepare my morning drink at home?"


async def ranked_context(use_truth_weight: bool):
    results = await cognee.search(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[DATASET],
        node_name=["beverages"],  # rank only the corpus, not the lesson chunks
        only_context=True,
        retriever_specific_config={
            "chunks_top_k": len(CORPUS),
            "entities_top_k": 0,  # focus on chunk-lane reranking
            "facts_top_k": 0,
            "use_truth_weight": use_truth_weight,
        },
    )
    return results[0] if results else "[no context]"


async def main():
    try:
        await cognee.forget(dataset=DATASET)
    except ValueError:
        pass  # First run — the dataset does not exist yet.

    await cognee.remember(
        CORPUS, dataset_name=DATASET, node_set=["beverages"], self_improvement=False
    )

    print(f"QUERY: {QUERY}")
    print("\nBASELINE CONTEXT (truth weighting off)")
    print(await ranked_context(use_truth_weight=False))

    # Record the session learnings, then distill them into the truth subspace.
    await cognee.remember(
        LESSONS, dataset_name=DATASET, node_set=["session_learnings"], self_improvement=False
    )
    await cognee.improve(dataset=DATASET, build_truth_subspace=True)

    print("\nTRUTH-WEIGHTED CONTEXT (truth weighting on)")
    print(await ranked_context(use_truth_weight=True))
    print("\nThe learned coffee preference reshapes the retrieval ordering.")


if __name__ == "__main__":
    asyncio.run(main())
