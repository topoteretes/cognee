import asyncio

import cognee
from cognee import SearchType

DATASET = "global_context_index_recall_demo"

FACTS = [
    "Alice hiked a new trail near Lake Como in winter.",
    "Alice reached the summit of the peak she'd trained for all year in summer.",
    "Bob sailed to a small island he had never visited in winter.",
    "Bob completed his first solo overnight crossing in summer.",
    "Alice started a sourdough starter in winter.",
    "Alice baked her first focaccia for a dinner party in summer.",
    "Bob took his first watercolor class in winter.",
    "Bob sold a painting at a local market in summer.",
    "Alice began German lessons in winter.",
    "Alice had her first full conversation in German in summer.",
]

QUERY = "What changed across all of Alice and Bob's hobbies between winter and summer?"


async def main():
    await cognee.remember(
        FACTS,
        dataset_name=DATASET,
        self_improvement=False,
    )
    await cognee.improve(dataset=DATASET, build_global_context_index=True)

    context_without = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        top_k=4,
        only_context=True,
        retriever_specific_config={"include_global_context_index": False},
    )
    context_with = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        top_k=4,
        only_context=True,
        retriever_specific_config={
            "include_global_context_index": True,
            "global_context_index_top_k": 3,
        },
    )

    print("Context WITHOUT global context index:\n")
    print(context_without[0].text)
    print("\nContext WITH global context index:\n")
    print(context_with[0].text)

    answer_without = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        top_k=4,
        retriever_specific_config={"include_global_context_index": False},
    )
    answer_with = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        top_k=4,
        retriever_specific_config={
            "include_global_context_index": True,
            "global_context_index_top_k": 3,
        },
    )

    print("\nAnswer WITHOUT global context index:\n")
    print(answer_without[0].text)
    print("\nAnswer WITH global context index:\n")
    print(answer_with[0].text)


if __name__ == "__main__":
    asyncio.run(main())
