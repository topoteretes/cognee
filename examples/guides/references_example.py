"""Lightweight references (Evidence) in recall answers.

``include_references=True`` appends an Evidence section to the answer, so you can see
what the answer was built from:

- ``RAG_COMPLETION``   -> chunk evidence, from the retrieved vector payloads
- ``GRAPH_COMPLETION`` -> entity/chunk evidence, walked from the graph
- ``include_references=False`` (the default) -> the concise answer, no Evidence section
"""

import asyncio

import cognee
from cognee import SearchType

DATASET = "references_guide"

REPORT = """\
Acme Corporation 2024 Annual Report.

Acme Corporation reported total revenue of 1.2 billion dollars in 2024,
a 12 percent increase over 2023. The growth was driven primarily by the
Cloud Platform division, which expanded into the European market.

Jane Doe was appointed Chief Executive Officer of Acme Corporation in
March 2024. Under her leadership, operating margin expanded to 18 percent.

Acme Corporation is headquartered in Seattle and employs roughly 4,500 people.
"""

QUERY = "What were Acme's 2024 revenue and who is the CEO?"


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def main() -> None:
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)

    await cognee.remember(REPORT, dataset_name=DATASET, self_improvement=False)

    banner("1) RAG_COMPLETION with references -> chunk evidence")
    rag = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.RAG_COMPLETION,
        datasets=[DATASET],
        include_references=True,
    )
    print(rag[0] if rag else "<no result>")

    banner("2) GRAPH_COMPLETION with references -> graph/entity evidence")
    graph = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        include_references=True,
    )
    print(graph[0] if graph else "<no result>")

    banner("3) GRAPH_COMPLETION without references -> no Evidence section")
    plain = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        include_references=False,
    )
    print(plain[0] if plain else "<no result>")


if __name__ == "__main__":
    asyncio.run(main())
