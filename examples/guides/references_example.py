"""Lightweight references (Evidence) in recall answers.

``include_references=True`` appends an Evidence section to the answer text itself, so you
can see which chunks the answer is grounded in. Each bullet cites a document name, a chunk
number, and a snippet. Off (the default), you get the concise answer alone.

The Evidence is **answer-grounded**: candidates are filtered and ranked by term overlap
with the generated answer, so the bullets show where the answer came from rather than
whatever retrieval happened to return.

Every completion search type supports the flag, and the Evidence block looks the same in
each — only the candidate pool differs. ``RAG_COMPLETION`` cites the chunks it already
retrieved; ``GRAPH_COMPLETION`` retrieves triplets rather than chunks, so it re-queries the
chunk index with the answer text to find them. On a corpus this small both arrive at the
same chunks, which is why this guide shows one search type rather than comparing two.

Two caveats. Evidence is only added to plain-string answers — pass a ``response_model``
and it is skipped rather than corrupting the structured output. And a backend failure
degrades to no Evidence, so a missing block does not by itself prove the flag was off.
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

    banner("WITHOUT references -> the answer alone")
    plain_results = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        include_references=False,
    )
    print(plain_results[0].text)

    # `text` holds the answer, with the Evidence block appended to it.
    banner("WITH references -> the same answer, plus an Evidence block")
    referenced_results = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        include_references=True,
    )
    print(referenced_results[0].text)


if __name__ == "__main__":
    asyncio.run(main())
