"""Time-bounded recall over an ingested timeline.

``remember()`` extracts the dates it reads as ``Timestamp`` nodes, so
``SearchType.TEMPORAL`` can answer questions that depend on ordering — before, after,
and between a pair of dates — rather than on embedding similarity alone. ``recall()``
also routes a question with an absolute date to TEMPORAL on its own; the query type is
pinned here so every question takes that path.
"""

import asyncio

import cognee
from cognee import SearchType

TEXT = """
In 1998 the project launched. In 2001 version 1.0 shipped. In 2004 the team merged
with another group. In 2010 support for v1 ended.
"""

QUERIES = [
    "What happened before 2000?",
    "What happened after 2004?",
    "Events between 2001 and 2004",
]


async def main():
    await cognee.forget(everything=True)

    # The default extraction stores each date as a Timestamp node next to the usual graph.
    await cognee.remember(TEXT, dataset_name="timeline_demo", self_improvement=False)

    for query in QUERIES:
        results = await cognee.recall(
            query_text=query,
            query_type=SearchType.TEMPORAL,
            datasets=["timeline_demo"],
            top_k=15,
        )
        print(f"\nQ: {query}")
        print(f"A: {results[0].text if results else '(nothing found)'}")


if __name__ == "__main__":
    asyncio.run(main())
