"""Cite the memories a recall used, and read one of them back by its tag.

`result_mode="cited"` returns the HYBRID_COMPLETION `only_context` prompt with a tag line,
`[mem:<kind>/<dataset_id>/<record_id>@<version>]`, before every passage, entity and fact, and
lists the tagged records in `metadata["records"]`. A model given that prompt can name the
memory it used; `expand_items` then reads the named record by id, with no query.

Requires: LLM_API_KEY.
Run: uv run python examples/guides/cited_recall.py
"""

import asyncio

import cognee
from cognee import SearchType

DATASET = "cited_recall_guide"
QUERY = "What did Alice and Bob work on together?"


async def main():
    # Start from an empty dataset on every run.
    if any(dataset.name == DATASET for dataset in await cognee.datasets.list_datasets()):
        await cognee.forget(dataset=DATASET)

    await cognee.remember(
        [
            "Alice and Bob were PhD students in Berlin from 2021 to 2024.",
            "Alice and Bob worked on a paper together in 2023.",
            "Alice joined Cognee as a backend engineer in 2025.",
        ],
        dataset_name=DATASET,
        self_improvement=False,
    )

    # One item per dataset searched: without `datasets`, every dataset you can read.
    (cited,) = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[DATASET],
        only_context=True,
        result_mode="cited",
    )
    # The prompt an LLM would receive, each memory behind its tag.
    print(cited.text)

    # One record per tag, in prompt order: pick by kind and rank, fit a budget by chars.
    records = cited.metadata["records"]
    for record in records:
        print(record["record_kind"], record["rank"], record["chars"], record["tag"])

    # Later, with nothing but the tags (a model quoted them, a user asked to see one):
    # the query is ignored, so it is empty.
    tags = [record["tag"] for record in records[:2]]
    for row in await cognee.recall("", expand_items=tags):
        # status is "available", "missing" (no longer stored) or "forbidden". A body that
        # changed since it was cited comes back with another version.
        print(row.metadata["status"], row.metadata["version"], row.metadata["sources"])
        print(row.text)


if __name__ == "__main__":
    asyncio.run(main())
