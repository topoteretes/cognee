"""Sync PubMed articles matching a query into cognee memory and ask about them.

Setup:
    pip install "cognee-connector-pubmed[cognee]"
    export LLM_API_KEY=...            # cognee's LLM provider key

Optional: an NCBI API key (https://www.ncbi.nlm.nih.gov/account/settings/) raises the
E-utilities rate limit from 3 to 10 requests a second:

    export NCBI_API_KEY=...

Run:
    python examples/example.py ["your pubmed query"]
"""

import asyncio
import os
import sys

import cognee

from cognee_connector_pubmed import pubmed_source

QUERY = sys.argv[1] if len(sys.argv) > 1 else "memory consolidation[title/abstract]"
DATASET = "pubmed_example"


async def main() -> None:
    await cognee.remember(
        pubmed_source(
            api_key=os.environ.get("NCBI_API_KEY", ""),
            resource_name="pubmed_example",
            query=QUERY,
            # Only what entered the index in the last year; drop the argument to reach
            # back as far as the query matches.
            start_date="2025/11/16",
        ),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",  # required: incremental upserts and deletes
        max_rows_per_table=0,  # no row cap; orphan cleanup must see the whole corpus
    )

    answer = await cognee.recall(
        f"What does the recent literature say about: {QUERY}?",
        datasets=[DATASET],
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
