"""Sync recent cs.AI papers from arXiv into cognee memory.

Needs cognee configured with an LLM_API_KEY (and an embedding provider), like any
other cognee run. The arXiv API is public: no key, no consent, nothing to connect.

Run from the connector's directory after
    pip install "cognee-connector-arxiv[cognee]":
    python examples/arxiv_papers.py

The first run ingests the selection (here: cs.AI papers submitted in 2024, a small
window so the demo finishes in minutes at arXiv's one-request-per-3-seconds rate);
re-running the same script syncs only what was submitted since the previous run.
"""

import asyncio

import cognee

from cognee_connector_arxiv import arxiv_source


async def main():
    await cognee.remember(
        arxiv_source(
            categories=["cs.AI"],
            resource_name="arxiv_cs_ai",
            submitted_after="2024-01-01",
        ),
        dataset_name="arxiv",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
        self_improvement=False,
    )

    answer = await cognee.recall("What are the most recent cs.AI papers about?", datasets=["arxiv"])
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
