"""Greenhouse connector demo — give cognee a memory of your hiring funnel.

Pull Greenhouse jobs, job posts, and (opt-in) interview scorecards into cognee
with forget-on-delete. ``greenhouse_source`` returns a ``dlt`` source you hand
straight to ``cognee.remember`` — no routing kwargs needed. Records are ingested
as normal documents, so they go through the full cognify entity-extraction
pipeline.

Each run is a full snapshot: unchanged records keep a stable id and are not
re-cognified, and anything deleted in Greenhouse (or removed by GDPR
anonymization) drops out of the snapshot, so cognee's orphan cleanup forgets it
from memory on the next sync.

────────────────────────────────────────────────────────────────────────────
Privacy / opt-in
────────────────────────────────────────────────────────────────────────────
Scorecards are candidate feedback. They are NOT sync by default: pass
``include_scorecards=True`` only if you really need them, and use
``scorecard_application_ids=[...]`` to scope which applications' scorecards
leave Greenhouse. The connector never writes to Greenhouse and never renders
private notes or candidate contact details.

────────────────────────────────────────────────────────────────────────────
One-time setup
────────────────────────────────────────────────────────────────────────────
1. Install the extra:

       pip install "cognee[greenhouse]"   # or: uv sync --extra greenhouse

2. In Greenhouse (Configure > Dev Center > API Credential Management) create
   "Harvest V3 (OAuth)" credentials. Harvest v1/v2 are retired — do not use an
   API key. These credentials are scopes, e.g. jobs:read, job_posts:read,
   scorecards:read.

3. Export the client id/secret and your LLM key, then run:

       export GREENHOUSE_CLIENT_ID="..."
       export GREENHOUSE_CLIENT_SECRET="..."
       export LLM_API_KEY="sk-..."
       uv run python examples/demos/greenhouse_connector_example.py
"""

import asyncio
import os

import cognee

from cognee_community_connector_greenhouse import greenhouse_source

DATASET_NAME = "greenhouse"


async def main() -> None:
    if not (os.environ.get("GREENHOUSE_CLIENT_ID") and os.environ.get("GREENHOUSE_CLIENT_SECRET")):
        print(
            "Set GREENHOUSE_CLIENT_ID and GREENHOUSE_CLIENT_SECRET "
            "(Harvest v3 OAuth credentials) to run this example."
        )
        return

    # Jobs and job posts sync by default. Add include_scorecards=True only if
    # you explicitly want interview scorecards in memory.
    source = greenhouse_source()

    print("Syncing Greenhouse into cognee ...")
    await cognee.remember(source, dataset_name=DATASET_NAME)

    answer = await cognee.search(
        query_text="What roles is the company hiring for, and what do they involve?",
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        datasets=[DATASET_NAME],
    )
    print("\nSearch result:\n", answer)

    print(
        "\nEdit or close a job in Greenhouse, then re-run: edits re-sync and "
        "deleted/removed records are reconciled out of memory."
    )


if __name__ == "__main__":
    asyncio.run(main())
