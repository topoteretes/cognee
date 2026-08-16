"""Large-scale migration release test — Phase 1 (runs on the LEGACY version).

Seeds a LARGE production-shaped system on the pinned legacy cognee (>= v1.2.0),
mirroring the manual 1.5.0 release validation: the war-and-peace large mock
(~100k nodes / ~100k semantic edges per dataset) is ingested into TWO datasets
with identical content. On the legacy version that dedups into ONE shared Data
row with two dataset memberships — exactly the shape the dataset-scoping
backfill later splits into a keeper and a fork, forcing the fork dataset
through the full ``rekey_fork_document_ids`` graph re-key at 100k scale.

Backend access control stays at its default (ON): per-dataset isolated stores
are what force the fork dataset through the full graph re-key rather than the
shared-store provenance-only path.

All AI calls are replayed from the mock file (see
``cognee/tests/utils/mock_ingestion``): zero LLM/embedding API calls, fully
deterministic corpus. The rest of the pipeline is the legacy version's real
add + cognify.

This script is SAVED before the legacy checkout (the module does not exist on
old tags) and run with its own directory on sys.path — the ``mock_ingestion``
package is copied next to it by the workflow. Verification and counting live
entirely in phase 2, which runs on the current branch and its APIs.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_ingestion import ingest_mock  # noqa: E402

DATASET_A = "large_fork_a"
DATASET_B = "large_fork_b"

MEMORIES_FILE = Path(os.environ["LARGE_MEMORIES_FILE"])
MOCK_FILE = Path(os.environ["LARGE_MOCK_FILE"])


async def main():
    t0 = time.time()
    timings = await ingest_mock(
        MEMORIES_FILE,
        MOCK_FILE,
        [DATASET_A, DATASET_B],
        prune_first=True,
    )
    print(f"phase1: ingest timings {timings}", flush=True)
    print(f"phase1: completed in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
