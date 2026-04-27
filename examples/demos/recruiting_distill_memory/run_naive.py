"""Run the three tools without human_memory retrieval (baseline).

Reads RECRUITING_CANDIDATE (default: dev_rao) to pick which candidate
JSON to load. Writes output/naive_plan_<candidate>.json.
"""

import asyncio
import os

os.environ.setdefault("RECRUITING_WITH_MEMORY", "false")

from examples.demos.recruiting_distill_memory._run import run_plan  # noqa: E402


if __name__ == "__main__":
    asyncio.run(run_plan())
