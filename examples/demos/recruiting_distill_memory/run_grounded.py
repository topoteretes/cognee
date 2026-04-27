"""Run the three tools with human_memory retrieval (rules-grounded).

Reads RECRUITING_CANDIDATE (default: dev_rao) to pick which candidate
JSON to load. Writes output/grounded_plan_<candidate>.json and invokes
the decorator's memify pipeline to cognify session traces into
`agent_proposed_rule` nodes in human_memory.
"""

import asyncio
import os

os.environ.setdefault("RECRUITING_WITH_MEMORY", "true")

from examples.demos.recruiting_distill_memory._run import run_plan  # noqa: E402


if __name__ == "__main__":
    asyncio.run(run_plan())
