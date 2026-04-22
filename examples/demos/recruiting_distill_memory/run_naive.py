"""Run the three tools without human_memory retrieval (baseline)."""

import asyncio
import os

os.environ.setdefault("RECRUITING_WITH_MEMORY", "false")
os.environ.setdefault("RECRUITING_SESSION_ID", "recruiting-demo-naive")

from examples.demos.recruiting_distill_memory._run import run_plan  # noqa: E402


if __name__ == "__main__":
    asyncio.run(run_plan("naive_plan.json"))
