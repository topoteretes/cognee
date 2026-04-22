"""Run the three tools with human_memory retrieval (rules-grounded)."""

import asyncio
import os

os.environ.setdefault("RECRUITING_WITH_MEMORY", "true")
os.environ.setdefault("RECRUITING_SESSION_ID", "recruiting-demo-grounded")

from examples.demos.recruiting_distill_memory._run import run_plan  # noqa: E402


if __name__ == "__main__":
    asyncio.run(run_plan("grounded_plan.json"))
