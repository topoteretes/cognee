"""Executable entrypoint for the job-finding agent demo."""

from __future__ import annotations

import asyncio
import json

from cognee.shared.logging_utils import INFO, setup_logging

from examples.demos.job_finding_agent.orchestrator import run_jobs_from_json


async def main() -> None:
    setup_logging(INFO)
    outcome = await run_jobs_from_json()
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
