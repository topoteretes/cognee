"""Run the simple agent demo with memory enabled."""

from __future__ import annotations

import asyncio
import os


async def main() -> None:
    os.environ["LOG_LEVEL"] = "ERROR"
    from cognee.shared.logging_utils import ERROR, setup_logging
    from memory_impl import run_stream, setup_memory

    setup_logging(ERROR)
    await setup_memory()
    await run_stream()


if __name__ == "__main__":
    asyncio.run(main())
