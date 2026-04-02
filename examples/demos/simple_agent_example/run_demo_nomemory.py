"""Run the simple agent demo without memory."""

from __future__ import annotations

import asyncio
import os


async def main() -> None:
    os.environ["LOG_LEVEL"] = "ERROR"
    from cognee.shared.logging_utils import ERROR, setup_logging
    from nomemory_impl import run_stream, setup_nomemory

    setup_logging(ERROR)
    await setup_nomemory()
    await run_stream()


if __name__ == "__main__":
    asyncio.run(main())
