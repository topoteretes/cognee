"""Run the simple agent demo without memory."""

from __future__ import annotations

import asyncio

from cognee.shared.logging_utils import ERROR, setup_logging

from nomemory_impl import run_stream, setup_nomemory


async def main() -> None:
    setup_logging(ERROR)
    await setup_nomemory()
    await run_stream()


if __name__ == "__main__":
    asyncio.run(main())
