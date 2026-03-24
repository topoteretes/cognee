"""Run email-offer stream in memory mode."""

from __future__ import annotations

import asyncio

from cognee.shared.logging_utils import ERROR, setup_logging

from memory_impl import run_stream, setup_memory


async def main() -> None:
    setup_logging(ERROR)
    await setup_memory()
    await run_stream()


if __name__ == "__main__":
    asyncio.run(main())
