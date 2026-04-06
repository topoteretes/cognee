"""Single entry point: runs all three modes, prints comparison."""

from __future__ import annotations

import asyncio

from cognee.shared.logging_utils import ERROR, setup_logging

from . import context_impl, memory_impl, nomemory_impl
from .metrics import MetricsCollector, print_comparison


async def main() -> None:
    setup_logging(ERROR)

    # --- 1. No-memory baseline ---
    print("=" * 70)
    print("  MODE 1: NO MEMORY (baseline)")
    print("=" * 70)
    nomem = MetricsCollector()
    nomem.activate()
    await nomemory_impl.setup_nomemory()
    await nomemory_impl.run_all_leads(nomem)
    nomem.deactivate()

    # --- 2. Context stuffing ---
    print("\n" + "=" * 70)
    print("  MODE 2: CONTEXT STUFFING (all past summaries in prompt)")
    print("=" * 70)
    ctx = MetricsCollector()
    ctx.activate()
    await context_impl.setup_context()
    await context_impl.run_all_leads(ctx)
    ctx.deactivate()

    # --- 3. Full Cognee graph memory ---
    print("\n" + "=" * 70)
    print("  MODE 3: COGNEE GRAPH MEMORY (shared dataset + graph search)")
    print("=" * 70)
    mem = MetricsCollector()
    mem.activate()
    await memory_impl.setup_memory()
    await memory_impl.run_all_leads(mem)
    mem.deactivate()

    # --- Comparison ---
    print_comparison(nomem, ctx, mem)


if __name__ == "__main__":
    asyncio.run(main())
