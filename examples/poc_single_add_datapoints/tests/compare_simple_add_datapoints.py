"""
Run non-POC and POC add_data_points with the same KG, query graph DB for nodes/edges, compare.
"""

import asyncio
from examples.poc_single_add_datapoints.tests.kg_from_text import get_demo_kg, run_with_kg
from utils import _get_graph_snapshot, _compare


async def run_non_poc(kg) -> tuple[set, set]:
    """Prune, run_with_kg(use_poc=False), return normalized nodes and edges."""
    await run_with_kg(kg, use_poc=False)
    return await _get_graph_snapshot()


async def run_poc(kg) -> tuple[set, set]:
    """Prune, run_with_kg(use_poc=True), return normalized nodes and edges."""
    await run_with_kg(kg, use_poc=True)
    return await _get_graph_snapshot()


async def compare_kg_from_text_runs():
    kg = await get_demo_kg()

    print("Running non-POC...")
    non_poc_nodes, non_poc_edges = await run_non_poc(kg)
    print(f"  non-POC: {len(non_poc_nodes)} nodes, {len(non_poc_edges)} edges")

    print("Running POC...")
    poc_nodes, poc_edges = await run_poc(kg)
    print(f"  POC: {len(poc_nodes)} nodes, {len(poc_edges)} edges")

    _compare("non-POC nodes", non_poc_nodes, "POC nodes", poc_nodes)
    _compare("non-POC edges", non_poc_edges, "POC edges", poc_edges)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(compare_kg_from_text_runs())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
