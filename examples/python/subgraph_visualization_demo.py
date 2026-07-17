"""Demo: bounded subgraph visualization (no LLM / no live DB required).

Builds a synthetic 20-node chain graph and drives the real subgraph
selection layer plus HTML renderer:

* ``fetch_visualization_graph_data`` — seed resolution + ``get_neighborhood``
* ``cognee_network_visualization`` — existing renderer (unchanged)

Writes one HTML file per mode under ``examples/python/.artifacts/``.
"""

import asyncio
import json
import os
import re
from unittest.mock import AsyncMock, MagicMock

from cognee.modules.visualization.cognee_network_visualization import cognee_network_visualization
from cognee.modules.visualization.subgraph_data import (
    DEFAULT_MAX_NODES,
    DEFAULT_NEIGHBORHOOD_DEPTH,
    DEFAULT_SEED_TOP_K,
    fetch_visualization_graph_data,
    resolve_seeds_from_recall,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), ".artifacts", "subgraph_demo")


def _chain_graph(node_count: int = 20):
    nodes = [(str(i), {"type": "Entity", "name": f"Node{i}"}) for i in range(node_count)]
    edges = [(str(i), str(i + 1), "related_to", {}) for i in range(node_count - 1)]
    return nodes, edges


def _story_node_count(html: str) -> int:
    match = re.search(r"var nodes\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if not match:
        return 0
    return len(json.loads(match.group(1)))


def _mock_engine(full_graph):
    engine = MagicMock()
    nodes, edges = full_graph
    engine.get_graph_data = AsyncMock(return_value=full_graph)
    engine.get_neighborhood = AsyncMock(
        return_value=(nodes[8:14], edges[8:13]),
    )
    engine.query = AsyncMock(return_value=[("10",)])
    return engine


async def _render_mode(engine, label: str, **fetch_kwargs):
    graph_data, meta = await fetch_visualization_graph_data(engine, **fetch_kwargs)
    output_path = os.path.join(ARTIFACTS, f"{label}.html")
    html = await cognee_network_visualization(graph_data, output_path)
    print(
        f"{label:16} scope={meta.scope:<8} seeds={len(meta.seed_ids):<2} "
        f"source={meta.seed_source:<8} nodes={len(graph_data[0]):<3} "
        f"html_nodes={_story_node_count(html):<3} -> {output_path}"
    )


async def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    full_graph = _chain_graph(20)
    engine = _mock_engine(full_graph)

    print("Subgraph visualization caps:")
    print(f"  neighborhood_depth={DEFAULT_NEIGHBORHOOD_DEPTH}")
    print(f"  neighborhood_seed_top_k={DEFAULT_SEED_TOP_K}")
    print(f"  max_nodes={DEFAULT_MAX_NODES}")
    print()

    await _render_mode(engine, "default_degree", seed_node_ids=None)
    await _render_mode(engine, "explicit_seeds", seed_node_ids=["10", "11"])
    await _render_mode(
        engine,
        "recall_seeds",
        recall_result={"node_ids": resolve_seeds_from_recall({"node_ids": ["9", "10"]})},
    )
    await _render_mode(engine, "full_graph", full=True)

    print()
    print("SDK usage (live dataset):")
    print('  await visualize_graph("out.html")  # bounded subgraph (default)')
    print('  await visualize_graph("out.html", query="What is Python?")')
    print('  await visualize_graph("out.html", seed_node_ids=["..."])')
    print('  await visualize_graph("out.html", full=True)  # legacy full graph')
    print('  await visualize_search_subgraph(recall_results, "out.html")')


if __name__ == "__main__":
    asyncio.run(main())
