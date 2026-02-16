import asyncio
import cognee

import os
from os import path

from cognee import visualize_graph
from cognee.infrastructure.databases.graph import get_graph_engine
from poc_single_add_datapoints_pipeline import poc_cognify
# By default cognee uses OpenAI's gpt-5-mini LLM model
# Provide your OpenAI LLM API KEY


def _edge_key(edge_tuple):
    return (str(edge_tuple[0]), str(edge_tuple[1]), str(edge_tuple[2]))


async def _get_graph_snapshot(label: str):
    graph_engine = await get_graph_engine()
    nodes_data, edges_data = await graph_engine.get_graph_data()

    node_ids = {str(node_id) for node_id, _ in nodes_data}
    edge_keys = {_edge_key(edge) for edge in edges_data}
    node_info_by_id = {str(node_id): node_info for node_id, node_info in nodes_data}
    node_labels = {node_info.get("name") for node_info in node_info_by_id.values() if node_info}

    return {
        "label": label,
        "node_ids": node_ids,
        "edge_keys": edge_keys,
        "node_info_by_id": node_info_by_id,
        "node_labels": node_labels,
        "node_count": len(node_ids),
        "edge_count": len(edge_keys),
    }


def _diff_graph_snapshots(base, other):
    missing_nodes = sorted(set(base["node_labels"]) - set(other["node_labels"]))
    missing_edges = sorted(base["edge_keys"] - other["edge_keys"])

    print("")
    print(f"Graph diff: {base['label']} -> {other['label']}")
    print(f"Nodes: {base['node_count']} -> {other['node_count']}")
    print(f"Edges: {base['edge_count']} -> {other['edge_count']}")
    print(f"Missing nodes in {other['label']}: {len(missing_nodes)}")
    print(f"Missing edges in {other['label']}: {len(missing_edges)}")

    if missing_nodes:
        print("Sample missing node labels (first 20):")
        for node_label in missing_nodes[:20]:
            print(f"  {node_label}")

    if missing_edges:
        print("Sample missing edges (first 20):")
        for source, target, relation in missing_edges[:20]:
            print(f"  {source} -[{relation}]-> {target}")


async def main(use_poc):
    # Get file path to document to process
    from pathlib import Path

    current_directory = Path(__file__).resolve().parent
    file_path = os.path.join(current_directory, "data", "alice_in_wonderland.txt")

    graph_visualization_path = path.join(
        path.dirname(__file__), f"results/{'poc_' if use_poc else ''}simple_example_result.html"
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Call Cognee to process document
    await cognee.add(file_path)

    if use_poc:
        await poc_cognify(use_single_add_datapoints_poc=True)
    else:
        await cognee.cognify()

    await visualize_graph(graph_visualization_path)
    return await _get_graph_snapshot("poc" if use_poc else "default")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        default_snapshot = loop.run_until_complete(main(use_poc=False))
        poc_snapshot = loop.run_until_complete(main(use_poc=True))
        _diff_graph_snapshots(default_snapshot, poc_snapshot)
        print("POC")
        _diff_graph_snapshots(poc_snapshot, default_snapshot)

    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
