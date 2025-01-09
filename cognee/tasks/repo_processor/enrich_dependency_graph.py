import networkx as nx
from typing import AsyncGenerator, Dict, List
from tqdm.asyncio import tqdm

from cognee.infrastructure.engine import DataPoint
from cognee.shared.CodeGraphEntities import CodeFile
from cognee.modules.graph.utils import get_graph_from_model, convert_node_to_data_point
from cognee.infrastructure.databases.graph import get_graph_engine


def topologically_sort_subgraph(
    subgraph_node_to_indegree: Dict[str, int], graph: nx.DiGraph
) -> List[str]:
    """Performs a topological sort on a subgraph based on node indegrees."""
    results = []
    remaining_nodes = subgraph_node_to_indegree.copy()

    while remaining_nodes:
        next_node = min(remaining_nodes, key=remaining_nodes.get)
        results.append(next_node)

        for successor in graph.successors(next_node):
            if successor in remaining_nodes:
                remaining_nodes[successor] -= 1

        remaining_nodes.pop(next_node)

    return results


def topologically_sort(graph: nx.DiGraph) -> List[str]:
    """Performs a topological sort on the entire graph."""
    subgraphs = (graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph))

    topological_order = []

    for subgraph in subgraphs:
        node_to_indegree = {node: len(list(subgraph.successors(node))) for node in subgraph.nodes}
        topological_order.extend(topologically_sort_subgraph(node_to_indegree, subgraph))

    return topological_order


async def node_enrich_and_connect(
    graph: nx.MultiDiGraph,
    topological_order: List[str],
    node: CodeFile,
    data_points_map: Dict[str, DataPoint],
) -> None:
    """Adds 'depends_on' edges to the graph based on topological order."""
    topological_rank = topological_order.index(node.id)
    node.topological_rank = topological_rank
    node_descendants = nx.descendants(graph, node.id)

    if graph.has_edge(node.id, node.id):
        node_descendants.add(node.id)

    new_connections = []
    graph_engine = await get_graph_engine()

    for desc_id in node_descendants:
        if desc_id not in topological_order[: topological_rank + 1]:
            continue

        desc = None

        if desc_id in data_points_map:
            desc = data_points_map[desc_id]
        else:
            node_data = await graph_engine.extract_node(str(desc_id))
            try:
                desc = convert_node_to_data_point(node_data)
            except Exception:
                pass

        if desc is not None:
            new_connections.append(desc)

    node.depends_directly_on = node.depends_directly_on or []
    node.depends_directly_on.extend(new_connections)


async def enrich_dependency_graph(
    data_points: list[DataPoint],
) -> AsyncGenerator[list[DataPoint], None]:
    """Enriches the graph with topological ranks and 'depends_on' edges."""
    nodes = []
    edges = []
    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    for data_point in data_points:
        graph_nodes, graph_edges = await get_graph_from_model(
            data_point,
            added_nodes=added_nodes,
            added_edges=added_edges,
            visited_properties=visited_properties,
        )
        nodes.extend(graph_nodes)
        edges.extend(graph_edges)

    graph = nx.MultiDiGraph()

    simple_nodes = [(node.id, node.model_dump()) for node in nodes]

    graph.add_nodes_from(simple_nodes)

    graph.add_edges_from(edges)

    topological_order = topologically_sort(graph)

    node_rank_map = {node: idx for idx, node in enumerate(topological_order)}

    # for node_id, node in tqdm(graph.nodes(data = True), desc = "Enriching dependency graph", unit = "node"):
    #     if node_id not in node_rank_map:
    #         continue

    #     data_points.append(node_enrich_and_connect(graph, topological_order, node))

    data_points_map = {data_point.id: data_point for data_point in data_points}
    # data_points_futures = []

    for data_point in tqdm(data_points, desc="Enriching dependency graph", unit="data_point"):
        if data_point.id not in node_rank_map:
            continue

        if isinstance(data_point, CodeFile):
            # data_points_futures.append(node_enrich_and_connect(graph, topological_order, data_point, data_points_map))
            await node_enrich_and_connect(graph, topological_order, data_point, data_points_map)

        yield data_point

    # await asyncio.gather(*data_points_futures)

    # return data_points
