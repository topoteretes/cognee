import networkx as nx
from typing import Dict, List


def topologically_sort_subgraph(subgraph_node_to_indegree: Dict[str, int], graph: nx.DiGraph) -> List[str]:
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
        node_to_indegree = {
            node: len(list(subgraph.successors(node)))
            for node in subgraph.nodes
        }
        topological_order.extend(
            topologically_sort_subgraph(node_to_indegree, subgraph)
        )
    return topological_order


def node_enrich_and_connect(graph: nx.MultiDiGraph, topological_order: List[str], node: str) -> None:
    """Adds 'depends_on' edges to the graph based on topological order."""
    topological_rank = topological_order.index(node)
    graph.nodes[node]['topological_rank'] = topological_rank
    node_descendants = nx.descendants(graph, node)
    if graph.has_edge(node,node):
        node_descendants.add(node)
    for desc in node_descendants:
        if desc not in topological_order[:topological_rank+1]:
            continue
        graph.add_edge(node, desc, relation='depends_on')


async def enrich_dependency_graph(graph: nx.DiGraph) -> nx.MultiDiGraph:
    """Enriches the graph with topological ranks and 'depends_on' edges."""
    graph = nx.MultiDiGraph(graph)
    topological_order = topologically_sort(graph)
    node_rank_map = {node: idx for idx, node in enumerate(topological_order)}
    for node in graph.nodes:
        if node not in node_rank_map:
            continue
        node_enrich_and_connect(graph, topological_order, node)
    return graph
