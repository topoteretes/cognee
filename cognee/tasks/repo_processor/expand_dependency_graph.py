import networkx as nx

from cognee.tasks.repo_processor.extract_code_parts import extract_code_parts
from cognee.tasks.repo_processor import logger


def _add_code_parts_nodes_and_edges(graph, parent_node_id, part_type, code_parts):
    """Add code part nodes and edges for a specific part type."""
    if not code_parts:
        logger.debug(f"No code parts to add for parent_node_id {parent_node_id} and part_type {part_type}.")
        return

    for idx, code_part in enumerate(code_parts):
        if not code_part.strip():
            logger.warning(f"Empty code part in parent_node_id {parent_node_id} and part_type {part_type}.")
            continue
        part_node_id = f"{parent_node_id}_{part_type}_{idx}"
        graph.add_node(part_node_id, source_code=code_part, node_type=part_type)
        graph.add_edge(parent_node_id, part_node_id, relation="contains")


def _process_single_node(graph, node_id, node_data):
    """Process a single Python file node."""
    graph.nodes[node_id]["node_type"] = "python_file"
    source_code = node_data.get("source_code", "")

    if not source_code.strip():
        logger.warning(f"Node {node_id} has no or empty 'source_code'. Skipping.")
        return

    try:
        code_parts_dict = extract_code_parts(source_code)
    except Exception as e:
        logger.error(f"Error processing node {node_id}: {e}")
        return

    for part_type, code_parts in code_parts_dict.items():
        _add_code_parts_nodes_and_edges(graph, node_id, part_type, code_parts)


def expand_dependency_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Process Python file nodes, adding code part nodes and edges."""
    expanded_graph = graph.copy()
    for node_id, node_data in graph.nodes(data=True):
        if not node_data:  # Check if node_data is empty
            logger.warning(f"Node {node_id} has no data. Skipping.")
            continue
        _process_single_node(expanded_graph, node_id, node_data)
    return expanded_graph
