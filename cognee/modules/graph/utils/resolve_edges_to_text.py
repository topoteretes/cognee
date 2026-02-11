import string
from typing import List
from collections import Counter
from cognee.shared.logging_utils import get_logger

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS

logger = get_logger(__name__) # namespace to help user filter through logs

def _get_top_n_frequent_words(
    text: str, stop_words: set = None, top_n: int = 3, separator: str = ", "
) -> str:
    """Concatenates the top N frequent words in text."""
    if stop_words is None:
        stop_words = DEFAULT_STOP_WORDS

    words = [word.lower().strip(string.punctuation) for word in text.split()]
    words = [word for word in words if word and word not in stop_words]

    top_words = [word for word, freq in Counter(words).most_common(top_n)]
    return separator.join(top_words)


def _create_title_from_text(text: str, first_n_words: int = 7, top_n_words: int = 3) -> str:
    """Creates a title by combining first words with most frequent words from the text."""
    first_words = text.split()[:first_n_words]
    top_words = _get_top_n_frequent_words(text, top_n=top_n_words)
    return f"{' '.join(first_words)}... [{top_words}]"


def _extract_nodes_from_edges(retrieved_edges: List[Edge]) -> dict:
    """Creates a dictionary of nodes with their names and content."""
    nodes = {}

    for edge in retrieved_edges:
        for node in (edge.node1, edge.node2):
            if node.id in nodes:
                logger.debug("Skipped duplicate edge node: %s", node.id)
                continue

            text = node.attributes.get("text")
            if text:
                name = _create_title_from_text(text)
                content = text
            else:
                name = node.attributes.get("name", "Unnamed Node")
                content = node.attributes.get("description", name)
                logger.debug("No text found for node: %s, setting as Unamed Node", node.id)

            nodes[node.id] = {"node": node, "name": name, "content": content}

    return nodes


async def resolve_edges_to_text(retrieved_edges: List[Edge]) -> str:
    """Converts retrieved graph edges into a human-readable string format."""
    nodes = _extract_nodes_from_edges(retrieved_edges)
    logger.info("Resolving %d edges (%d unique nodes) to text format", len(retrieved_edges), len(nodes))

    node_section = "\n".join(
        f"Node: {info['name']}\n__node_content_start__\n{info['content']}\n__node_content_end__\n"
        for info in nodes.values()
    )

    connections = []
    for edge in retrieved_edges:
        #Checking if node1 and/or node2 exist
        if edge.node1.id not in nodes or edge.node2.id not in nodes:
            missing_id = edge.node1.id if edge.node1.id not in nodes else edge.node2.id
            logger.warning("Edge resolution failed: Node '%s' not found in extraction.",missing_id)
            continue
        source_name = nodes[edge.node1.id]["name"]
        target_name = nodes[edge.node2.id]["name"]
        edge_text = edge.attributes.get("edge_text")
        edge_relationship_type = edge.attributes.get("relationship_type")
        edge_label = edge_text or edge_relationship_type
        if edge_label is None:
            logger.warning("Edge is missing both 'edge_text' and 'relationship_type'. Using default 'related_to' label.")
            edge_label = "related_to"
        connections.append(f"{source_name} --[{edge_label}]--> {target_name}")
        logger.debug("Resolved edge: %s --[%s]--> %s", source_name, edge_label, target_name)

    connection_section = "\n".join(connections)
    logger.debug("Successfully resolved %d edges and %d connections", len(nodes), len(connections))
    return f"Nodes:\n{node_section}\n\nConnections:\n{connection_section}"
