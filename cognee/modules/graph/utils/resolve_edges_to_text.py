from typing import List
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


async def resolve_edges_to_text(retrieved_edges: List[Edge]) -> str:
    """
    Converts retrieved graph edges into a human-readable string format.

    Parameters:
    -----------

        - retrieved_edges (list): A list of edges retrieved from the graph.

    Returns:
    --------

        - str: A formatted string representation of the nodes and their connections.
    """

    def _get_nodes(retrieved_edges: List[Edge]) -> dict:
        def _get_title(text: str, first_n_words: int = 7, top_n_words: int = 3) -> str:
            def _top_n_words(text, stop_words=None, top_n=3, separator=", "):
                """Concatenates the top N frequent words in text."""
                if stop_words is None:
                    from cognee.modules.retrieval.utils.stop_words import DEFAULT_STOP_WORDS

                    stop_words = DEFAULT_STOP_WORDS

                import string

                words = [word.lower().strip(string.punctuation) for word in text.split()]

                if stop_words:
                    words = [word for word in words if word and word not in stop_words]

                from collections import Counter

                top_words = [word for word, freq in Counter(words).most_common(top_n)]

                return separator.join(top_words)

            """Creates a title, by combining first words with most frequent words from the text."""
            first_words = text.split()[:first_n_words]
            top_words = _top_n_words(text, top_n=first_n_words)
            return f"{' '.join(first_words)}... [{top_words}]"

        """Creates a dictionary of nodes with their names and content."""
        nodes = {}
        for edge in retrieved_edges:
            for node in (edge.node1, edge.node2):
                if node.id not in nodes:
                    text = node.attributes.get("text")
                    if text:
                        name = _get_title(text)
                        content = text
                    else:
                        name = node.attributes.get("name", "Unnamed Node")
                        content = node.attributes.get("description", name)
                    nodes[node.id] = {"node": node, "name": name, "content": content}
        return nodes

    nodes = _get_nodes(retrieved_edges)
    node_section = "\n".join(
        f"Node: {info['name']}\n__node_content_start__\n{info['content']}\n__node_content_end__\n"
        for info in nodes.values()
    )
    connection_section = "\n".join(
        f"{nodes[edge.node1.id]['name']} --[{edge.attributes['relationship_type']}]--> {nodes[edge.node2.id]['name']}"
        for edge in retrieved_edges
    )
    return f"Nodes:\n{node_section}\n\nConnections:\n{connection_section}"
