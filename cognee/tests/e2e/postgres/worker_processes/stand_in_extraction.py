"""A stand-in for the model's graph extraction: a graph made of a text's words.

The test it serves is about processes and stores, not about the model. The same
text gives the same graph in every process, so a run in one process and a run
in several must store the same thing.
"""

import sys
from itertools import pairwise

from cognee.shared.data_models import Edge, KnowledgeGraph, Node


async def extract_content_graph(content: str, response_model, **kwargs):
    words = sorted({word.strip(".,").lower() for word in content.split() if len(word) > 3})
    nodes = [
        Node(id=word, name=word, type="Concept", description=f"the word {word}", label=word)
        for word in words
    ]
    edges = [
        Edge(source_node_id=first, target_node_id=second, relationship_name="appears_with")
        for first, second in pairwise(words)
    ]
    return KnowledgeGraph(summary="", description="", nodes=nodes, edges=edges)


def install(extract=extract_content_graph) -> None:
    """Make cognify's graph extraction call ``extract`` in this process."""
    import cognee.tasks.graph

    # By name in sys.modules: the package exports a function of the same name,
    # which is what ``import cognee.tasks.graph.extract_graph_from_data`` binds.
    sys.modules["cognee.tasks.graph.extract_graph_from_data"].extract_content_graph = extract
