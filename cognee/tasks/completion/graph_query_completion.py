# TODO: delete after merging COG-1365, see COG-1403
import json
import logging
import os
from cognee.infrastructure.engine import ExtendableDataPoint
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.tasks.completion.exceptions import NoRelevantDataFound
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from typing import Callable


logger = logging.getLogger(__name__)


async def retrieved_edges_to_string(retrieved_edges: list) -> str:
    """
    Converts a list of retrieved graph edges into a human-readable string format.

    """
    edge_strings = []
    for edge in retrieved_edges:
        node1_string = edge.node1.attributes.get("text") or edge.node1.attributes.get("name")
        node2_string = edge.node2.attributes.get("text") or edge.node2.attributes.get("name")
        edge_string = edge.attributes["relationship_type"]
        edge_str = f"{node1_string} -- {edge_string} -- {node2_string}"
        edge_strings.append(edge_str)
    return "\n---\n".join(edge_strings)


async def graph_query_completion(
    query: str, context_resolver: Callable = None, save_context_path: str = None
) -> list:
    """
    Executes a query on the graph database and retrieves a relevant completion based on the found data.

    Parameters:
    - query (str): The query string to compute.
    - context_resolver (Callable): A function to convert retrieved edges to a string.
    - save_context_path (str): Path to save the retrieved context.

    Returns:
    - list: Answer to the query.

    Notes:
    - The `brute_force_triplet_search` is used to retrieve relevant graph data.
    - Prompts are dynamically rendered and provided to the LLM for contextual understanding.
    - Ensure that the LLM client and graph database are properly configured and accessible.
    """
    subclasses = get_all_subclasses(DataPoint)

    vector_index_collections = []

    for subclass in subclasses:
        index_fields = subclass.model_fields["metadata"].default.get("index_fields", [])
        for field_name in index_fields:
            vector_index_collections.append(f"{subclass.__name__}_{field_name}")

    found_triplets = await brute_force_triplet_search(
        query, top_k=5, collections=vector_index_collections or None
    )

    if len(found_triplets) == 0:
        raise NoRelevantDataFound

    if not context_resolver:
        context_resolver = retrieved_edges_to_string

    # Get context and optionally dump it
    context = await context_resolver(found_triplets)
    if save_context_path:
        try:
            os.makedirs(os.path.dirname(save_context_path), exist_ok=True)
            with open(save_context_path, "w") as f:
                json.dump(context, f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save context to {save_context_path}: {str(e)}")
            # Consider whether to raise or continue silently
    args = {
        "question": query,
        "context": context,
    }
    user_prompt = render_prompt("graph_context_for_question.txt", args)
    system_prompt = read_query_prompt("answer_simple_question.txt")

    llm_client = get_llm_client()
    computed_answer = await llm_client.acreate_structured_output(
        text_input=user_prompt,
        system_prompt=system_prompt,
        response_model=str,
    )

    return [computed_answer]
