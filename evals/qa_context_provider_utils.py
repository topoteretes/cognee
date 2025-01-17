import cognee
from cognee.api.v1.search import SearchType
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.brute_force_triplet_search import brute_force_triplet_search
from cognee.tasks.completion.graph_query_completion import retrieved_edges_to_string
from functools import partial
from cognee.api.v1.cognify.cognify_v2 import get_default_tasks
import logging

logger = logging.getLogger(__name__)


async def get_raw_context(instance: dict) -> str:
    return instance["context"]


async def cognify_instance(instance: dict, task_indices: list[int] = None):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    for title, sentences in instance["context"]:
        await cognee.add("\n".join(sentences), dataset_name="QA")
    all_cognify_tasks = await get_default_tasks()
    if task_indices:
        selected_tasks = [all_cognify_tasks[ind] for ind in task_indices]
    else:
        selected_tasks = all_cognify_tasks
    await cognee.cognify("QA", tasks=selected_tasks)


def _insight_to_string(triplet: tuple) -> str:
    if not (isinstance(triplet, tuple) and len(triplet) == 3):
        logger.warning("Invalid input: Expected a tuple of length 3.")
        return ""

    node1, edge, node2 = triplet

    if not (isinstance(node1, dict) and isinstance(edge, dict) and isinstance(node2, dict)):
        logger.warning("Invalid input: Each element in the tuple must be a dictionary.")
        return ""

    node1_name = node1["name"] if "name" in node1 else "N/A"
    node1_description = node1["description"] if "description" in node1 else node1["text"]
    node1_string = f"name: {node1_name}, description: {node1_description}"
    node2_name = node2["name"] if "name" in node2 else "N/A"
    node2_description = node2["description"] if "description" in node2 else node2["text"]
    node2_string = f"name: {node2_name}, description: {node2_description}"

    edge_string = edge.get("relationship_name", "")

    if not edge_string:
        logger.warning("Missing required field: 'relationship_name' in edge dictionary.")
        return ""

    triplet_str = f"{node1_string} -- {edge_string} -- {node2_string}"
    return triplet_str


async def get_context_with_cognee(
    instance: dict,
    task_indices: list[int] = None,
    search_types: list[SearchType] = [SearchType.SUMMARIES, SearchType.CHUNKS],
) -> str:
    await cognify_instance(instance, task_indices)

    search_results = []
    for search_type in search_types:
        raw_search_results = await cognee.search(search_type, query_text=instance["question"])

        if search_type == SearchType.INSIGHTS:
            res_list = [_insight_to_string(edge) for edge in raw_search_results]
        else:
            res_list = [
                context_item.get("text", "")
                for context_item in raw_search_results
                if isinstance(context_item, dict)
            ]
            if all(not text for text in res_list):
                logger.warning(
                    "res_list contains only empty strings: No valid 'text' entries found in raw_search_results."
                )

        search_results += res_list

    search_results_str = "\n".join(search_results)

    return search_results_str


def create_cognee_context_getter(
    task_indices=None, search_types=[SearchType.SUMMARIES, SearchType.CHUNKS]
):
    return partial(get_context_with_cognee, task_indices=task_indices, search_types=search_types)


async def get_context_with_simple_rag(instance: dict) -> str:
    await cognify_instance(instance)

    vector_engine = get_vector_engine()
    found_chunks = await vector_engine.search("document_chunk_text", instance["question"], limit=5)

    search_results_str = "\n".join([context_item.payload["text"] for context_item in found_chunks])

    return search_results_str


async def get_context_with_brute_force_triplet_search(instance: dict) -> str:
    await cognify_instance(instance)

    found_triplets = await brute_force_triplet_search(instance["question"], top_k=5)

    search_results_str = retrieved_edges_to_string(found_triplets)

    return search_results_str


valid_pipeline_slices = {
    "extract_graph": {
        "slice": [0, 1, 2, 3, 5],
        "search_types": [SearchType.INSIGHTS, SearchType.CHUNKS],
    },
    "summarize": {
        "slice": [0, 1, 2, 3, 4, 5],
        "search_types": [SearchType.INSIGHTS, SearchType.SUMMARIES, SearchType.CHUNKS],
    },
}

qa_context_providers = {
    "no_rag": get_raw_context,
    "cognee": get_context_with_cognee,
    "simple_rag": get_context_with_simple_rag,
    "brute_force": get_context_with_brute_force_triplet_search,
} | {
    name: create_cognee_context_getter(
        task_indices=value["slice"], search_types=value["search_types"]
    )
    for name, value in valid_pipeline_slices.items()
}
