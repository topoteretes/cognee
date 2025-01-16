import cognee
from cognee.api.v1.search import SearchType
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.brute_force_triplet_search import brute_force_triplet_search
from cognee.tasks.completion.graph_query_completion import retrieved_edges_to_string
from functools import partial
from cognee.api.v1.cognify.cognify_v2 import get_default_tasks


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


async def get_context_with_cognee(
    instance: dict,
    task_indices: list[int] = None,
    search_types: list[SearchType] = [SearchType.INSIGHTS, SearchType.SUMMARIES, SearchType.CHUNKS],
) -> str:
    await cognify_instance(instance, task_indices)

    search_results = []
    for search_type in search_types:
        search_results += await cognee.search(search_type, query_text=instance["question"])

    search_results_str = "\n".join([context_item["text"] for context_item in search_results])

    return search_results_str


def create_cognee_context_getter(task_indices=None, search_types=[SearchType.CHUNKS]):
    return partial(get_context_with_cognee, task_indices=task_indices, search_types=search_types)


async def get_context_with_simple_rag(instance: dict) -> str:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    for title, sentences in instance["context"]:
        await cognee.add("\n".join(sentences), dataset_name="QA")

    vector_engine = get_vector_engine()
    found_chunks = await vector_engine.search("document_chunk_text", instance["question"], limit=5)

    search_results_str = "\n".join([context_item.payload["text"] for context_item in found_chunks])

    return search_results_str


async def get_context_with_brute_force_triplet_search(instance: dict) -> str:
    await cognify_instance(instance)

    found_triplets = await brute_force_triplet_search(instance["question"], top_k=5)

    search_results_str = retrieved_edges_to_string(found_triplets)

    return search_results_str


qa_context_providers = {
    "no_rag": get_raw_context,
    "cognee": get_context_with_cognee,
    "simple_rag": get_context_with_simple_rag,
    "brute_force": get_context_with_brute_force_triplet_search,
}
