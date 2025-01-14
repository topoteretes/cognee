import cognee
from cognee.api.v1.search import SearchType
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.brute_force_triplet_search import brute_force_triplet_search
from cognee.tasks.completion.graph_query_completion import retrieved_edges_to_string


async def get_context_without_rag(instance: dict) -> str:
    return instance["context"]


async def cognify_instance(instance: dict):
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    for title, sentences in instance["context"]:
        await cognee.add("\n".join(sentences), dataset_name="QA")
    await cognee.cognify("QA")


async def get_context_with_cognee(instance: dict) -> str:
    await cognify_instance(instance)

    search_results = await cognee.search(SearchType.INSIGHTS, query_text=instance["question"])
    search_results_second = await cognee.search(
        SearchType.SUMMARIES, query_text=instance["question"]
    )
    search_results = search_results + search_results_second

    search_results_str = "\n".join([context_item["text"] for context_item in search_results])

    return search_results_str


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


qa_context_providers = {
    "no_rag": get_context_without_rag,
    "cognee": get_context_with_cognee,
    "simple_rag": get_context_with_simple_rag,
    "brute_force": get_context_with_brute_force_triplet_search,
}
