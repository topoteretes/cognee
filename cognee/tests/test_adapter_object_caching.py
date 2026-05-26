import asyncio

import cognee
from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
from cognee.modules.search.types import SearchType


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "simple_add_cognify_search"
    text = "Berlin is the capital of Germany."

    await cognee.add(data=text, dataset_name=dataset_name)
    await cognee.cognify(datasets=[dataset_name])

    results = await cognee.search(
        query_text="What is the capital of Germany?",
        query_type=SearchType.CHUNKS,
        datasets=[dataset_name],
        only_context=True,
    )

    graph_cache_size = _create_graph_engine.cache_info().currsize
    vector_cache_size = _create_vector_engine.cache_info().currsize

    assert graph_cache_size <= 2, f"Graph engine cache size too large: {graph_cache_size}"
    assert vector_cache_size <= 2, f"Vector engine cache size too large: {vector_cache_size}"

    print(results)

    datasets = await cognee.datasets.list_datasets()
    dataset = next(d for d in datasets if d.name == dataset_name)
    await cognee.datasets.empty_dataset(dataset_id=dataset.id)

    graph_cache_size = _create_graph_engine.cache_info().currsize
    vector_cache_size = _create_vector_engine.cache_info().currsize

    assert graph_cache_size <= 2, (
        f"Graph engine cache size too large after delete: {graph_cache_size}"
    )
    assert vector_cache_size <= 2, (
        f"Vector engine cache size too large after delete: {vector_cache_size}"
    )


if __name__ == "__main__":
    asyncio.run(main())
