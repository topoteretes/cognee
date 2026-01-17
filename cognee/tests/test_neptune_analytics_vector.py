import os
import pathlib
import cognee
import uuid
import pytest
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
    NeptuneAnalyticsAdapter,
    IndexSchema,
)

logger = get_logger()


async def main():
    graph_id = os.getenv("GRAPH_ID", "")
    cognee.config.set_vector_db_provider("neptune_analytics")
    cognee.config.set_vector_db_url(f"neptune-graph://{graph_id}")
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_neptune")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_neptune")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    dataset_name = "cs_explanations"

    explanation_file_path_nlp = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Natural_language_processing.txt"
    )
    await cognee.add([explanation_file_path_nlp], dataset_name)

    explanation_file_path_quantum = os.path.join(
        pathlib.Path(__file__).parent, "test_data/Quantum_computers.txt"
    )

    await cognee.add([explanation_file_path_quantum], dataset_name)

    await cognee.cognify([dataset_name])

    vector_engine = get_vector_engine()
    random_node = (
        await vector_engine.search("Entity_name", "Quantum computer", include_payload=True)
    )[0]
    random_node_name = random_node.payload["text"]

    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=random_node_name
    )
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted sentences are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(query_type=SearchType.CHUNKS, query_text=random_node_name)
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nExtracted chunks are:\n")
    for result in search_results:
        print(f"{result}\n")

    search_results = await cognee.search(
        query_type=SearchType.SUMMARIES, query_text=random_node_name
    )
    assert len(search_results) != 0, "Query related summaries don't exist."
    print("\nExtracted summaries are:\n")
    for result in search_results:
        print(f"{result}\n")

    user = await get_default_user()
    history = await get_history(user.id)
    assert len(history) == 6, "Search history is not correct."

    await cognee.prune.prune_data()
    assert not os.path.isdir(data_directory_path), "Local data files are not deleted"

    await cognee.prune.prune_system(metadata=True)


async def vector_backend_api_test():
    cognee.config.set_vector_db_provider("neptune_analytics")

    # When URL is absent
    cognee.config.set_vector_db_url(None)
    with pytest.raises(OSError):
        get_vector_engine()

    # Assert invalid graph ID.
    cognee.config.set_vector_db_url("invalid_url")
    with pytest.raises(ValueError):
        get_vector_engine()

    # Return a valid engine object with valid URL.
    graph_id = os.getenv("GRAPH_ID", "")
    cognee.config.set_vector_db_url(f"neptune-graph://{graph_id}")
    engine = get_vector_engine()
    assert isinstance(engine, NeptuneAnalyticsAdapter)

    TEST_COLLECTION_NAME = "test"
    # Data point - 1
    TEST_UUID = str(uuid.uuid4())
    TEST_TEXT = "Hello world"
    datapoint = IndexSchema(id=TEST_UUID, text=TEST_TEXT)
    # Data point - 2
    TEST_UUID_2 = str(uuid.uuid4())
    TEST_TEXT_2 = "Cognee"
    datapoint_2 = IndexSchema(id=TEST_UUID_2, text=TEST_TEXT_2)

    # Prun all vector_db entries
    await engine.prune()

    # Always return true
    has_collection = await engine.has_collection(TEST_COLLECTION_NAME)
    assert has_collection
    # No-op
    await engine.create_collection(TEST_COLLECTION_NAME, IndexSchema)

    # Save data-points
    await engine.create_data_points(TEST_COLLECTION_NAME, [datapoint, datapoint_2])
    # Search single text
    result_search = await engine.search(
        collection_name=TEST_COLLECTION_NAME,
        query_text=TEST_TEXT,
        query_vector=None,
        limit=10,
        with_vector=True,
    )
    assert len(result_search) == 2

    # # Retrieve data-points
    result = await engine.retrieve(TEST_COLLECTION_NAME, [TEST_UUID, TEST_UUID_2])
    assert any(str(r.id) == TEST_UUID and r.payload["text"] == TEST_TEXT for r in result)
    assert any(str(r.id) == TEST_UUID_2 and r.payload["text"] == TEST_TEXT_2 for r in result)
    # Search multiple
    result_search_batch = await engine.batch_search(
        collection_name=TEST_COLLECTION_NAME,
        query_texts=[TEST_TEXT, TEST_TEXT_2],
        limit=10,
        with_vectors=False,
    )
    assert len(result_search_batch) == 2 and all(len(batch) == 2 for batch in result_search_batch)

    # Delete datapoint from vector store
    await engine.delete_data_points(TEST_COLLECTION_NAME, [TEST_UUID, TEST_UUID_2])

    # Retrieve should return an empty list.
    result_deleted = await engine.retrieve(TEST_COLLECTION_NAME, [TEST_UUID])
    assert result_deleted == []


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    asyncio.run(vector_backend_api_test())
