import uuid

import pytest

import cognee
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.neptune_analytics.NeptuneAnalyticsAdapter import NeptuneAnalyticsAdapter, \
    IndexSchema
from cognee.shared.logging_utils import get_logger
import os

logger = get_logger()

async def main():
    cognee.config.set_vector_db_provider("neptune")

    # When URL is absent
    cognee.config.set_vector_db_url(None)
    with pytest.raises(OSError):
        get_vector_engine()

    # Assert invalid graph ID.
    cognee.config.set_vector_db_url("invalid_url")
    with pytest.raises(ValueError):
        get_vector_engine()

    # Return a valid engine object with valid URL.
    graph_id = os.getenv('GRAPH_ID', "")
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
        with_vector=True)
    assert (len(result_search) == 2)

    # # Retrieve data-points
    result = await engine.retrieve(TEST_COLLECTION_NAME, [TEST_UUID, TEST_UUID_2])
    assert any(
        str(r.id) == TEST_UUID and r.payload['text'] == TEST_TEXT
        for r in result
    )
    assert any(
        str(r.id) == TEST_UUID_2 and r.payload['text'] == TEST_TEXT_2
        for r in result
    )
    # Search multiple
    result_search_batch = await engine.batch_search(
        collection_name=TEST_COLLECTION_NAME,
        query_texts=[TEST_TEXT, TEST_TEXT_2],
        limit=10,
        with_vectors=False
    )
    assert (len(result_search_batch) == 2 and
            all(len(batch) == 2 for batch in result_search_batch))

    # Delete datapoint from vector store
    await engine.delete_data_points(TEST_COLLECTION_NAME, [TEST_UUID, TEST_UUID_2])

    # Retrieve should return an empty list.
    result_deleted = await engine.retrieve(TEST_COLLECTION_NAME, [TEST_UUID])
    assert result_deleted == []

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
