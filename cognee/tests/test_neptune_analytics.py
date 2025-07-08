import pytest

import cognee
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.neptune_analytics.NeptuneAnalyticsAdapter import NeptuneAnalyticsAdapter, \
    IndexSchema
from cognee.shared.logging_utils import get_logger

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
    cognee.config.set_vector_db_url("neptune-graph://g-3eu7qmuf9a")
    engine = get_vector_engine()
    assert isinstance(engine, NeptuneAnalyticsAdapter)

    TEST_COLLECTION_NAME = "test"
    TEST_UUID = "78a28770-2cd5-41f8-9b65-065a34f16aff"
    TEST_TEXT = "Hello world"

    # Prun all vector_db entries
    await engine.prune()

    # No-op
    await engine.has_collection(TEST_COLLECTION_NAME)
    await engine.create_collection(TEST_COLLECTION_NAME)

    # Persist an node entry
    datapoint = IndexSchema(id=TEST_UUID, text=TEST_TEXT)
    await engine.create_data_points(TEST_COLLECTION_NAME, [datapoint])

    # Retrieve its
    result = await engine.retrieve(TEST_COLLECTION_NAME, [TEST_UUID])
    assert str(result[0].id) == TEST_UUID
    assert result[0].payload['~properties']['text'] == TEST_TEXT

    # Delete datapoint from vector store
    await engine.delete_data_points(TEST_COLLECTION_NAME, [TEST_UUID])

    # Retrieve should return an empty list.
    result_deleted = await engine.retrieve(TEST_COLLECTION_NAME, [TEST_UUID])
    assert result_deleted == []

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
