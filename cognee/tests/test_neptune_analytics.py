import pytest

import cognee
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.neptune_analytics.NeptuneAnalyticsAdapter import NeptuneAnalyticsAdapter
from cognee.shared.logging_utils import get_logger

logger = get_logger()

async def main():
    cognee.config.set_vector_db_provider("neptune")
    #
    # # When URL is absent
    # cognee.config.set_vector_db_url(None)
    # with pytest.raises(OSError):
    #     get_vector_engine()
    #
    # # Assert invalid graph ID.
    # cognee.config.set_vector_db_url("invalid_url")
    # with pytest.raises(ValueError):
    #     get_vector_engine()

    # Return a valid engine object with valid URL.
    cognee.config.set_vector_db_url("neptune-graph://g-kd394cozz4")
    engine = get_vector_engine()
    assert isinstance(engine, NeptuneAnalyticsAdapter)

    # await engine.prune()

    # await engine.delete_data_points("test", ["Bob", "Carol"])
    # await engine.has_collection("test_new")

    await engine.create_data_points("test", ["Bob", "Carol"])



if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
