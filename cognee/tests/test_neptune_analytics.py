import pytest

import cognee
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.neptune_analytics.NeptuneAnalyticsAdapter import NeptuneAnalyticsAdapter
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.shared.logging_utils import get_logger

logger = get_logger()

async def main():
    # cognee.config.set_vector_db_provider("neptune")
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

    await engine.has_collection("test_new")


    # await engine.prune()

    # await engine.delete_data_points("test", ["Bob", "Carol"])


    # await engine.create_data_points("test", [DataPoint(k="value")])
    # result = await engine.retrieve("test", ["556c794c-0f3b-44d8-aeec-4a882f300749",
    #                                "556c794c-0f3b-44d8-aeec-4a882f300749",
    #                                "556c794c-0f3b-44d8-aeec-4a882f300749"])

    print(result)



if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
