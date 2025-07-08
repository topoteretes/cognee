import pytest

import cognee
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.neptune_analytics.NeptuneAnalyticsAdapter import NeptuneAnalyticsAdapter
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
    cognee.config.set_vector_db_url("neptune-graph://g-12345678910")
    engine = get_vector_engine()
    assert isinstance(engine, NeptuneAnalyticsAdapter)



if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
