"""First-use provisioning of a dataset's databases must survive concurrent callers (SDK-603).

A query on a brand-new dataset can enter the dataset context at the same moment
the first ingestion (or another query) does. Every caller must end up with the
one registry row, and none of them may fail.
"""

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.utils.get_or_create_dataset_database import (
    get_or_create_dataset_database,
)
from cognee.modules.data.methods import create_dataset
from cognee.modules.data.models import Dataset
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import DatasetDatabase


@pytest.mark.asyncio
async def test_concurrent_first_use_provisioning_yields_one_row_and_no_errors():
    user = await get_default_user()
    engine = get_relational_engine()

    async with engine.get_async_session() as session:
        dataset = await create_dataset(f"sdk603_race_{uuid4().hex[:8]}", user, session)
        await session.commit()
        dataset_id = dataset.id

    try:
        rows = await asyncio.gather(
            *(get_or_create_dataset_database(dataset_id, user) for _ in range(4))
        )

        assert {row.dataset_id for row in rows} == {dataset_id}
        assert {row.graph_database_name for row in rows} == {rows[0].graph_database_name}

        async with engine.get_async_session() as session:
            stored = (
                await session.scalars(
                    select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
                )
            ).all()
        assert len(stored) == 1
    finally:
        async with engine.get_async_session() as session:
            await session.execute(
                delete(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
            )
            await session.execute(delete(Dataset).where(Dataset.id == dataset_id))
            await session.commit()
