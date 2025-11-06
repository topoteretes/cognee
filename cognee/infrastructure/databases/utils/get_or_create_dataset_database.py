from __future__ import annotations

from uuid import UUID
from typing import Union

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from cognee.modules.data.methods import create_dataset

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.utils.constants import (
    DEFAULT_GRAPH_DB_PROVIDER,
    DEFAULT_VECTOR_DB_NAME,
    GRAPH_DB_EXTENSIONS,
)
from cognee.modules.data.methods import get_unique_dataset_id
from cognee.modules.users.models import DatasetDatabase
from cognee.modules.users.models import User


async def get_or_create_dataset_database(
    dataset: Union[str, UUID],
    user: User,
    graph_provider: str | None = None,
) -> DatasetDatabase:
    """
    Return the `DatasetDatabase` row for the given owner + dataset.

    • If the row already exists, it is fetched and returned.
    • Otherwise a new one is created atomically and returned.

    Parameters
    ----------
    user : User
        Principal that owns this dataset.
    dataset : Union[str, UUID]
        Dataset being linked.
    """
    db_engine = get_relational_engine()

    dataset_id = await get_unique_dataset_id(dataset, user)

    vector_db_name = f"{dataset_id}.{DEFAULT_VECTOR_DB_NAME}"
    provider = _resolve_graph_provider(graph_provider)
    graph_extension = GRAPH_DB_EXTENSIONS.get(
        provider, GRAPH_DB_EXTENSIONS[DEFAULT_GRAPH_DB_PROVIDER]
    )
    graph_db_name = f"{dataset_id}{graph_extension}"

    async with db_engine.get_async_session() as session:
        # Create dataset if it doesn't exist
        if isinstance(dataset, str):
            dataset = await create_dataset(dataset, user, session)

        # Try to fetch an existing row first
        stmt = select(DatasetDatabase).where(
            DatasetDatabase.owner_id == user.id,
            DatasetDatabase.dataset_id == dataset_id,
        )
        existing: DatasetDatabase = await session.scalar(stmt)
        if existing:
            return existing

        # If there are no existing rows build a new row
        record = DatasetDatabase(
            owner_id=user.id,
            dataset_id=dataset_id,
            vector_database_name=vector_db_name,
            graph_database_name=graph_db_name,
        )

        try:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

        except IntegrityError:
            await session.rollback()
            raise


def _resolve_graph_provider(provider: str | None) -> str:
    from cognee.infrastructure.databases.utils.constants import (
        DEFAULT_GRAPH_DB_PROVIDER,
        GRAPH_DBS_WITH_MULTI_USER_SUPPORT,
    )

    normalized = (provider or DEFAULT_GRAPH_DB_PROVIDER).lower()
    if normalized not in GRAPH_DBS_WITH_MULTI_USER_SUPPORT:
        return DEFAULT_GRAPH_DB_PROVIDER
    return normalized
