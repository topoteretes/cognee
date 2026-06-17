import os
from uuid import UUID
from typing import Union, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from cognee.modules.data.methods import create_dataset
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.modules.data.methods import get_unique_dataset_id
from cognee.modules.users.models import DatasetDatabase
from cognee.modules.users.models import User
from cognee.version import get_cognee_version
from cognee.modules.migrations.migration import head_revision
from cognee.modules.migrations.registry import MIGRATIONS


async def _get_vector_db_info(dataset_id: UUID, user: User) -> dict:
    vector_config = get_vectordb_config()

    from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
        supported_dataset_database_handlers,
    )

    handler = supported_dataset_database_handlers[vector_config.vector_dataset_database_handler]
    return await handler["handler_instance"].create_dataset(dataset_id, user)


async def _get_graph_db_info(dataset_id: UUID, user: User) -> dict:
    graph_config = get_graph_config()

    from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
        supported_dataset_database_handlers,
    )

    handler = supported_dataset_database_handlers[graph_config.graph_dataset_database_handler]
    return await handler["handler_instance"].create_dataset(dataset_id, user)


async def _existing_dataset_database(
    dataset_id: UUID,
    user: User,
) -> Optional[DatasetDatabase]:
    """
    Check if a DatasetDatabase row already exists for the given owner + dataset.
    Return None if it doesn't exist, return the row if it does.
    Args:
        dataset_id:
        user:

    Returns:
        DatasetDatabase or None
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        stmt = select(DatasetDatabase).where(
            DatasetDatabase.owner_id == user.id,
            DatasetDatabase.dataset_id == dataset_id,
        )
        existing: DatasetDatabase = await session.scalar(stmt)
        return existing


async def get_or_create_dataset_database(
    dataset: Union[str, UUID],
    user: User,
) -> DatasetDatabase:
    """
    Return the `DatasetDatabase` row for the given owner + dataset.

    • If the row already exists, it is fetched and returned.
    • Otherwise a new one is created atomically and returned.

    DatasetDatabase row contains connection and provider info for vector and graph databases.

    Parameters
    ----------
    user : User
        Principal that owns this dataset.
    dataset : Union[str, UUID]
        Dataset being linked.
    """
    db_engine = get_relational_engine()

    dataset_id = await get_unique_dataset_id(dataset, user)

    # If dataset is given as name make sure the dataset is created first
    if isinstance(dataset, str):
        from cognee.modules.data.methods import create_authorized_dataset

        async with db_engine.get_async_session() as session:
            if isinstance(dataset, str):
                dataset = await create_authorized_dataset(dataset, user)

    # If dataset database already exists return it
    existing_dataset_database = await _existing_dataset_database(dataset_id, user)
    if existing_dataset_database:
        return existing_dataset_database

    graph_config_dict = await _get_graph_db_info(dataset_id, user)
    vector_config_dict = await _get_vector_db_info(dataset_id, user)

    async with db_engine.get_async_session() as session:
        # If there are no existing rows build a new row. A freshly created
        # database is stamped at the current migration head so it skips all
        # existing migrations; cognee_version is recorded for audit only.
        #
        # Assumption: creating the row coincides with creating an EMPTY graph/
        # vector DB, so head-stamping is correct. If a physical DB can outlive
        # its dataset_database row and be re-attached (e.g. Neo4j CREATE DATABASE
        # IF NOT EXISTS), this row would wrongly skip migrations on populated
        # data — handle that case explicitly if/when that lifecycle is supported.
        record = DatasetDatabase(
            owner_id=user.id,
            dataset_id=dataset_id,
            cognee_version=get_cognee_version(),
            migration_revision=head_revision(MIGRATIONS),
            **graph_config_dict,  # Unpack graph db config
            **vector_config_dict,  # Unpack vector db config
        )

        try:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

        except IntegrityError:
            await session.rollback()
            raise
