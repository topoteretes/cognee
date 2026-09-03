import os
from uuid import UUID
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.modules.data.methods import get_unique_dataset_id
from cognee.modules.users.models import DatasetDatabase
from cognee.modules.users.models import User
from cognee.version import get_cognee_version
from cognee.modules.migrations.migration import head_revision
from cognee.modules.migrations.registry import MIGRATIONS


async def _get_vector_db_info(dataset_id: UUID, owner: User) -> dict:
    vector_config = get_vectordb_config()

    from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
        supported_dataset_database_handlers,
    )

    handler = supported_dataset_database_handlers[vector_config.vector_dataset_database_handler]
    return await handler["handler_instance"].create_dataset(dataset_id, owner)


async def _get_graph_db_info(dataset_id: UUID, owner: User) -> dict:
    graph_config = get_graph_config()

    from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
        supported_dataset_database_handlers,
    )

    handler = supported_dataset_database_handlers[graph_config.graph_dataset_database_handler]
    return await handler["handler_instance"].create_dataset(dataset_id, owner)


async def _existing_dataset_database(
    dataset_id: UUID,
) -> Optional[DatasetDatabase]:
    """
    Check if a DatasetDatabase row already exists for the given dataset.
    Return None if it doesn't exist, return the row if it does.

    dataset_id is the table's primary key — one row per dataset, shared by the
    owner and every ACL-granted user — so the lookup must not filter by owner:
    for a non-owner caller that turns a hit into a miss and the fall-through
    INSERT crashes on the primary key (#4829).

    Args:
        dataset_id:

    Returns:
        DatasetDatabase or None
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        stmt = select(DatasetDatabase).where(DatasetDatabase.dataset_id == dataset_id)
        existing: DatasetDatabase = await session.scalar(stmt)
        return existing


async def get_or_create_dataset_database(
    dataset_id: UUID,
    owner: User,
) -> DatasetDatabase:
    """
    Return the `DatasetDatabase` row for the given dataset; provision it on first use.

    • If the row already exists, it is fetched and returned.
    • Otherwise a new one is created atomically and returned.

    DatasetDatabase row contains connection and provider info for vector and graph databases.

    Parameters
    ----------
    dataset_id : UUID
        Id of an existing dataset. Name resolution and dataset creation happen
        at the API layer before the database context is entered.
    owner : User
        The dataset's owner — guaranteed by apply_database_context_variables,
        which derives it from the dataset. An existing row is returned without
        consulting it; on first provisioning it namespaces the physical
        databases and is stamped as the row's owner_id.
    """
    db_engine = get_relational_engine()

    dataset_id = await get_unique_dataset_id(dataset_id, owner)

    # If dataset database already exists return it
    existing_dataset_database = await _existing_dataset_database(dataset_id)
    if existing_dataset_database:
        return existing_dataset_database

    graph_config_dict = await _get_graph_db_info(dataset_id, owner)
    vector_config_dict = await _get_vector_db_info(dataset_id, owner)

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
            owner_id=owner.id,
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
