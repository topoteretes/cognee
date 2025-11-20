import os
import asyncio
import requests
from uuid import UUID
from typing import Union, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from cognee.base_config import get_base_config
from cognee.modules.data.methods import create_dataset
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.modules.data.methods import get_unique_dataset_id
from cognee.modules.users.models import DatasetDatabase
from cognee.modules.users.models import User


async def _get_vector_db_info(dataset_id: UUID, user: User) -> dict:
    vector_config = get_vectordb_config()

    # Determine vector configuration
    if vector_config.vector_db_provider == "lancedb":
        # TODO: Have the create_database method be called from interface adapter automatically for all providers instead of specifically here
        from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

        return await LanceDBAdapter.create_dataset(dataset_id, user)

    else:
        # Note: for hybrid databases both graph and vector DB name have to be the same
        vector_db_name = vector_config.vector_db_name
        vector_db_url = vector_config.vector_database_url

    return {
        "vector_database_name": vector_db_name,
        "vector_database_url": vector_db_url,
        "vector_database_provider": vector_config.vector_db_provider,
        "vector_database_key": vector_config.vector_db_key,
    }


async def _get_graph_db_info(dataset_id: UUID, user: User) -> dict:
    graph_config = get_graph_config()
    # Determine graph database URL
    if graph_config.graph_database_provider == "neo4j":
        from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

        return await Neo4jAdapter.create_dataset(dataset_id, user)

    elif graph_config.graph_database_provider == "kuzu":
        # TODO: Add graph file path info for kuzu (also in DatasetDatabase model)
        graph_db_name = f"{dataset_id}.pkl"
        graph_db_url = graph_config.graph_database_url
        graph_db_key = graph_config.graph_database_key
        graph_db_username = graph_config.graph_database_username
        graph_db_password = graph_config.graph_database_password
    elif graph_config.graph_database_provider == "falkor":
        # Note: for hybrid databases both graph and vector DB name have to be the same
        graph_db_name = f"{dataset_id}"
        graph_db_url = graph_config.graph_database_url
        graph_db_key = graph_config.graph_database_key
        graph_db_username = graph_config.graph_database_username
        graph_db_password = graph_config.graph_database_password
    else:
        raise EnvironmentError(
            f"Unsupported graph database provider for backend access control: {graph_config.graph_database_provider}"
        )

    return {
        "graph_database_name": graph_db_name,
        "graph_database_url": graph_db_url,
        "graph_database_provider": graph_config.graph_database_provider,
        "graph_database_key": graph_db_key,  # TODO: Hashing of keys/passwords in relational DB
        "graph_database_username": graph_db_username,
        "graph_database_password": graph_db_password,
    }


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
        async with db_engine.get_async_session() as session:
            await create_dataset(dataset, user, session)

    # If dataset database already exists return it
    existing_dataset_database = await _existing_dataset_database(dataset_id, user)
    if existing_dataset_database:
        return existing_dataset_database

    graph_config_dict = await _get_graph_db_info(dataset_id, user)
    vector_config_dict = await _get_vector_db_info(dataset_id, user)

    async with db_engine.get_async_session() as session:
        # If there are no existing rows build a new row
        # TODO: Update Dataset Database migrations, also make sure database_name is not unique anymore
        record = DatasetDatabase(
            owner_id=user.id,
            dataset_id=dataset_id,
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
