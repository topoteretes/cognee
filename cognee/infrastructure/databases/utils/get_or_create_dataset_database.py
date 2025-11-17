import os
from uuid import UUID
from typing import Union

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from cognee.base_config import get_base_config
from cognee.modules.data.methods import create_authorized_dataset
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.modules.data.methods import get_unique_dataset_id
from cognee.modules.users.models import DatasetDatabase
from cognee.modules.users.models import User


async def get_or_create_dataset_database(
    dataset: Union[str, UUID],
    user: User,
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

    vector_config = get_vectordb_config()
    graph_config = get_graph_config()

    # Note: for hybrid databases both graph and vector DB name have to be the same
    if graph_config.graph_database_provider == "kuzu":
        graph_db_name = f"{dataset_id}.pkl"
    else:
        graph_db_name = f"{dataset_id}"

    if vector_config.vector_db_provider == "lancedb":
        vector_db_name = f"{dataset_id}.lance.db"
    else:
        vector_db_name = f"{dataset_id}"

    base_config = get_base_config()
    databases_directory_path = os.path.join(
        base_config.system_root_directory, "databases", str(user.id)
    )

    # Determine vector database URL
    if vector_config.vector_db_provider == "lancedb":
        vector_db_url = os.path.join(databases_directory_path, vector_config.vector_db_name)
    else:
        vector_db_url = vector_config.vector_database_url

    # Determine graph database URL

    async with db_engine.get_async_session() as session:
        # Create dataset if it doesn't exist
        if isinstance(dataset, str):
            dataset = await create_authorized_dataset(dataset, user)

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
            vector_database_provider=vector_config.vector_db_provider,
            graph_database_provider=graph_config.graph_database_provider,
            vector_database_url=vector_db_url,
            graph_database_url=graph_config.graph_database_url,
            vector_database_key=vector_config.vector_db_key,
            graph_database_key=graph_config.graph_database_key,
        )

        try:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

        except IntegrityError:
            await session.rollback()
            raise
