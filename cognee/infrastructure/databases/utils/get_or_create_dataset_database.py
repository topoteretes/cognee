from uuid import UUID
from typing import Union

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from cognee.modules.data.methods import create_dataset

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.modules.data.methods import get_unique_dataset_id
from cognee.modules.users.models import DatasetDatabase
from cognee.modules.users.models import User
from .constants import (
    GRAPH_DBS_WITH_MULTI_USER_SUPPORT,
    VECTOR_DBS_WITH_MULTI_USER_SUPPORT,
    HYBRID_DBS,
)


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

    graph_db_name = f"{dataset_id}.pkl"

    if graph_config.graph_database_provider in HYBRID_DBS:
        vector_db_name = graph_db_name
    else:
        if vector_config.vector_db_provider == "lancedb":
            vector_db_name = f"{dataset_id}.lance.db"
        else:
            vector_db_name = f"{dataset_id}.db"

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

        # Check if we support multi-user for this provider. If not, use default
        if graph_config.graph_database_provider not in GRAPH_DBS_WITH_MULTI_USER_SUPPORT:
            raise EnvironmentError(
                f"Multi-user is currently not supported for the graph database provider: {graph_config.graph_database_provider}. "
                f"Supported providers are: {', '.join(GRAPH_DBS_WITH_MULTI_USER_SUPPORT)}. Either use one of these"
                f"providers, or disable BACKEND_ACCESS_CONTROL"
            )
        if vector_config.vector_db_provider not in VECTOR_DBS_WITH_MULTI_USER_SUPPORT:
            raise EnvironmentError(
                f"Multi-user is currently not supported for the vector database provider: {vector_config.vector_db_provider}. "
                f"Supported providers are: {', '.join(VECTOR_DBS_WITH_MULTI_USER_SUPPORT)}. Either use one of these"
                f"providers, or disable BACKEND_ACCESS_CONTROL"
            )

        # If there are no existing rows build a new row
        record = DatasetDatabase(
            owner_id=user.id,
            dataset_id=dataset_id,
            vector_database_name=vector_db_name,
            graph_database_name=graph_db_name,
            vector_database_provider=vector_config.vector_db_provider,
            graph_database_provider=graph_config.graph_database_provider,
            vector_database_url=vector_config.vector_db_url,
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
