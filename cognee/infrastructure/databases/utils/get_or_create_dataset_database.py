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

    base_config = get_base_config()
    databases_directory_path = os.path.join(
        base_config.system_root_directory, "databases", str(user.id)
    )

    # Determine vector configuration
    if vector_config.vector_db_provider == "lancedb":
        vector_db_name = f"{dataset_id}.lance.db"
        vector_db_url = os.path.join(databases_directory_path, vector_db_name)
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
        graph_db_name = f"{dataset_id}"
        # Auto deploy instance to Aura DB
        # OAuth2 token endpoint

        # Your client credentials
        client_id = os.environ.get("NEO4J_CLIENT_ID", None)
        client_secret = os.environ.get("NEO4J_CLIENT_SECRET", None)
        tenant_id = os.environ.get("NEO4J_TENANT_ID", None)

        # Make the request with HTTP Basic Auth
        def get_aura_token(client_id: str, client_secret: str) -> dict:
            url = "https://api.neo4j.io/oauth/token"
            data = {"grant_type": "client_credentials"}  # sent as application/x-www-form-urlencoded

            resp = requests.post(url, data=data, auth=(client_id, client_secret))
            resp.raise_for_status()  # raises if the request failed
            return resp.json()

        resp = get_aura_token(client_id, client_secret)

        url = "https://api.neo4j.io/v1/instances"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {resp['access_token']}",
            "Content-Type": "application/json",
        }

        payload = {
            "version": "5",
            "region": "europe-west1",
            "memory": "1GB",
            "name": graph_db_name[0:29],
            "type": "professional-db",
            "tenant_id": tenant_id,
            "cloud_provider": "gcp",
        }

        response = requests.post(url, headers=headers, json=payload)

        print(response.status_code)
        print(response.text)
        # TODO: Find better name to name Neo4j instance within 30 character limit
        print(graph_db_name[0:29])
        graph_db_name = "neo4j"
        graph_db_url = response.json()["data"]["connection_url"]
        graph_db_key = resp["access_token"]
        graph_db_username = response.json()["data"]["username"]
        graph_db_password = response.json()["data"]["password"]

        async def _wait_for_neo4j_instance_provisioning(instance_id: str, headers: dict):
            # Poll until the instance is running
            status_url = f"https://api.neo4j.io/v1/instances/{instance_id}"
            status = ""
            for attempt in range(30):  # Try for up to ~5 minutes
                status_resp = requests.get(status_url, headers=headers)
                status = status_resp.json()["data"]["status"]
                if status.lower() == "running":
                    return
                await asyncio.sleep(10)
            raise TimeoutError(
                f"Neo4j instance '{graph_db_name}' did not become ready within 5 minutes. Status: {status}"
            )

        instance_id = response.json()["data"]["id"]
        await _wait_for_neo4j_instance_provisioning(instance_id, headers)

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
        "graph_database_key": graph_db_key,
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
