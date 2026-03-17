import os
import aiohttp
import asyncio
import base64
import hashlib
from uuid import UUID
from typing import Optional
from urllib.parse import urlparse
from cryptography.fernet import Fernet
from aiohttp import BasicAuth

from cognee.infrastructure.databases.graph import get_graph_config
from cognee.modules.users.models import User, DatasetDatabase
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface


class Neo4jAuraDevDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for a quick development PoC integration of Cognee multi-user and permission mode with Neo4j Aura databases.
    This handler creates a new Neo4j Aura instance for each Cognee dataset created.

    Improvements needed to be production ready:
    - Secret management for client credentials, currently secrets are encrypted and stored in the Cognee relational database,
      a secret manager or a similar system should be used instead.

    Quality of life improvements:
    - Allow configuration of different Neo4j Aura plans and regions.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """
        Create a new Neo4j Aura instance for the dataset. Return connection info that will be mapped to the dataset.

        Args:
            dataset_id: Dataset UUID
            user: User object who owns the dataset and is making the request

        Returns:
            dict: Connection details for the created Neo4j instance

        """
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "neo4j":
            raise ValueError(
                "Neo4jAuraDevDatasetDatabaseHandler can only be used with Neo4j graph database provider."
            )

        graph_db_name = f"{dataset_id}"

        # Client credentials and encryption
        # Note: Should not be used as class variables so that they are not persisted in memory longer than needed
        client_id = os.environ.get("NEO4J_CLIENT_ID", None)
        client_secret = os.environ.get("NEO4J_CLIENT_SECRET", None)
        tenant_id = os.environ.get("NEO4J_TENANT_ID", None)
        encryption_env_key = os.environ.get("NEO4J_ENCRYPTION_KEY", "test_key")
        encryption_key = base64.urlsafe_b64encode(
            hashlib.sha256(encryption_env_key.encode()).digest()
        )
        cipher = Fernet(encryption_key)

        if client_id is None or client_secret is None or tenant_id is None:
            raise ValueError(
                "NEO4J_CLIENT_ID, NEO4J_CLIENT_SECRET, and NEO4J_TENANT_ID environment variables must be set to use Neo4j Aura DatasetDatabase Handling."
            )

        resp_token = await cls._get_aura_token(client_id, client_secret)

        url = "https://api.neo4j.io/v1/instances"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {resp_token['access_token']}",
            "Content-Type": "application/json",
        }

        # TODO: Maybe we can allow **kwargs parameter forwarding for cases like these
        #       Too allow different configurations between datasets
        payload = {
            "version": "5",
            "region": "europe-west1",
            "memory": "1GB",
            "name": graph_db_name[
                0:29
            ],  # TODO: Find better name to name Neo4j instance within 30 character limit
            "type": "professional-db",
            "tenant_id": tenant_id,
            "cloud_provider": "gcp",
        }

        async def _create_database_instance_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()

        resp_create = await _create_database_instance_request()

        graph_db_name = "neo4j"  # Has to be 'neo4j' for Aura
        graph_db_url = resp_create["data"]["connection_url"]
        graph_db_key = resp_token["access_token"]
        graph_db_username = resp_create["data"]["username"]
        graph_db_password = resp_create["data"]["password"]

        async def _wait_for_neo4j_instance_provisioning(instance_id: str, headers: dict):
            # Poll until the instance is running
            status_url = f"https://api.neo4j.io/v1/instances/{instance_id}"
            status = ""
            for attempt in range(30):  # Try for up to ~5 minutes
                async with aiohttp.ClientSession() as session:
                    async with session.get(status_url, headers=headers) as resp:
                        resp.raise_for_status()
                        status_resp = await resp.json()
                        status = status_resp["data"]["status"]
                        if status.lower() == "running":
                            return
                        await asyncio.sleep(10)
            raise TimeoutError(
                f"Neo4j instance '{graph_db_name}' did not become ready within 5 minutes. Status: {status}"
            )

        instance_id = resp_create["data"]["id"]
        await _wait_for_neo4j_instance_provisioning(instance_id, headers)

        encrypted_db_password_bytes = cipher.encrypt(graph_db_password.encode())
        encrypted_db_password_string = encrypted_db_password_bytes.decode()

        return {
            "graph_database_name": graph_db_name,
            "graph_database_url": graph_db_url,
            "graph_database_provider": "neo4j",
            "graph_database_key": graph_db_key,
            "graph_dataset_database_handler": "neo4j_aura_dev",
            "graph_database_connection_info": {
                "graph_database_username": graph_db_username,
                "graph_database_password": encrypted_db_password_string,
            },
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        """
        Resolve and decrypt connection info for the Neo4j dataset database.
        In this case, decrypt the password stored in the database.

        Args:
            dataset_database: DatasetDatabase instance containing encrypted connection info.
        """
        encryption_env_key = os.environ.get("NEO4J_ENCRYPTION_KEY", "test_key")
        encryption_key = base64.urlsafe_b64encode(
            hashlib.sha256(encryption_env_key.encode()).digest()
        )
        cipher = Fernet(encryption_key)
        graph_db_password = cipher.decrypt(
            dataset_database.graph_database_connection_info["graph_database_password"].encode()
        ).decode()

        dataset_database.graph_database_connection_info["graph_database_password"] = (
            graph_db_password
        )
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        # Get dataset database information and credentials
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        parsed_url = urlparse(dataset_database.graph_database_url)
        instance_id = parsed_url.hostname.split(".")[0]

        url = f"https://api.neo4j.io/v1/instances/{instance_id}"

        # Get access token for Neo4j Aura API
        # Client credentials
        client_id = os.environ.get("NEO4J_CLIENT_ID", None)
        client_secret = os.environ.get("NEO4J_CLIENT_SECRET", None)
        resp = await cls._get_aura_token(client_id, client_secret)

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {resp['access_token']}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json()

    @classmethod
    async def _get_aura_token(cls, client_id: str, client_secret: str) -> dict:
        url = "https://api.neo4j.io/oauth/token"
        data = {"grant_type": "client_credentials"}  # sent as application/x-www-form-urlencoded

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, data=data, auth=BasicAuth(client_id, client_secret)
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
