import os
import asyncio
import requests
from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph import get_graph_config
from cognee.modules.users.models import User

from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface


class Neo4jAuraDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with Neo4j Aura Dataset databases.
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
                "Neo4jAuraDatasetDatabaseHandler can only be used with Neo4j graph database provider."
            )

        graph_db_name = f"{dataset_id}"

        # Client credentials
        client_id = os.environ.get("NEO4J_CLIENT_ID", None)
        client_secret = os.environ.get("NEO4J_CLIENT_SECRET", None)
        tenant_id = os.environ.get("NEO4J_TENANT_ID", None)

        if client_id is None or client_secret is None or tenant_id is None:
            raise ValueError(
                "NEO4J_CLIENT_ID, NEO4J_CLIENT_SECRET, and NEO4J_TENANT_ID environment variables must be set to use Neo4j Aura DatasetDatabase Handling."
            )

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

        response = requests.post(url, headers=headers, json=payload)

        graph_db_name = "neo4j"  # Has to be 'neo4j' for Aura
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
        return {
            "graph_database_name": graph_db_name,
            "graph_database_url": graph_db_url,
            "graph_database_provider": "neo4j",
            "graph_database_key": graph_db_key,  # TODO: Hashing of keys/passwords in relational DB
            "graph_database_username": graph_db_username,
            "graph_database_password": graph_db_password,
        }

    @classmethod
    async def delete_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]):
        pass
