import os
import base64
import hashlib
import secrets
from uuid import UUID
from typing import Optional

from cryptography.fernet import Fernet

from cognee.infrastructure.databases.graph import get_graph_config
from cognee.infrastructure.databases.dataset_database_handler import (
    DatasetDatabaseHandlerInterface,
)
from cognee.modules.users.models import User, DatasetDatabase
from cognee.shared.logging_utils import get_logger

from .neo4j_community_containers import (
    allocate_free_port,
    bolt_url_for_port,
    check_docker_available,
    container_name_for_dataset,
    get_container_manager,
    volume_name_for_dataset,
)

logger = get_logger("Neo4jCommunityDatasetDatabaseHandler")

NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER = "neo4j_community"

# Community edition serves exactly one user database and it is named "neo4j";
# per-dataset isolation comes from one container per dataset, not CREATE DATABASE.
NEO4J_COMMUNITY_DATABASE_NAME = "neo4j"
NEO4J_COMMUNITY_USERNAME = "neo4j"

# Port allocation races with other processes between the free-port probe and
# docker binding it; retry container creation with a fresh port a few times.
PORT_ALLOCATION_ATTEMPTS = 3


class Neo4jCommunityDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """Per-dataset graph isolation on Neo4j Community edition via Docker.

    Neo4j Community supports exactly ONE database per running server —
    ``CREATE DATABASE`` (used by ``Neo4jDatasetDatabaseHandler``) is
    Enterprise-only. This handler instead manages one Docker container per
    dataset, mirroring ``Neo4jAuraDevDatasetDatabaseHandler``'s
    one-instance-per-dataset model with Docker in place of the Aura REST API:

    - ``create_dataset`` provisions a named container with a named data volume
      and a fixed localhost-published bolt port, secured with a random
      per-dataset password (stored Fernet-encrypted, key derived from
      ``NEO4J_ENCRYPTION_KEY`` — same scheme as the Aura handler).
    - ``resolve_dataset_connection_info`` runs before every operation: it
      decrypts the credentials and auto-starts the container when it is
      stopped (data persists on the volume across stops and Cognee restarts).
    - Idle containers auto-stop when the graph-engine LRU cache evicts the
      dataset's engine (``Neo4jCommunityAdapter.close``), and the number of
      concurrently running containers is bounded by
      ``NEO4J_COMMUNITY_MAX_CONTAINERS`` (default:
      ``DATABASE_MAX_LRU_CACHE_SIZE``).
    - ``delete_dataset`` removes the container and its volume.

    Production caveats (shared with the Aura dev handler): credentials are
    encrypted with a key from the environment rather than a secret manager,
    and the Docker daemon must be reachable from the Cognee process.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "neo4j":
            raise ValueError(
                "Neo4jCommunityDatasetDatabaseHandler can only be used with the Neo4j "
                "graph database provider."
            )

        if dataset_id is None:
            raise ValueError("dataset_id is required to create a Neo4j Community dataset database.")

        docker_ok, docker_message = check_docker_available()
        if not docker_ok:
            raise RuntimeError(docker_message)

        container_name = container_name_for_dataset(dataset_id)
        volume_name = volume_name_for_dataset(dataset_id)
        password = secrets.token_urlsafe(24)

        manager = get_container_manager()
        host_port = None
        last_error: Optional[Exception] = None
        for _ in range(PORT_ALLOCATION_ATTEMPTS):
            candidate_port = allocate_free_port()
            try:
                await manager.create_container(
                    dataset_id=dataset_id,
                    container_name=container_name,
                    volume_name=volume_name,
                    host_port=candidate_port,
                    password=password,
                )
                host_port = candidate_port
                break
            except RuntimeError as error:
                last_error = error
                if "port is already allocated" not in str(error).lower():
                    raise
                logger.warning(
                    "Host port %s was taken before the container could bind it; retrying "
                    "with a fresh port",
                    candidate_port,
                )
                await manager.remove_container(container_name, volume_name)

        if host_port is None:
            raise RuntimeError(
                f"Could not allocate a free host port for the Neo4j Community container "
                f"'{container_name}' after {PORT_ALLOCATION_ATTEMPTS} attempts."
            ) from last_error

        encrypted_password = cls._get_cipher().encrypt(password.encode()).decode()

        return {
            "graph_database_provider": "neo4j",
            "graph_database_url": bolt_url_for_port(host_port),
            "graph_database_name": NEO4J_COMMUNITY_DATABASE_NAME,
            "graph_database_key": "",
            "graph_dataset_database_handler": NEO4J_COMMUNITY_DATASET_DATABASE_HANDLER,
            "graph_database_connection_info": {
                "graph_database_username": NEO4J_COMMUNITY_USERNAME,
                "graph_database_password": encrypted_password,
                "container_name": container_name,
                "volume_name": volume_name,
                "host_port": host_port,
            },
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        """Decrypt credentials and make sure the dataset's container is running.

        This runs right before every operation that opens a connection for the
        dataset, so a container stopped by the idle scale-down (or a Cognee
        restart) is transparently started again here.
        """
        info = dataset_database.graph_database_connection_info

        info["graph_database_password"] = (
            cls._get_cipher().decrypt(info["graph_database_password"].encode()).decode()
        )

        await get_container_manager().ensure_running(
            container_name=info["container_name"],
            host_port=int(info["host_port"]),
        )

        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            graph_engine_cache,
        )

        info = dataset_database.graph_database_connection_info or {}
        bolt_url = dataset_database.graph_database_url

        if bolt_url:
            # Drop cached engines for this container and wait for their
            # in-flight closes so nothing races the container removal.
            await graph_engine_cache.aevict_for_url(bolt_url)

        await get_container_manager().remove_container(
            container_name=info.get(
                "container_name",
                container_name_for_dataset(dataset_database.dataset_id),
            ),
            volume_name=info.get(
                "volume_name",
                volume_name_for_dataset(dataset_database.dataset_id),
            ),
            bolt_url=bolt_url,
        )

    @classmethod
    def _get_cipher(cls) -> Fernet:
        # Same key-derivation scheme as Neo4jAuraDevDatasetDatabaseHandler so
        # deployments configure a single NEO4J_ENCRYPTION_KEY.
        encryption_env_key = os.environ.get("NEO4J_ENCRYPTION_KEY", "test_key")
        encryption_key = base64.urlsafe_b64encode(
            hashlib.sha256(encryption_env_key.encode()).digest()
        )
        return Fernet(encryption_key)
