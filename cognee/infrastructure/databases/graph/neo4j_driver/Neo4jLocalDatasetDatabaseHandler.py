import os
from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph import get_graph_config
from cognee.modules.users.models import User, DatasetDatabase
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface


class Neo4jLocalDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for multi-user access control with a single local Neo4j instance.
    
    This handler supports two strategies based on Neo4j edition:
    1. Enterprise Edition: Uses Neo4j's multi-database feature to create a separate
       database for each dataset, providing true data isolation.
    2. Community Edition: Uses label-based separation with dataset prefixes to
       filter nodes/relationships per dataset.
    
    Environment variables required:
    - GRAPH_DATABASE_URL: Neo4j connection URL (e.g., bolt://neo4j:7687)
    - GRAPH_DATABASE_USERNAME: Neo4j username
    - GRAPH_DATABASE_PASSWORD: Neo4j password
    - GRAPH_DATABASE_NAME: Default database name (default: 'neo4j')
    - NEO4J_EDITION: 'enterprise' or 'community' (default: auto-detect via API call)
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """
        Create connection info for a dataset using local Neo4j.
        
        For Enterprise: Creates a new database named after the dataset.
        For Community: Sets up label-based filtering for the dataset.
        
        Args:
            dataset_id: UUID of the dataset
            user: User object who owns the dataset
            
        Returns:
            dict: Connection info for the dataset's graph database
        """
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "neo4j":
            raise ValueError(
                "Neo4jLocalDatasetDatabaseHandler can only be used with Neo4j graph database provider."
            )

        if dataset_id is None:
            raise ValueError("dataset_id is required for Neo4jLocalDatasetDatabaseHandler.")

        # Get Neo4j connection details from environment
        graph_db_url = os.environ.get("GRAPH_DATABASE_URL", "")
        graph_db_username = os.environ.get("GRAPH_DATABASE_USERNAME", "")
        graph_db_password = os.environ.get("GRAPH_DATABASE_PASSWORD", "")
        graph_db_name = os.environ.get("GRAPH_DATABASE_NAME", "neo4j")

        if not graph_db_url:
            raise EnvironmentError(
                "GRAPH_DATABASE_URL environment variable must be set for Neo4jLocalDatasetDatabaseHandler."
            )

        # Determine dataset database name (Enterprise) or label (Community)
        dataset_db_name = f"dataset_{str(dataset_id).replace('-', '_')}"
        # Neo4j database names: max 63 chars, alphanumeric and underscores only
        # dataset_id is 36 chars, prefix "dataset_" is 8 chars, underscores replace 4 hyphens = 36 + 8 = 44 chars
        dataset_db_name = dataset_db_name[:63]

        neo4j_edition = os.environ.get("NEO4J_EDITION", "").lower()

        # If edition not specified, try to detect from Neo4j
        if not neo4j_edition:
            try:
                from neo4j import AsyncGraphDatabase
                driver = AsyncGraphDatabase.driver(
                    graph_db_url,
                    auth=(graph_db_username, graph_db_password) if graph_db_username and graph_db_password else None,
                )
                async with driver.session(database="system") as session:
                    result = await session.run("SHOW HOME DATABASES")
                    # Check if we can create databases (Enterprise feature)
                    try:
                        await session.run("CREATE DATABASE test_detection IF NOT EXISTS")
                        await session.run("DROP DATABASE test_detection")
                        neo4j_edition = "enterprise"
                    except Exception:
                        neo4j_edition = "community"
                await driver.close()
            except Exception:
                # Default to community if detection fails
                neo4j_edition = "community"

        # Build connection info
        connection_info = {
            "graph_database_username": graph_db_username,
            "graph_database_password": graph_db_password,
        }

        if neo4j_edition == "enterprise":
            # For Enterprise, we'll create a database per dataset
            # Store the dataset_db_name for database selection
            connection_info["dataset_database_name"] = dataset_db_name
            connection_info["use_separate_database"] = True
        else:
            # For Community, use label-based separation
            dataset_label = f"Dataset_{str(dataset_id).replace('-', '_')}"
            connection_info["dataset_label"] = dataset_label
            connection_info["use_separate_database"] = False

        return {
            "graph_database_name": dataset_db_name if neo4j_edition == "enterprise" else graph_db_name,
            "graph_database_url": graph_db_url,
            "graph_database_provider": "neo4j",
            "graph_database_key": "",
            "graph_dataset_database_handler": "neo4j_local",
            "graph_database_connection_info": connection_info,
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        """
        Resolve runtime connection details for the dataset's Neo4j database.
        
        For local Neo4j, the connection info is stored directly in the environment
        variables, so we just need to read them and populate the dataset_database
        with the correct URL, credentials, and database name.
        
        Args:
            dataset_database: DatasetDatabase row from the relational database
            
        Returns:
            DatasetDatabase: Updated instance with resolved connection info
        """
        connection_info = dataset_database.graph_database_connection_info

        # Get connection details from environment
        graph_db_url = os.environ.get("GRAPH_DATABASE_URL", "")
        graph_db_username = os.environ.get("GRAPH_DATABASE_USERNAME", "")
        graph_db_password = os.environ.get("GRAPH_DATABASE_PASSWORD", "")

        # Update dataset_database with resolved info
        dataset_database.graph_database_url = graph_db_url
        dataset_database.graph_database_username = graph_db_username
        dataset_database.graph_database_password = graph_db_password

        # For Enterprise with separate databases, use the dataset's own database
        if connection_info.get("use_separate_database", False):
            dataset_database.graph_database_name = connection_info.get("dataset_database_name", "neo4j")
        else:
            # For Community or single database mode, use the default database
            dataset_database.graph_database_name = os.environ.get("GRAPH_DATABASE_NAME", "neo4j")

        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        """
        Delete the dataset's graph database or清除 dataset data.
        
        For Enterprise: Drops the dataset's database.
        For Community: Deletes all nodes with the dataset label.
        
        Args:
            dataset_database: DatasetDatabase row containing connection info
        """
        from neo4j import AsyncGraphDatabase

        connection_info = dataset_database.graph_database_connection_info

        graph_db_url = os.environ.get("GRAPH_DATABASE_URL", "")
        graph_db_username = os.environ.get("GRAPH_DATABASE_USERNAME", "")
        graph_db_password = os.environ.get("GRAPH_DATABASE_PASSWORD", "")

        driver = AsyncGraphDatabase.driver(
            graph_db_url,
            auth=(graph_db_username, graph_db_password) if graph_db_username and graph_db_password else None,
        )

        try:
            if connection_info.get("use_separate_database", False):
                # Enterprise: Drop the dataset database
                dataset_db_name = connection_info.get("dataset_database_name")
                if dataset_db_name:
                    async with driver.session(database="system") as session:
                        await session.run(f"DROP DATABASE {dataset_db_name} IF EXISTS")
            else:
                # Community: Delete nodes with dataset label
                dataset_label = connection_info.get("dataset_label")
                if dataset_label:
                    async with driver.session(database=dataset_database.graph_database_name or "neo4j") as session:
                        await session.run(
                            f"MATCH (n:`{dataset_label}`) DETACH DELETE n"
                        )
        finally:
            await driver.close()