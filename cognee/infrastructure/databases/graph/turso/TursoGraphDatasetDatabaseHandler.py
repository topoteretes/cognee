import os
from uuid import UUID
from typing import Optional

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    evict_graph_engine,
)
from cognee.modules.users.models import User, DatasetDatabase


class TursoGraphDatasetDatabaseHandler:
    """Handler for per-dataset Turso/libSQL graph databases.

    Each dataset gets its own libSQL file under the system databases directory, so
    the existing multi-user permission system isolates datasets by file.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "turso":
            raise ValueError(
                "TursoGraphDatasetDatabaseHandler can only be used "
                "with the turso graph database provider."
            )

        base_config = get_base_config()
        databases_dir = os.path.join(base_config.system_root_directory, "databases")
        os.makedirs(databases_dir, exist_ok=True)

        db_file = os.path.join(databases_dir, f"graph_{dataset_id}.db")
        # sqlite+aiosqlite:/// needs three slashes for absolute path
        dataset_url = f"/{db_file}" if not db_file.startswith("/") else db_file

        engine = create_graph_engine(
            graph_database_provider="turso",
            graph_file_path="",
            graph_database_url=dataset_url,
            graph_database_key="",
        )
        await engine.initialize()

        return {
            "graph_database_provider": "turso",
            "graph_database_url": dataset_url,
            "graph_database_name": str(dataset_id),
            "graph_database_key": "",
            "graph_dataset_database_handler": "turso_graph",
            "graph_database_connection_info": {},
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        # A local libSQL file has no connection credentials to resolve.
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_url = str(dataset_database.graph_database_url or "")

        evict_graph_engine(
            graph_database_provider="turso",
            graph_file_path="",
            graph_database_url=dataset_url,
            graph_database_key="",
        )

        # Remove the dataset's libSQL file.
        if dataset_url.startswith("/") and os.path.exists(dataset_url):
            os.remove(dataset_url)
