"""kuzu-11-migration

Revision ID: b9274c27a25a
Revises: e4ebee1091e7
Create Date: 2025-07-24 17:11:52.174737

"""

import os
from typing import Sequence, Union
import kuzu

from cognee.infrastructure.databases.graph.kuzu.kuzu_migrate import (
    kuzu_migration,
    read_kuzu_storage_version,
)
from cognee.infrastructure.files.storage import get_file_storage, LocalFileStorage
from cognee.infrastructure.utils.run_sync import run_sync


# revision identifiers, used by Alembic.
revision: str = "b9274c27a25a"
down_revision: Union[str, None] = "e4ebee1091e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This migration is only for multi-user Cognee mode
    if os.getenv("ENABLE_BACKEND_ACCESS_CONTROL", "false").lower() == "true":
        from cognee.base_config import get_base_config

        base_config = get_base_config()
        storage_manager = get_file_storage(base_config.system_root_directory)
        database_root_name = "databases"
        databases_root = os.path.join(base_config.system_root_directory, database_root_name)
        if not run_sync(storage_manager.is_dir(database_root_name)):
            raise FileNotFoundError(f"Directory not found: {databases_root}")

        # Get a generator for walking the database root
        is_cloud_storage = not isinstance(storage_manager.storage, LocalFileStorage)
        walk_generator = (
            storage_manager.storage.fs.walk(databases_root)
            if is_cloud_storage
            else os.walk(databases_root)
        )

        for current_path, dirnames, _ in walk_generator:
            # If file is kuzu graph database
            if ".pkl" in current_path[-4:]:
                # Add the scheme to the current path if it's a cloud storage path
                if is_cloud_storage and not current_path.startswith(storage_manager.storage.scheme):
                    current_path = os.path.join(storage_manager.storage.scheme, current_path)

                kuzu_db_version = read_kuzu_storage_version(current_path)
                if (
                    kuzu_db_version == "0.9.0" or kuzu_db_version == "0.8.2"
                ) and kuzu_db_version != kuzu.__version__:
                    # Try to migrate kuzu database to latest version
                    kuzu_migration(
                        new_db=current_path + "_new",
                        old_db=current_path,
                        new_version=kuzu.__version__,
                        old_version=kuzu_db_version,
                        overwrite=True,
                    )

    else:
        from cognee.infrastructure.databases.graph import get_graph_config

        graph_config = get_graph_config()
        databases_root = graph_config.graph_file_path
        databases_root_dir = os.path.dirname(databases_root)
        databases_root_name = os.path.basename(databases_root)
        storage_manager = get_file_storage(databases_root_dir)

        if graph_config.graph_database_provider.lower() == "kuzu":
            if run_sync(storage_manager.file_exists(databases_root_name)):
                kuzu_db_version = read_kuzu_storage_version(graph_config.graph_file_path)
                if (
                    kuzu_db_version == "0.9.0" or kuzu_db_version == "0.8.2"
                ) and kuzu_db_version != kuzu.__version__:
                    # Try to migrate kuzu database to latest version
                    kuzu_migration(
                        new_db=graph_config.graph_file_path + "_new",
                        old_db=graph_config.graph_file_path,
                        new_version=kuzu.__version__,
                        old_version=kuzu_db_version,
                        overwrite=True,
                    )


def downgrade() -> None:
    # To downgrade you will have to manually change the backup old kuzu graph databases
    # stored in the user folder to its previous name and remove the new kuzu graph
    # database that replaced it
    pass
