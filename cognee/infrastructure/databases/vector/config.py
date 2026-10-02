import json
import os
from functools import lru_cache
from pathlib import Path

import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.base_config import get_base_config
from cognee.root_dir import ensure_absolute_path


class VectorConfig(BaseSettings):
    """
    Manage the configuration settings for the vector database.

    Public methods:
    - to_dict: Convert the configuration to a dictionary.

    Instance variables:
    - vector_db_url: The URL of the vector database.
    - vector_db_port: The port for the vector database.
    - vector_db_name: The name of the vector database.
    - vector_db_key: The key for accessing the vector database.
    - vector_db_provider: The provider for the vector database.
    """

    vector_db_url: str = ""
    vector_db_port: int = 1234
    vector_db_name: str = ""
    vector_db_key: str = ""
    vector_db_provider: str = "lancedb"
    vector_dataset_database_handler: str = "lancedb"
    vector_db_username: str = ""
    vector_db_password: str = ""
    vector_db_host: str = ""
    vector_db_subprocess_enabled: bool = True
    vector_pool_args: str | None = None
    # LanceDB compaction (ignored by other backends). Every upsert appends a
    # new fragment and leaves the superseded rows on disk, and LanceDB never
    # reclaims them on its own -- one store reached 136 GB for ~6 GB of live
    # vectors (https://github.com/topoteretes/cognee/issues/4684).
    # `LanceDBAdapter.compact` runs once per pipeline run and merges only the
    # small fragments, so a run costs at most one fragment of
    # `target_rows_per_fragment` rows, not a rewrite of the whole table.
    # `enabled = False` restores the uncompacted behaviour exactly.
    vector_db_compaction_enabled: bool = True
    # Fragments with fewer rows than this are merged; larger ones are left
    # alone. 20k rows is ~250 MB at 3072 dims: seconds even on a slow disk.
    vector_db_compaction_target_rows_per_fragment: int = 20_000
    # Compaction tasks executed per maintenance pass, shared across all tables
    # of the store (a task rewrites one group of fragments into one fragment of
    # at most `target_rows_per_fragment` rows). An existing backlog drains over
    # several runs instead of stalling one. `0` = no limit.
    vector_db_compaction_max_tasks_per_run: int = 4
    # Superseded files are deleted only once older than this, so a reader that
    # opened the table shortly before a compaction keeps working.
    vector_db_compaction_retention_seconds: int = 300

    model_config = SettingsConfigDict(extra="allow")

    @pydantic.model_validator(mode="after")
    def fill_derived(self):
        # Note: When the vector provider is pgvector, automatically use the pgvector
        # dataset handler instead of the default lancedb one. This mirrors the same
        # pattern used in GraphConfig for postgres → postgres_graph.
        provider = self.vector_db_provider.lower()
        self.vector_db_provider = provider
        vector_dataset_database_handler = self.vector_dataset_database_handler.lower()
        self.vector_dataset_database_handler = vector_dataset_database_handler
        if provider == "pgvector" and vector_dataset_database_handler in ("lancedb", "pgvector"):
            self.vector_dataset_database_handler = "pgvector"
        elif provider == "turso" and vector_dataset_database_handler in ("lancedb", "turso"):
            self.vector_dataset_database_handler = "turso"
        return self

    @pydantic.model_validator(mode="after")
    def parse_vector_pool_args(self):
        if self.vector_pool_args and isinstance(self.vector_pool_args, str):
            try:
                parsed = json.loads(self.vector_pool_args)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"VECTOR_POOL_ARGS must be valid JSON: {e.msg} (line {e.lineno}, column {e.colno})"
                ) from e
            if isinstance(parsed, dict):
                # Stored as sorted tuple for hashability (cache key compatibility).
                self.vector_pool_args = tuple(sorted(parsed.items()))
            else:
                raise ValueError("VECTOR_POOL_ARGS must be a JSON string representing a dictionary")
        return self

    @pydantic.model_validator(mode="after")
    def validate_paths(self):
        base_config = get_base_config()

        # If vector_db_url is provided and is not a path skip checking if path is absolute (as it can also be a url)
        if self.vector_db_url and Path(self.vector_db_url).exists():
            # Relative path to absolute
            self.vector_db_url = ensure_absolute_path(
                self.vector_db_url,
            )
        elif not self.vector_db_url:
            # Default path
            databases_directory_path = os.path.join(base_config.system_root_directory, "databases")
            self.vector_db_url = os.path.join(databases_directory_path, "cognee.lancedb")

        import sys

        if sys.platform == "win32" and self.vector_db_url and "://" not in self.vector_db_url:
            if os.path.isabs(self.vector_db_url) and not self.vector_db_url.startswith("\\\\?\\"):
                self.vector_db_url = "\\\\?\\" + os.path.normpath(self.vector_db_url)

        return self

    def to_dict(self) -> dict:
        """
        Convert the configuration settings to a dictionary.

        Returns:
        --------

            - dict: A dictionary containing the vector database configuration settings.
        """
        return {
            "vector_db_url": self.vector_db_url,
            "vector_db_port": self.vector_db_port,
            "vector_db_name": self.vector_db_name,
            "vector_db_key": self.vector_db_key,
            "vector_db_provider": self.vector_db_provider,
            "vector_dataset_database_handler": self.vector_dataset_database_handler,
            "vector_db_username": self.vector_db_username,
            "vector_db_password": self.vector_db_password,
            "vector_db_host": self.vector_db_host,
            "vector_db_subprocess_enabled": self.vector_db_subprocess_enabled,
        }


@lru_cache
def get_vectordb_config():
    """
    Retrieve the cached vector database configuration.

    This function uses the LRU cache to store the instance of `VectorConfig`, allowing for
    efficient reuse without needing to recreate the object multiple times. If a
    configuration is already cached, it returns that instead of creating a new one.

    Returns:
    --------

        - VectorConfig: An instance of `VectorConfig` containing the vector database
          configuration.
    """
    return VectorConfig()


def get_vectordb_context_config():
    """This function will get the appropriate vector db config based on async context."""
    from cognee.context_global_variables import vector_db_config

    if vector_db_config.get():
        return vector_db_config.get()
    return get_vectordb_config().to_dict()
