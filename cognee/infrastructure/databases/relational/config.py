import os
import pydantic
from typing import Union
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.base_config import get_base_config


class RelationalConfig(BaseSettings):
    """
    Configure database connection settings.
    """

    db_path: str = ""
    db_name: str = "cognee_db"
    db_host: Union[str, None] = None  # "localhost"
    db_port: Union[str, None] = None  # "5432"
    db_username: Union[str, None] = None  # "cognee"
    db_password: Union[str, None] = None  # "cognee"
    db_provider: str = "sqlite"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @pydantic.model_validator(mode="after")
    def fill_derived(self):
        # Set file path based on graph database provider if no file path is provided
        if not self.db_path:
            base_config = get_base_config()
            databases_directory_path = os.path.join(base_config.system_root_directory, "databases")
            self.db_path = databases_directory_path

        return self

    def to_dict(self) -> dict:
        """
        Return the database configuration as a dictionary.

        Returns:
        --------

            - dict: A dictionary containing database configuration settings including db_path,
              db_name, db_host, db_port, db_username, db_password, and db_provider.
        """
        return {
            "db_path": self.db_path,
            "db_name": self.db_name,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_username": self.db_username,
            "db_password": self.db_password,
            "db_provider": self.db_provider,
        }


@lru_cache
def get_relational_config():
    """
    Cache and return the relational database configuration.

    This function retrieves an instance of the RelationalConfig class, caching it to avoid
    recreation on subsequent calls. It is designed to provide a consistent configuration
    globally for relational database connections.

    Returns:
    --------

        - RelationalConfig: An instance of the RelationalConfig containing the database
          configuration settings.
    """
    return RelationalConfig()


class MigrationConfig(BaseSettings):
    """
    Manage and configure migration settings for a database, inheriting from BaseSettings.

    Public methods:
    - to_dict: Convert the migration configuration to a dictionary format.

    Instance variables:
    - migration_db_path: Path to the migration database.
    - migration_db_name: Name of the migration database.
    - migration_db_host: Host of the migration database.
    - migration_db_port: Port of the migration database.
    - migration_db_username: Username for connecting to the migration database.
    - migration_db_password: Password for connecting to the migration database.
    - migration_db_provider: Provider type for the migration database.
    """

    migration_db_path: Union[str, None] = None
    migration_db_name: Union[str, None] = None
    migration_db_host: Union[str, None] = None
    migration_db_port: Union[str, None] = None
    migration_db_username: Union[str, None] = None
    migration_db_password: Union[str, None] = None
    migration_db_provider: Union[str, None] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        """
        Convert migration configuration to dictionary format.

        Returns:
        --------

            - dict: A dictionary containing the migration configuration details.
        """
        return {
            "migration_db_path": self.migration_db_path,
            "migration_db_name": self.migration_db_name,
            "migration_db_host": self.migration_db_host,
            "migration_db_port": self.migration_db_port,
            "migration_db_username": self.migration_db_username,
            "migration_db_password": self.migration_db_password,
            "migration_db_provider": self.migration_db_provider,
        }


@lru_cache
def get_migration_config():
    """
    Retrieve the migration configuration instance.

    This function uses memoization to cache the MigrationConfig instance, ensuring that
    subsequent calls return the same instance. It is crucial to call this function in a
    context where the environment variables are properly set, as they will configure the
    MigrationConfig instance accordingly, potentially affecting the application's database
    connectivity.

    Returns:
    --------

        - MigrationConfig: An instance of MigrationConfig containing the migration database
          configuration.
    """
    return MigrationConfig()
