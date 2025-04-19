import os
from typing import Union
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.root_dir import get_absolute_path


class RelationalConfig(BaseSettings):
    db_path: str = os.path.join(get_absolute_path(".cognee_system"), "databases")
    db_name: str = "cognee_db"
    db_host: Union[str, None] = None  # "localhost"
    db_port: Union[str, None] = None  # "5432"
    db_username: Union[str, None] = None  # "cognee"
    db_password: Union[str, None] = None  # "cognee"
    db_provider: str = "sqlite"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
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
    return RelationalConfig()


class MigrationConfig(BaseSettings):
    migration_db_path: Union[str, None] = None
    migration_db_name: Union[str, None] = None
    migration_db_host: Union[str, None] = None
    migration_db_port: Union[str, None] = None
    migration_db_username: Union[str, None] = None
    migration_db_password: Union[str, None] = None
    migration_db_provider: Union[str, None] = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
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
    return MigrationConfig()
