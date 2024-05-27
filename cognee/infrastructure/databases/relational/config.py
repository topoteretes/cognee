import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.infrastructure.databases.relational import DuckDBAdapter
from cognee.base_config import get_base_config
config = get_base_config()

class RelationalConfig(BaseSettings):
    db_path: str =  os.path.join(config.system_root_directory, "databases")
    db_name: str =  "cognee.db"
    db_host: str =  "localhost"
    db_port: str =  "5432"
    db_user: str = "cognee"
    db_password: str =  "cognee"
    db_engine: object = DuckDBAdapter(
        db_name=db_name,
        db_path=db_path
    )
    database_engine: object = db_engine
    db_file_path:str = os.path.join(db_path, db_name)
    database_path: str = os.path.join(config.system_root_directory, "databases")
    database_directory_path: str = db_path

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "db_path": self.db_path,
            "db_name": self.db_name,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_user": self.db_user,
            "db_password": self.db_password,
            "db_engine": self.db_engine,
            "database_path": self.database_path,
        }

@lru_cache
def get_relationaldb_config():
    return RelationalConfig()