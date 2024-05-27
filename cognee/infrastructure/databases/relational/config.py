import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.base_config import get_base_config
from .create_relational_engine import create_relational_engine

class RelationalConfig(BaseSettings):
    db_path: str =  os.path.join(get_base_config().system_root_directory, "databases")
    db_name: str =  "cognee.db"
    db_host: str =  "localhost"
    db_port: str =  "5432"
    db_user: str = "cognee"
    db_password: str =  "cognee"
    database_engine: object = create_relational_engine(db_path, db_name)
    db_file_path: str = os.path.join(db_path, db_name)

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def create_engine(self):
        return create_relational_engine(self.db_path, self.db_name)

    def to_dict(self) -> dict:
        return {
            "db_path": self.db_path,
            "db_name": self.db_name,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_user": self.db_user,
            "db_password": self.db_password,
            "db_engine": self.database_engine,
        }

@lru_cache
def get_relationaldb_config():
    return RelationalConfig()
