import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.root_dir import get_absolute_path
from .create_relational_engine import create_relational_engine

class RelationalConfig(BaseSettings):
    db_path: str =  os.path.join(get_absolute_path(".cognee_system"), "databases")
    db_name: str =  "cognee.db"
    db_host: str =  "localhost"
    db_port: str =  "5432"
    db_user: str = "cognee"
    db_password: str =  "cognee"
    database_engine: object = create_relational_engine(db_path, db_name)
    db_file_path: str = os.path.join(db_path, db_name)

    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def create_engine(self):
        self.db_file_path = os.path.join(self.db_path, self.db_name)
        self.database_engine = create_relational_engine(self.db_path, self.db_name)

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
