import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.root_dir import get_absolute_path

class RelationalConfig(BaseSettings):
    db_path: str =  os.path.join(get_absolute_path(".cognee_system"), "databases")
    db_name: str =  "cognee_db"
    db_host: str =  "localhost"
    db_port: str =  "5432"
    db_user: str = "cognee"
    db_password: str =  "cognee"
    db_provider: str = "postgresql+asyncpg"
    db_file_path: str = os.path.join(db_path, db_name)


    model_config = SettingsConfigDict(env_file = ".env", extra = "allow")

    def to_dict(self) -> dict:
        return {
            "db_path": self.db_path,
            "db_name": self.db_name,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_user": self.db_user,
            "db_password": self.db_password,
            "db_provider": self.db_provider,
        }

@lru_cache
def get_relational_config():
    return RelationalConfig()
