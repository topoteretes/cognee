from cognee.config import Config
from .databases.relational import SqliteEngine, DatabaseEngine

config = Config()
config.load()

class InfrastructureConfig():
    database_engine: DatabaseEngine = None

    def get_config(self) -> dict:
        if self.database_engine is None:
            self.database_engine = SqliteEngine(config.db_path, config.db_name)

        return {
            "database_engine": self.database_engine
        }

    def set_config(self, new_config: dict):
        self.database_engine = new_config["database_engine"]

infrastructure_config = InfrastructureConfig()
