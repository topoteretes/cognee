from cognee.config import Config
from .databases.relational import SqliteEngine, DatabaseEngine
from .databases.vector import WeaviateAdapter, VectorDBInterface, DefaultEmbeddingEngine

config = Config()
config.load()

class InfrastructureConfig():
    database_engine: DatabaseEngine = None
    vector_engine: VectorDBInterface = None

    def get_config(self) -> dict:
        if self.database_engine is None:
            self.database_engine = SqliteEngine(config.db_path, config.db_name)

        if self.vector_engine is None:
            self.vector_engine = WeaviateAdapter(
                config.weaviate_url,
                config.weaviate_api_key,
                embedding_engine = DefaultEmbeddingEngine()
            )

        return {
            "database_engine": self.database_engine,
            "vector_engine": self.vector_engine
        }

    def set_config(self, new_config: dict):
        self.database_engine = new_config["database_engine"]
        self.vector_engine = new_config["vector_engine"]

infrastructure_config = InfrastructureConfig()
