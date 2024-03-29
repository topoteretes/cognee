from cognee.config import Config
from .databases.relational import SqliteEngine, DatabaseEngine
from .databases.vector.weaviate_db import WeaviateAdapter
from .databases.vector.vector_db_interface import VectorDBInterface
from .databases.vector.embeddings.DefaultEmbeddingEngine import DefaultEmbeddingEngine
from .llm.llm_interface import LLMInterface
from .llm.openai.adapter import OpenAIAdapter

config = Config()
config.load()

class InfrastructureConfig():
    data_path: str = config.data_path
    database_engine: DatabaseEngine = None
    vector_engine: VectorDBInterface = None
    llm_engine: LLMInterface = None

    def get_config(self) -> dict:
        if self.database_engine is None:
            self.database_engine = SqliteEngine(config.db_path, config.db_name)

        if self.llm_engine is None:
            self.llm_engine = OpenAIAdapter(config.openai_key, config.model)

        if self.vector_engine is None:
            self.vector_engine = WeaviateAdapter(
                config.weaviate_url,
                config.weaviate_api_key,
                embedding_engine = DefaultEmbeddingEngine()
            )

        return {
            "data_path": self.data_path,
            "llm_engine": self.llm_engine,
            "vector_engine": self.vector_engine,
            "database_engine": self.database_engine,
        }

    def set_config(self, new_config: dict):
        if "data_path" in new_config:
            self.data_path = new_config["data_path"]

        if "database_engine" in new_config:
            self.database_engine = new_config["database_engine"]

        if "vector_engine" in new_config:
            self.vector_engine = new_config["vector_engine"]

        if "llm_engine" in new_config:
            self.llm_engine = new_config["llm_engine"]

infrastructure_config = InfrastructureConfig()
