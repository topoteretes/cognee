from cognee.config import Config
from .databases.relational import DuckDBAdapter, DatabaseEngine
from .databases.vector.vector_db_interface import VectorDBInterface
from .databases.vector.qdrant.QDrantAdapter import QDrantAdapter
from .databases.vector.embeddings.DefaultEmbeddingEngine import DefaultEmbeddingEngine
from .llm.llm_interface import LLMInterface
from .llm.openai.adapter import OpenAIAdapter
from .files.storage import LocalStorage

config = Config()
config.load()

class InfrastructureConfig():
    system_root_directory: str = config.system_root_directory
    data_root_directory: str = config.data_root_directory
    database_engine: DatabaseEngine = None
    vector_engine: VectorDBInterface = None
    llm_engine: LLMInterface = None

    def get_config(self) -> dict:
        if self.database_engine is None:
            db_path = self.system_root_directory + "/" + config.db_path

            LocalStorage.ensure_directory_exists(db_path)

            self.database_engine = DuckDBAdapter(
                db_name = config.db_name,
                db_path = db_path
            )

        if self.llm_engine is None:
            self.llm_engine = OpenAIAdapter(config.openai_key, config.model)

        if self.vector_engine is None:
            try:
                from .databases.vector.weaviate_db import WeaviateAdapter

                if config.weaviate_url is None and config.weaviate_api_key is None:
                    raise EnvironmentError("Weaviate is not configured!")

                self.vector_engine = WeaviateAdapter(
                    config.weaviate_url,
                    config.weaviate_api_key,
                    embedding_engine = DefaultEmbeddingEngine()
                )
            except ImportError:
                if config.qdrant_url is None and config.qdrant_api_key is None:
                    raise EnvironmentError("Qdrant is not configured!")

                self.vector_engine = QDrantAdapter(
                    qdrant_url = config.qdrant_url,
                    qdrant_api_key = config.qdrant_api_key,
                    embedding_engine = DefaultEmbeddingEngine()
                )

        return {
            "llm_engine": self.llm_engine,
            "vector_engine": self.vector_engine,
            "database_engine": self.database_engine,
            "system_root_directory": self.system_root_directory,
            "data_root_directory": self.data_root_directory,
            "database_directory_path": self.system_root_directory + "/" + config.db_path,
            "database_path": self.system_root_directory + "/" + config.db_path + "/" + config.db_name,
        }

    def set_config(self, new_config: dict):
        if "system_root_directory" in new_config:
            self.system_root_directory = new_config["system_root_directory"]
      
        if "data_root_directory" in new_config:
            self.data_root_directory = new_config["data_root_directory"]
      
        if "database_engine" in new_config:
            self.database_engine = new_config["database_engine"]

        if "vector_engine" in new_config:
            self.vector_engine = new_config["vector_engine"]

        if "llm_engine" in new_config:
            self.llm_engine = new_config["llm_engine"]

infrastructure_config = InfrastructureConfig()
