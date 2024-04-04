from cognee.config import Config
from .databases.graph.graph_db_interface import GraphDBInterface
from .databases.graph.networkx.adapter import NetworXAdapter
from .databases.relational import DuckDBAdapter, DatabaseEngine
from .databases.vector.vector_db_interface import VectorDBInterface
from .databases.vector.qdrant.QDrantAdapter import QDrantAdapter
from .databases.vector.embeddings.DefaultEmbeddingEngine import DefaultEmbeddingEngine
from .llm.llm_interface import LLMInterface
from .llm.openai.adapter import OpenAIAdapter
from .files.storage import LocalStorage
from ..shared.data_models import GraphDBType, DefaultContentPrediction, KnowledgeGraph, SummarizedContent, \
    LabeledContent, DefaultCognitiveLayer
from pydantic import BaseModel

config = Config()
config.load()

class InfrastructureConfig():
    system_root_directory: str = config.system_root_directory
    data_root_directory: str = config.data_root_directory
    llm_provider: str = config.llm_provider
    database_engine: DatabaseEngine = None
    vector_engine: VectorDBInterface = None
    graph_engine: GraphDBType = None
    llm_engine: LLMInterface = None
    classification_model = None
    summarization_model = None
    labeling_model = None
    graph_model = None,
    cognitive_layer_model = None
    intra_layer_score_treshold = None
    embedding_engine = None



    def get_config(self) -> dict:
        if self.database_engine is None:
            db_path = self.system_root_directory + "/" + config.db_path

            LocalStorage.ensure_directory_exists(db_path)

            self.database_engine = DuckDBAdapter(
                db_name = config.db_name,
                db_path = db_path
            )
        if self.graph_engine is None:
            self.graph_engine =  GraphDBType.NETWORKX

        if self.classification_model is None:
            self.classification_model = DefaultContentPrediction
        if self.summarization_model is None:
            self.summarization_model = SummarizedContent
        if self.labeling_model is None:
            self.labeling_model = LabeledContent
        if self.graph_model is None:
            self.graph_model = KnowledgeGraph
        if self.cognitive_layer_model is None:
            self.cognitive_layer_model = DefaultCognitiveLayer

        if self.intra_layer_score_treshold is None:
            self.intra_layer_score_treshold = config.intra_layer_score_treshold
        if self.embedding_engine is None:
            self.embedding_engine = DefaultEmbeddingEngine()


        if self.llm_engine is None:
            self.llm_engine = OpenAIAdapter(config.openai_key, config.openai_model)

        if self.vector_engine is None:
            try:
                from .databases.vector.weaviate_db import WeaviateAdapter

                if config.weaviate_url is None and config.weaviate_api_key is None:
                    raise EnvironmentError("Weaviate is not configured!")

                self.vector_engine = WeaviateAdapter(
                    config.weaviate_url,
                    config.weaviate_api_key,
                    embedding_engine = self.embedding_engine
                )
            except ImportError:
                if config.qdrant_url is None and config.qdrant_api_key is None:
                    raise EnvironmentError("Qdrant is not configured!")

                self.vector_engine = QDrantAdapter(
                    qdrant_url = config.qdrant_url,
                    qdrant_api_key = config.qdrant_api_key,
                    embedding_engine = self.embedding_engine
                )

        return {
            "llm_engine": self.llm_engine,
            "vector_engine": self.vector_engine,
            "database_engine": self.database_engine,
            "system_root_directory": self.system_root_directory,
            "data_root_directory": self.data_root_directory,
            "database_directory_path": self.system_root_directory + "/" + config.db_path,
            "database_path": self.system_root_directory + "/" + config.db_path + "/" + config.db_name,
            "graph_engine": self.graph_engine,
            "classification_model": self.classification_model,
            "summarization_model": self.summarization_model,
            "labeling_model": self.labeling_model,
            "graph_model": self.graph_model,
            "congitive_layer_model": self.cognitive_layer_model,
            "llm_provider": self.llm_provider,
            "intra_layer_score_treshold": self.intra_layer_score_treshold,
            "embedding_engine": self.embedding_engine,

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

        if "graph_engine" in new_config:
            self.graph_engine = new_config["graph_engine"]

        if "classification_model" in new_config:
            self.classification_model = new_config["classification_model"]

        if "summarization_model" in new_config:
            self.summarization_model = new_config["summarization_model"]

        if "labeling_model" in new_config:
            self.labeling_model = new_config["labeling_model"]

        if "graph_model" in new_config:
            self.graph_model = new_config["graph_model"]

        if "llm_provider" in new_config:
            self.llm_provider = new_config["llm_provider"]

        if "cognitive_layer_model" in new_config:
            self.cognitive_layer_model = new_config["cognitive_layer_model"]

        if "intra_layer_score_treshold" in new_config:
            self.intra_layer_score_treshold = new_config["intra_layer_score_treshold"]

        if "embedding_engine" in new_config:
            self.embedding_engine = new_config["embedding_engine"]

infrastructure_config = InfrastructureConfig()
