import logging
import os

from cognee.config import Config
from .data.chunking.config import get_chunk_config
from .databases.relational import DatabaseEngine
from .llm.llm_interface import LLMInterface
from .llm.get_llm_client import get_llm_client
from .files.storage import LocalStorage
from ..shared.data_models import GraphDBType, DefaultContentPrediction, KnowledgeGraph, SummarizedContent, \
    LabeledContent, DefaultCognitiveLayer

logging.basicConfig(level=logging.DEBUG)
config = Config()
config.load()
from cognee.infrastructure.databases.relational.config import get_relationaldb_config
from cognee.infrastructure.databases.vector.config import get_vectordb_config
vector_db_config = get_vectordb_config()
relational = get_relationaldb_config()
chunk_config = get_chunk_config()
class InfrastructureConfig():

    system_root_directory: str = config.system_root_directory
    data_root_directory: str = config.data_root_directory
    llm_provider: str = config.llm_provider
    database_engine: DatabaseEngine = None
    graph_engine: GraphDBType = None
    llm_engine: LLMInterface = None
    classification_model = None
    summarization_model = None
    labeling_model = None
    graph_model = None
    cognitive_layer_model = None
    intra_layer_score_treshold = None
    embedding_engine = None
    connect_documents = config.connect_documents
    database_directory_path: str = None
    database_file_path: str = None
    chunk_strategy = chunk_config.chunk_strategy
    chunk_engine = None
    graph_topology = config.graph_topology
    monitoring_tool = config.monitoring_tool
    llm_provider: str = None
    llm_model: str = None
    llm_endpoint: str = None
    llm_api_key: str = None

    def get_config(self, config_entity: str = None) -> dict:

        if (config_entity is None or config_entity == "database_engine") and self.database_engine is None:


            db_path = os.path.join(self.system_root_directory,relational.db_path)

            LocalStorage.ensure_directory_exists(db_path)

            self.database_engine = relational.db_engine

        if self.graph_engine is None:
            self.graph_engine = GraphDBType.NETWORKX

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

        if self.connect_documents is None:
            self.connect_documents = config.connect_documents

        if self.chunk_strategy is None:
            self.chunk_strategy = chunk_config.chunk_strategy

        if self.chunk_engine is None:
            self.chunk_engine = chunk_config.chunk_engine

        if self.graph_topology is None:
            self.graph_topology = config.graph_topology

        if (config_entity is None or config_entity == "llm_engine") and self.llm_engine is None:
            self.llm_engine = get_llm_client()

        if (config_entity is None or config_entity == "database_directory_path") and self.database_directory_path is None:
            self.database_directory_path = self.system_root_directory + "/" + relational.db_path
        if self.database_directory_path is None:
            self.database_directory_path = self.system_root_directory + "/" + relational.db_path

        if (config_entity is None or config_entity == "database_file_path") and self.database_file_path is None:
            self.database_file_path = self.system_root_directory + "/" + relational.db_path + "/" + relational.db_name

        if config_entity is not None:
            return getattr(self, config_entity)

        return {
            "llm_engine": self.llm_engine,
            "database_engine": self.database_engine,
            "system_root_directory": self.system_root_directory,
            "data_root_directory": self.data_root_directory,
            "graph_engine": self.graph_engine,
            "classification_model": self.classification_model,
            "summarization_model": self.summarization_model,
            "labeling_model": self.labeling_model,
            "graph_model": self.graph_model,
            "cognitive_layer_model": self.cognitive_layer_model,
            "llm_provider": self.llm_provider,
            "intra_layer_score_treshold": self.intra_layer_score_treshold,
            "embedding_engine": self.embedding_engine,
            "connect_documents": self.connect_documents,
            "database_directory_path": self.database_directory_path,
            "database_path": self.database_file_path,
            "chunk_strategy": self.chunk_strategy,
            "chunk_engine": self.chunk_engine,
            "graph_topology": self.graph_topology
        }

    def set_config(self, new_config: dict):
        if "system_root_directory" in new_config:
            self.system_root_directory = new_config["system_root_directory"]

        if "data_root_directory" in new_config:
            self.data_root_directory = new_config["data_root_directory"]

        if "database_engine" in new_config:
            self.database_engine = new_config["database_engine"]

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

        if "connect_documents" in new_config:
            self.connect_documents = new_config["connect_documents"]

        if "chunk_strategy" in new_config:
            self.chunk_strategy = new_config["chunk_strategy"]

        if "chunk_engine" in new_config:
            self.chunk_engine = new_config["chunk_engine"]

        if "graph_topology" in new_config:
            self.graph_topology = new_config["graph_topology"]

infrastructure_config = InfrastructureConfig()
