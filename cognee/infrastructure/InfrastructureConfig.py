import logging
from cognee.config import Config
from .data.chunking.config import get_chunk_config
from .llm.llm_interface import LLMInterface
from .llm.get_llm_client import get_llm_client
from ..shared.data_models import GraphDBType, DefaultContentPrediction, KnowledgeGraph, SummarizedContent, \
    LabeledContent, DefaultCognitiveLayer

logging.basicConfig(level=logging.DEBUG)

config = Config()
config.load()

chunk_config = get_chunk_config()
class InfrastructureConfig():
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
    chunk_strategy = chunk_config.chunk_strategy
    chunk_engine = None
    llm_provider: str = None
    llm_model: str = None
    llm_endpoint: str = None
    llm_api_key: str = None

    def get_config(self, config_entity: str = None) -> dict:
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

        if (config_entity is None or config_entity == "llm_engine") and self.llm_engine is None:
            self.llm_engine = get_llm_client()

        if config_entity is not None:
            return getattr(self, config_entity)

        return {
            "llm_engine": self.llm_engine,
            "classification_model": self.classification_model,
            "summarization_model": self.summarization_model,
            "labeling_model": self.labeling_model,
            "graph_model": self.graph_model,
            "cognitive_layer_model": self.cognitive_layer_model,
            "llm_provider": self.llm_provider,
            "intra_layer_score_treshold": self.intra_layer_score_treshold,
            "embedding_engine": self.embedding_engine,
            "connect_documents": self.connect_documents,
            "chunk_strategy": self.chunk_strategy,
            "chunk_engine": self.chunk_engine,
        }

    def set_config(self, new_config: dict):
        if "classification_model" in new_config:
            self.classification_model = new_config["classification_model"]

        if "summarization_model" in new_config:
            self.summarization_model = new_config["summarization_model"]

        if "labeling_model" in new_config:
            self.labeling_model = new_config["labeling_model"]

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

infrastructure_config = InfrastructureConfig()
