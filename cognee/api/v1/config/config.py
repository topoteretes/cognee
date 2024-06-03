""" This module is used to set the configuration of the system."""
import os
from cognee.base_config import get_base_config
from cognee.modules.cognify.config import get_cognify_config
from cognee.infrastructure.data.chunking.config import get_chunk_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.relational import get_relationaldb_config

class config():
    @staticmethod
    def system_root_directory(system_root_directory: str):
        databases_directory_path = os.path.join(system_root_directory, "databases")

        relational_config = get_relationaldb_config()
        relational_config.db_path = databases_directory_path
        relational_config.create_engine()

        vector_config = get_vectordb_config()
        vector_config.vector_db_path = databases_directory_path
        vector_config.create_engine()

    @staticmethod
    def data_root_directory(data_root_directory: str):
        base_config = get_base_config()
        base_config.data_root_directory = data_root_directory

    @staticmethod
    def monitoring_tool(monitoring_tool: object):
        base_config = get_base_config()
        base_config.monitoring_tool = monitoring_tool

    @staticmethod
    def set_classification_model(classification_model: object):
        cognify_config = get_cognify_config()
        cognify_config.classification_model =  classification_model

    @staticmethod
    def set_summarization_model(summarization_model: object):
        cognify_config = get_cognify_config()
        cognify_config.summarization_model=summarization_model

    @staticmethod
    def set_labeling_model(labeling_model: object):
        cognify_config = get_cognify_config()
        cognify_config.labeling_model =labeling_model

    @staticmethod
    def set_graph_model(graph_model: object):
        graph_config = get_graph_config()
        graph_config.graph_model = graph_model

    @staticmethod
    def set_cognitive_layer_model(cognitive_layer_model: object):
        cognify_config = get_cognify_config()
        cognify_config.cognitive_layer_model = cognitive_layer_model

    @staticmethod
    def set_graph_engine(graph_engine: object):
        graph_config = get_graph_config()
        graph_config.graph_engine = graph_engine

    @staticmethod
    def llm_provider(llm_provider: str):
        graph_config = get_graph_config()
        graph_config.llm_provider = llm_provider

    @staticmethod
    def llm_endpoint(llm_endpoint: str):
        graph_config = get_graph_config()
        graph_config.llm_endpoint = llm_endpoint

    @staticmethod
    def llm_model(llm_model: str):
        graph_config = get_graph_config()
        graph_config.llm_model = llm_model

    @staticmethod
    def intra_layer_score_treshold(intra_layer_score_treshold: str):
        cognify_config = get_cognify_config()
        cognify_config.intra_layer_score_treshold = intra_layer_score_treshold

    @staticmethod
    def connect_documents(connect_documents: bool):
        cognify_config = get_cognify_config()
        cognify_config.connect_documents = connect_documents

    @staticmethod
    def set_chunk_strategy(chunk_strategy: object):
        chunk_config = get_chunk_config()
        chunk_config.chunk_strategy = chunk_strategy
