""" This module is used to set the configuration of the system."""
from cognee.infrastructure import infrastructure_config

class config():
    @staticmethod
    def system_root_directory(system_root_directory: str):
        infrastructure_config.set_config({
            "system_root_directory": system_root_directory
        })

    @staticmethod
    def data_root_directory(data_root_directory: str):
        infrastructure_config.set_config({
            "data_root_directory": data_root_directory
        })

    @staticmethod
    def set_classification_model(classification_model: object):
        infrastructure_config.set_config({
            "classification_model": classification_model
        })

    @staticmethod
    def set_summarization_model(summarization_model: object):
        infrastructure_config.set_config({
            "summarization_model": summarization_model
        })

    @staticmethod
    def set_labeling_model(labeling_model: object):
        infrastructure_config.set_config({
            "labeling_model": labeling_model
        })

    @staticmethod
    def set_graph_model(graph_model: object):
        infrastructure_config.set_config({
            "graph_model": graph_model
        })

    @staticmethod
    def set_cognitive_layer_model(cognitive_layer_model: object):
        infrastructure_config.set_config({
            "cognitive_layer_model": cognitive_layer_model
        })

    @staticmethod
    def set_graph_engine(graph_engine: object):
        infrastructure_config.set_config({
            "graph_engine": graph_engine
        })

    @staticmethod
    def llm_provider(llm_provider: str):
        infrastructure_config.set_config({
            "llm_provider": llm_provider
        })

    @staticmethod
    def intra_layer_score_treshold(intra_layer_score_treshold: str):
        infrastructure_config.set_config({
            "intra_layer_score_treshold": intra_layer_score_treshold
        })

    @staticmethod
    def connect_documents(connect_documents: bool):
        infrastructure_config.set_config({
            "connect_documents": connect_documents
        })
    @staticmethod
    def set_chunk_strategy(chunk_strategy: object):
        infrastructure_config.set_config({
            "chunk_strategy": chunk_strategy
        })
