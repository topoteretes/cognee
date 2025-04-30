"""This module is used to set the configuration of the system."""

import os
from cognee.base_config import get_base_config
from cognee.exceptions import InvalidValueError, InvalidAttributeError
from cognee.modules.cognify.config import get_cognify_config
from cognee.infrastructure.data.chunking.config import get_chunk_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.databases.relational import get_relational_config, get_migration_config
from cognee.infrastructure.files.storage import LocalStorage


class config:
    @staticmethod
    def system_root_directory(system_root_directory: str):
        databases_directory_path = os.path.join(system_root_directory, "databases")

        relational_config = get_relational_config()
        relational_config.db_path = databases_directory_path
        LocalStorage.ensure_directory_exists(databases_directory_path)

        graph_config = get_graph_config()
        graph_config.graph_file_path = os.path.join(databases_directory_path, "cognee.graph")

        vector_config = get_vectordb_config()
        if vector_config.vector_db_provider == "lancedb":
            vector_config.vector_db_url = os.path.join(databases_directory_path, "cognee.lancedb")

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
        cognify_config.classification_model = classification_model

    @staticmethod
    def set_summarization_model(summarization_model: object):
        cognify_config = get_cognify_config()
        cognify_config.summarization_model = summarization_model

    @staticmethod
    def set_graph_model(graph_model: object):
        graph_config = get_graph_config()
        graph_config.graph_model = graph_model

    @staticmethod
    def set_graph_database_provider(graph_database_provider: str):
        graph_config = get_graph_config()
        graph_config.graph_database_provider = graph_database_provider

    @staticmethod
    def set_llm_provider(llm_provider: str):
        llm_config = get_llm_config()
        llm_config.llm_provider = llm_provider

    @staticmethod
    def set_llm_endpoint(llm_endpoint: str):
        llm_config = get_llm_config()
        llm_config.llm_endpoint = llm_endpoint

    @staticmethod
    def set_llm_model(llm_model: str):
        llm_config = get_llm_config()
        llm_config.llm_model = llm_model

    @staticmethod
    def set_llm_api_key(llm_api_key: str):
        llm_config = get_llm_config()
        llm_config.llm_api_key = llm_api_key

    @staticmethod
    def set_llm_config(config_dict: dict):
        """
        Updates the llm config with values from config_dict.
        """
        llm_config = get_llm_config()
        for key, value in config_dict.items():
            if hasattr(llm_config, key):
                object.__setattr__(llm_config, key, value)
            else:
                raise InvalidAttributeError(
                    message=f"'{key}' is not a valid attribute of the config."
                )

    @staticmethod
    def set_chunk_strategy(chunk_strategy: object):
        chunk_config = get_chunk_config()
        chunk_config.chunk_strategy = chunk_strategy

    @staticmethod
    def set_chunk_engine(chunk_engine: object):
        chunk_config = get_chunk_config()
        chunk_config.chunk_engine = chunk_engine

    @staticmethod
    def set_chunk_overlap(chunk_overlap: object):
        chunk_config = get_chunk_config()
        chunk_config.chunk_overlap = chunk_overlap

    @staticmethod
    def set_chunk_size(chunk_size: object):
        chunk_config = get_chunk_config()
        chunk_config.chunk_size = chunk_size

    @staticmethod
    def set_vector_db_provider(vector_db_provider: str):
        vector_db_config = get_vectordb_config()
        vector_db_config.vector_db_provider = vector_db_provider

    @staticmethod
    def set_relational_db_config(config_dict: dict):
        """
        Updates the relational db config with values from config_dict.
        """
        relational_db_config = get_relational_config()
        for key, value in config_dict.items():
            if hasattr(relational_db_config, key):
                object.__setattr__(relational_db_config, key, value)
            else:
                raise InvalidAttributeError(
                    message=f"'{key}' is not a valid attribute of the config."
                )

    @staticmethod
    def set_migration_db_config(config_dict: dict):
        """
        Updates the relational db config with values from config_dict.
        """
        migration_db_config = get_migration_config()
        for key, value in config_dict.items():
            if hasattr(migration_db_config, key):
                object.__setattr__(migration_db_config, key, value)
            else:
                raise InvalidAttributeError(
                    message=f"'{key}' is not a valid attribute of the config."
                )

    @staticmethod
    def set_graph_db_config(config_dict: dict) -> None:
        """
        Updates the graph db config with values from config_dict.
        """
        graph_db_config = get_graph_config()
        for key, value in config_dict.items():
            if hasattr(graph_db_config, key):
                object.__setattr__(graph_db_config, key, value)
            else:
                raise AttributeError(message=f"'{key}' is not a valid attribute of the config.")

    @staticmethod
    def set_vector_db_config(config_dict: dict):
        """
        Updates the vector db config with values from config_dict.
        """
        vector_db_config = get_vectordb_config()
        for key, value in config_dict.items():
            if hasattr(vector_db_config, key):
                object.__setattr__(vector_db_config, key, value)
            else:
                raise InvalidAttributeError(
                    message=f"'{key}' is not a valid attribute of the config."
                )

    @staticmethod
    def set_vector_db_key(db_key: str):
        vector_db_config = get_vectordb_config()
        vector_db_config.vector_db_key = db_key

    @staticmethod
    def set_vector_db_url(db_url: str):
        vector_db_config = get_vectordb_config()
        vector_db_config.vector_db_url = db_url

    @staticmethod
    def set_graphistry_config(graphistry_config: dict[str, str]):
        base_config = get_base_config()

        if "username" not in graphistry_config or "password" not in graphistry_config:
            raise InvalidValueError(
                message="graphistry_config dictionary must contain 'username' and 'password' keys."
            )

        base_config.graphistry_username = graphistry_config.get("username")
        base_config.graphistry_password = graphistry_config.get("password")
