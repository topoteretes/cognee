"""This module contains the configuration for the graph database."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.shared.data_models import KnowledgeGraph
from cognee.root_dir import get_absolute_path


class GraphConfig(BaseSettings):
    """
    Represents the configuration for a graph system, including parameters for graph file
    storage and database connections.

    Public methods:
    - to_dict
    - to_hashable_dict

    Instance variables:
    - graph_filename
    - graph_database_provider
    - graph_database_url
    - graph_database_username
    - graph_database_password
    - graph_database_port
    - graph_file_path
    - graph_model
    - graph_topology
    - model_config
    """

    graph_filename: str = "cognee_graph.pkl"
    graph_database_provider: str = "NETWORKX"
    graph_database_url: str = ""
    graph_database_username: str = ""
    graph_database_password: str = ""
    graph_database_port: int = 123
    graph_file_path: str = os.path.join(
        os.path.join(get_absolute_path(".cognee_system"), "databases"), graph_filename
    )
    graph_model: object = KnowledgeGraph
    graph_topology: object = KnowledgeGraph
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        """
        Return the configuration as a dictionary.

        This dictionary contains all the configurations related to the graph, which includes
        details for file storage and database connectivity.

        Returns:
        --------

            - dict: A dictionary representation of the configuration settings.
        """
        return {
            "graph_filename": self.graph_filename,
            "graph_database_provider": self.graph_database_provider,
            "graph_database_url": self.graph_database_url,
            "graph_database_username": self.graph_database_username,
            "graph_database_password": self.graph_database_password,
            "graph_database_port": self.graph_database_port,
            "graph_file_path": self.graph_file_path,
            "graph_model": self.graph_model,
            "graph_topology": self.graph_topology,
            "model_config": self.model_config,
        }

    def to_hashable_dict(self) -> dict:
        """
        Return a hashable dictionary with essential database configuration parameters.

        This dictionary excludes certain non-hashable objects and focuses on unique identifiers
        for database configurations.

        Returns:
        --------

            - dict: A dictionary representation of the essential database configuration
              settings.
        """
        return {
            "graph_database_provider": self.graph_database_provider,
            "graph_database_url": self.graph_database_url,
            "graph_database_username": self.graph_database_username,
            "graph_database_password": self.graph_database_password,
            "graph_database_port": self.graph_database_port,
            "graph_file_path": self.graph_file_path,
        }


@lru_cache
def get_graph_config():
    """
    Retrieve the graph configuration. This function utilizes caching to return a singleton
    instance of the GraphConfig class for efficiency.

    It creates and returns a GraphConfig object, which contains various settings related to
    graph configuration.

    Returns:
    --------

        - GraphConfig: A GraphConfig instance containing the graph configuration settings.
    """
    return GraphConfig()
