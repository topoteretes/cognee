"""This module contains the configuration for the graph database."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import pydantic
from pydantic import Field
from cognee.base_config import get_base_config
from cognee.root_dir import ensure_absolute_path
from cognee.shared.data_models import KnowledgeGraph


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

    # Using Field we are able to dynamically load current GRAPH_DATABASE_PROVIDER value in the model validator part
    # and determine default graph db file and path based on this parameter if no values are provided
    graph_database_provider: str = Field("kuzu", env="GRAPH_DATABASE_PROVIDER")

    graph_database_url: str = ""
    graph_database_name: str = ""
    graph_database_username: str = ""
    graph_database_password: str = ""
    graph_database_port: int = 123
    graph_file_path: str = ""
    graph_filename: str = ""
    graph_model: object = KnowledgeGraph
    graph_topology: object = KnowledgeGraph
    model_config = SettingsConfigDict(env_file=".env", extra="allow", populate_by_name=True)

    # Model validator updates graph_filename and path dynamically after class creation based on current database provider
    # If no specific graph_filename or path are provided
    @pydantic.model_validator(mode="after")
    def fill_derived(self):
        provider = self.graph_database_provider.lower()
        base_config = get_base_config()

        # Set default filename if no filename is provided
        if not self.graph_filename:
            self.graph_filename = f"cognee_graph_{provider}"

        # Handle graph file path
        if self.graph_file_path:
            # Check if absolute path is provided
            self.graph_file_path = ensure_absolute_path(
                os.path.join(self.graph_file_path, self.graph_filename)
            )
        else:
            # Default path
            databases_directory_path = os.path.join(base_config.system_root_directory, "databases")
            self.graph_file_path = os.path.join(databases_directory_path, self.graph_filename)

        return self

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
            "graph_database_name": self.graph_database_name,
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


def get_graph_context_config():
    """This function will get the appropriate graph db config based on async context.
    This allows the use of multiple graph databases for different threads, async tasks and parallelization
    """
    from cognee.context_global_variables import graph_db_config

    if graph_db_config.get():
        return graph_db_config.get()
    return get_graph_config().to_hashable_dict()
