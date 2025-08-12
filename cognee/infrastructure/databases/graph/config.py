"""This module contains the configuration for the graph database."""

import os
import pydantic
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.base_config import get_base_config
from cognee.shared.data_models import KnowledgeGraph


class GraphConfig(BaseSettings):
    """
    Represents the configuration for a graph system, including parameters for graph file
    storage and database connections.

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

    graph_database_provider: str = "kuzu"
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
    def fill_derived(cls, values):
        provider = values.graph_database_provider.lower()

        # Set default filename if no filename is provided
        if not values.graph_filename:
            values.graph_filename = f"cognee_graph_{provider}"

        # Set file path based on graph database provider if no file path is provided
        if not values.graph_file_path:
            base_config = get_base_config()

            databases_directory_path = os.path.join(base_config.system_root_directory, "databases")
            values.graph_file_path = os.path.join(databases_directory_path, values.graph_filename)

        return values


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
    context_config = get_graph_context_config()

    if context_config:
        return context_config

    return GraphConfig()


def get_graph_context_config() -> Optional[GraphConfig]:
    """This function will get the appropriate graph db config based on async context.
    This allows the use of multiple graph databases for different threads, async tasks and parallelization
    """
    from cognee.context_global_variables import graph_db_config

    return graph_db_config.get()
