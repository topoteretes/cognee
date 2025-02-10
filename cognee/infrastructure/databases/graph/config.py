"""This module contains the configuration for the graph database."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.shared.data_models import KnowledgeGraph
from cognee.root_dir import get_absolute_path


class GraphConfig(BaseSettings):
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
    return GraphConfig()
