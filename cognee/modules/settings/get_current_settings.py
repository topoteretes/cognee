from typing import TypedDict
from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.databases.graph import get_graph_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.relational.config import get_relational_config


class LLMConfig(TypedDict):
    model: str
    provider: str


class VectorDBConfig(TypedDict):
    url: str
    provider: str


class GraphDBConfig(TypedDict):
    url: str
    provider: str


class RelationalConfig(TypedDict):
    url: str
    provider: str


class SettingsDict(TypedDict):
    llm: LLMConfig
    graph: GraphDBConfig
    vector: VectorDBConfig
    relational: RelationalConfig


def get_current_settings() -> SettingsDict:
    llm_config = get_llm_config()
    graph_config = get_graph_config()
    vector_config = get_vectordb_config()
    relational_config = get_relational_config()

    return dict(
        llm={
            "provider": llm_config.llm_provider,
            "model": llm_config.llm_model,
        },
        graph={
            "provider": graph_config.graph_database_provider,
            "url": graph_config.graph_database_url or graph_config.graph_file_path,
        },
        vector={
            "provider": vector_config.vector_db_provider,
            "url": vector_config.vector_db_url,
        },
        relational={
            "provider": relational_config.db_provider,
            "url": f"{relational_config.db_host}:{relational_config.db_port}"
            if relational_config.db_host
            else f"{relational_config.db_path}/{relational_config.db_name}",
        },
    )
