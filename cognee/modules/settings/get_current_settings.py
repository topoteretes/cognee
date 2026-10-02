"""The provider stack every ``Pipeline Run *`` telemetry event carries.

This dict is telemetry's only consumer of the provider configuration, so what
goes in here goes to the warehouse: provider and model names only — URLs are
fingerprinted by ``send_telemetry``'s sanitizer, and nothing here may raise,
because it is computed before a pipeline's first event.
"""

from typing import TypedDict

from cognee.infrastructure.databases.graph import get_graph_config
from cognee.infrastructure.databases.relational.config import get_relational_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.vector.embeddings.config import (
    get_embedding_context_config,
    resolve_embedding_names,
)
from cognee.infrastructure.llm import get_llm_config
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.modules.cognify.config import EXTRACTORS, get_cognify_config, resolve_extractor_name


class LLMConfig(TypedDict):
    model: str
    provider: str


class EmbeddingSettings(TypedDict):
    model: str | None
    provider: str | None


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
    embedding: EmbeddingSettings
    graph_extractor: str
    graph: GraphDBConfig
    vector: VectorDBConfig
    relational: RelationalConfig


def _graph_extractor_setting() -> str:
    """``llm`` / ``gliner_demo`` as cognify would resolve it now, or ``invalid``.

    The same resolution as ``resolve_extractor`` minus its side effects: no
    install check (a missing ``gliner2`` is cognify's error to raise, at cognify
    time) and no notice. A setting outside the known extractors is reported as
    the literal ``invalid`` rather than echoed, so a typo in GRAPH_EXTRACTOR
    cannot put free text into telemetry.
    """
    extractor = resolve_extractor_name(None, get_cognify_config())
    return extractor if extractor in EXTRACTORS else "invalid"


def get_current_settings() -> SettingsDict:
    llm_config = get_llm_config()
    graph_config = get_graph_config()
    vector_config = get_vectordb_config()
    relational_config = get_relational_config()
    # The embedder the engine would build right now: the per-dataset context
    # config when one is set, with the keyless fastembed default applied — the
    # same inputs ``get_embedding_engine`` resolves from.
    embedding_provider, embedding_model = resolve_embedding_names(
        get_embedding_context_config(), get_llm_context_config()
    )

    return {
        "llm": {
            "provider": llm_config.llm_provider,
            "model": llm_config.llm_model,
        },
        "embedding": {
            "provider": embedding_provider,
            "model": embedding_model,
        },
        "graph_extractor": _graph_extractor_setting(),
        "graph": {
            "provider": graph_config.graph_database_provider,
            "url": graph_config.graph_database_url or graph_config.graph_file_path,
        },
        "vector": {
            "provider": vector_config.vector_db_provider,
            "url": vector_config.vector_db_url,
        },
        "relational": {
            "provider": relational_config.db_provider,
            "url": f"{relational_config.db_host}:{relational_config.db_port}"
            if relational_config.db_host
            else f"{relational_config.db_path}/{relational_config.db_name}",
        },
    }
