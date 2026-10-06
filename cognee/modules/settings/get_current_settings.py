"""The provider stack every ``Pipeline Run *`` telemetry event carries.

This dict is telemetry's only consumer of the provider configuration, so what
goes in here goes to the warehouse: provider and model names only — URLs are
fingerprinted by ``send_telemetry``'s sanitizer, and nothing here may raise,
because it is computed before a pipeline's first event.
"""

import re
from typing import TypedDict

from cognee.infrastructure.databases.graph import get_graph_config
from cognee.infrastructure.databases.relational.config import get_relational_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.vector.embeddings.config import (
    get_embedding_context_config,
    resolve_embedding_names,
)
from cognee.infrastructure.llm.config import (
    get_llm_config,
    get_llm_context_config,
    resolve_structured_output_framework,
)
from cognee.modules.cognify.config import EXTRACTORS
from cognee.modules.preflight import llm_available
from cognee.shared.utils import telemetry_model_label


class LLMConfig(TypedDict):
    model: str
    provider: str
    configured: bool
    structured_output: str
    instructor_mode: str


# An instructor mode is an identifier such as ``json_schema_mode`` or ``tool_call``.
_INSTRUCTOR_MODE = re.compile(r"^[a-z_]{1,32}$")


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


def _instructor_mode_setting(llm_config) -> str:
    """The instructor mode in force: the identifier set, ``default`` when unset, else ``invalid``."""
    mode = getattr(llm_config, "llm_instructor_mode", None)
    if mode is None:
        return "unknown"
    if not isinstance(mode, str) or not mode:
        return "default"
    mode = mode.lower()
    return mode if _INSTRUCTOR_MODE.match(mode) else "invalid"


def get_current_settings(
    *, graph_extractor: str | None = None, llm_config=None, embedding_config=None
) -> SettingsDict:
    # Explicit configs let the start event describe the pending run before
    # its database context is entered. Otherwise use the active context.
    llm_config = llm_config if llm_config is not None else get_llm_context_config()
    graph_config = get_graph_config()
    vector_config = get_vectordb_config()
    relational_config = get_relational_config()
    # The embedder the engine would build right now: the per-dataset context
    # config when one is set, with the keyless fastembed default applied — the
    # same inputs ``get_embedding_engine`` resolves from.
    embedding_provider, embedding_model = resolve_embedding_names(
        embedding_config if embedding_config is not None else get_embedding_context_config(),
        llm_config,
    )

    return {
        "llm": {
            "provider": llm_config.llm_provider,
            # A model that is a filesystem path leaves as "local_path", never the path.
            "model": telemetry_model_label(llm_config.llm_model),
            # provider/model are the configured values even when no key is set, so
            # a keyless install reports the unused default. ``configured`` says
            # whether that LLM is usable: the rule recall() and the keyless path use,
            # applied to the same per-call config as the rest of this payload.
            "configured": llm_available(llm_config),
            # How structured output is obtained, so a schema rejection can be
            # told apart by path (native response_format / instructor / BAML).
            # The gateway dispatches using the process setting, even when model
            # and credentials come from a per-call config.
            "structured_output": resolve_structured_output_framework(get_llm_config()),
            "instructor_mode": _instructor_mode_setting(llm_config),
        },
        "embedding": {
            "provider": embedding_provider,
            "model": telemetry_model_label(embedding_model),
        },
        # cognify has already resolved argument/env/default precedence. Custom
        # pipelines need not extract a graph at all; do not invent a selection.
        "graph_extractor": (
            "unknown"
            if graph_extractor is None
            else graph_extractor
            if graph_extractor in EXTRACTORS
            else "invalid"
        ),
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
