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
from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.modules.cognify.config import EXTRACTORS, get_cognify_config, resolve_extractor_name
from cognee.modules.preflight import llm_available
from cognee.shared.utils import telemetry_model_label


class LLMConfig(TypedDict):
    model: str
    provider: str
    configured: bool
    structured_output: str
    instructor_mode: str


# The structured-output frameworks cognee ships (STRUCTURED_OUTPUT_FRAMEWORK).
STRUCTURED_OUTPUT_FRAMEWORKS = ("litellm_native", "instructor", "baml")
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


def _structured_output_setting(llm_config) -> str:
    """Which structured-output path LLM calls take: a closed value, or ``invalid``.

    A graph extraction that fails on a schema the provider rejects looks the
    same as any other LLM error unless the event says whether the call went
    through litellm's native response_format, instructor, or BAML. An unknown
    setting is reported as ``invalid``, never echoed; a config without the
    attribute (an out-of-tree config object) as ``unknown``.
    """
    framework = getattr(llm_config, "structured_output_framework", None)
    if not isinstance(framework, str) or not framework:
        return "unknown"
    framework = framework.lower()
    return framework if framework in STRUCTURED_OUTPUT_FRAMEWORKS else "invalid"


def _instructor_mode_setting(llm_config) -> str:
    """The instructor mode in force: the identifier set, ``default`` when unset, else ``invalid``."""
    mode = getattr(llm_config, "llm_instructor_mode", None)
    if mode is None:
        return "unknown"
    if not isinstance(mode, str) or not mode:
        return "default"
    mode = mode.lower()
    return mode if _INSTRUCTOR_MODE.match(mode) else "invalid"


def get_current_settings() -> SettingsDict:
    # The context config when a per-call LLMConfig is set, else the process one:
    # the same resolution the embedding half below uses, so one event never
    # describes two configurations.
    llm_config = get_llm_context_config()
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
            # A model that is a filesystem path leaves as "local_path", never the path.
            "model": telemetry_model_label(llm_config.llm_model),
            # provider/model are the configured values even when no key is set, so
            # a keyless install reports the unused default. ``configured`` says
            # whether that LLM is usable: the rule recall() and the keyless path use,
            # applied to the same per-call config as the rest of this payload.
            "configured": llm_available(llm_config),
            # How structured output is obtained, so a schema rejection can be
            # told apart by path (native response_format / instructor / BAML).
            "structured_output": _structured_output_setting(llm_config),
            "instructor_mode": _instructor_mode_setting(llm_config),
        },
        "embedding": {
            "provider": embedding_provider,
            "model": telemetry_model_label(embedding_model),
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
