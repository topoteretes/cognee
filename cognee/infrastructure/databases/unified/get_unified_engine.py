from cognee.infrastructure.databases.graph.config import get_graph_context_config
from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine

from .capabilities import EngineCapability
from .unified_store_engine import UnifiedStoreEngine

HYBRID_PROVIDERS = {"neptune_analytics"}


def _is_hybrid_provider(graph_config: dict, vector_config: dict) -> bool:
    graph_provider = graph_config.get("graph_database_provider", "")
    vector_provider = vector_config.get("vector_db_provider", "")
    return graph_provider in HYBRID_PROVIDERS and graph_provider == vector_provider


def _create_hybrid_adapter(graph_config: dict, vector_config: dict):
    """Create a single adapter instance for a hybrid backend."""
    provider = graph_config["graph_database_provider"]

    if provider == "neptune_analytics":
        from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
            NeptuneAnalyticsAdapter,
            NEPTUNE_ANALYTICS_ENDPOINT_URL,
        )
        from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine

        graph_url = graph_config.get("graph_database_url", "")
        if not graph_url:
            raise EnvironmentError("Missing Neptune endpoint.")

        if not graph_url.startswith(NEPTUNE_ANALYTICS_ENDPOINT_URL):
            raise ValueError(
                f"Neptune endpoint must have the format "
                f"'{NEPTUNE_ANALYTICS_ENDPOINT_URL}<GRAPH_ID>'"
            )

        graph_identifier = graph_url.replace(NEPTUNE_ANALYTICS_ENDPOINT_URL, "")
        embedding_engine = get_embedding_engine()

        return NeptuneAnalyticsAdapter(
            graph_id=graph_identifier,
            embedding_engine=embedding_engine,
        )

    raise EnvironmentError(f"Unsupported hybrid provider: {provider}")


async def get_unified_engine() -> UnifiedStoreEngine:
    """Build a UnifiedStoreEngine for the current async context.

    - Reads the same context variables as get_graph_engine / get_vector_engine
      so multi-tenant routing works identically.
    - Detects hybrid providers (where graph and vector share a backend) and
      creates a single adapter instance with HYBRID_* capabilities.
    - For separate backends, delegates to the existing cached factories.

    This function is NOT cached itself because it must respect per-request
    ContextVar values.  The underlying engine factories (get_graph_engine,
    get_vector_engine) do their own caching.
    """
    graph_config = get_graph_context_config()
    vector_config = get_vectordb_context_config()

    if _is_hybrid_provider(graph_config, vector_config):
        adapter = _create_hybrid_adapter(graph_config, vector_config)
        if hasattr(adapter, "initialize"):
            await adapter.initialize()
        return UnifiedStoreEngine(
            graph_engine=adapter,
            vector_engine=adapter,
            capabilities=(
                EngineCapability.GRAPH
                | EngineCapability.VECTOR
                | EngineCapability.HYBRID_WRITE
                | EngineCapability.HYBRID_SEARCH
            ),
        )

    graph_engine = await get_graph_engine()
    vector_engine = get_vector_engine()

    return UnifiedStoreEngine(
        graph_engine=graph_engine,
        vector_engine=vector_engine,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )
