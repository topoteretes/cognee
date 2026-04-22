from cognee.infrastructure.databases.graph.config import get_graph_context_config
from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine

from .capabilities import EngineCapability
from .unified_store_engine import UnifiedStoreEngine

HYBRID_PROVIDERS = {}
UNIFIED_PROVIDERS = {"pghybrid"}


def _is_hybrid_provider(graph_config: dict, vector_config: dict) -> bool:
    import os

    # USE_UNIFIED_PROVIDER flag overrides both graph and vector providers
    unified = os.environ.get("USE_UNIFIED_PROVIDER", "")
    if unified and unified in UNIFIED_PROVIDERS:
        return True

    # Original logic for neptune and future paired providers
    graph_provider = graph_config.get("graph_database_provider", "")
    vector_provider = vector_config.get("vector_db_provider", "")
    return graph_provider in HYBRID_PROVIDERS and graph_provider == vector_provider


async def _create_hybrid_adapter(graph_config: dict, vector_config: dict):
    """Create a single adapter instance for a hybrid backend.

    For pghybrid, reuses the cached PGVectorAdapter from get_vector_engine()
    (requires VECTOR_DB_PROVIDER=pgvector) so that metadata caches persist
    across calls.
    """
    import os

    unified_provider = os.environ.get("USE_UNIFIED_PROVIDER", "")
    provider = unified_provider or graph_config["graph_database_provider"]

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

    if provider == "pghybrid":
        from cognee.infrastructure.databases.hybrid.postgres.adapter import (
            PostgresHybridAdapter,
        )
        from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter
        from cognee.infrastructure.databases.relational.get_relational_engine import (
            get_relational_engine,
        )

        # Graph adapter gets its own engine from the relational connection string
        graph_adapter = PostgresAdapter(connection_string=get_relational_engine().db_uri)

        # Vector adapter: reuse the cached PGVectorAdapter from the
        # vector engine factory. This requires VECTOR_DB_PROVIDER=pgvector.
        vector_adapter = get_vector_engine()

        return PostgresHybridAdapter(
            graph_adapter=graph_adapter,
            vector_adapter=vector_adapter,
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
        adapter = await _create_hybrid_adapter(graph_config, vector_config)
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
