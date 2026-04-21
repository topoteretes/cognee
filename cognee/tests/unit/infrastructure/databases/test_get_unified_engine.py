"""Unit tests for get_unified_engine factory."""

import pytest

from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.get_unified_engine import (
    _is_hybrid_provider,
    HYBRID_PROVIDERS,
)


class TestIsHybridProvider:
    # NOTE: This test is disabled temporarily, until we implement Neptune hybrid capabilities
    # def test_both_neptune_analytics_is_hybrid(self):
    #     g = {"graph_database_provider": "neptune_analytics"}
    #     v = {"vector_db_provider": "neptune_analytics"}
    #     assert _is_hybrid_provider(g, v) is True

    def test_kuzu_lancedb_is_not_hybrid(self):
        g = {"graph_database_provider": "kuzu"}
        v = {"vector_db_provider": "lancedb"}
        assert _is_hybrid_provider(g, v) is False

    def test_neptune_graph_only_is_not_hybrid(self):
        g = {"graph_database_provider": "neptune_analytics"}
        v = {"vector_db_provider": "lancedb"}
        assert _is_hybrid_provider(g, v) is False

    def test_mismatched_hybrid_is_not_hybrid(self):
        g = {"graph_database_provider": "neo4j"}
        v = {"vector_db_provider": "neptune_analytics"}
        assert _is_hybrid_provider(g, v) is False

    def test_empty_configs(self):
        assert _is_hybrid_provider({}, {}) is False

    # NOTE: This test is disabled temporarily, until we implement Neptune hybrid capabilities
    # def test_hybrid_providers_registry(self):
    #     assert "neptune_analytics" in HYBRID_PROVIDERS


@pytest.mark.asyncio
async def test_get_unified_engine_returns_separate_for_defaults():
    """With default kuzu+lancedb config, get_unified_engine returns separate backends."""
    from cognee.infrastructure.databases.unified import get_unified_engine

    engine = await get_unified_engine()

    assert engine.has_capability(EngineCapability.GRAPH)
    assert engine.has_capability(EngineCapability.VECTOR)
    assert not engine.has_capability(EngineCapability.HYBRID_WRITE)
    assert not engine.has_capability(EngineCapability.HYBRID_SEARCH)
    assert not engine.is_hybrid
    assert not engine.is_same_backend
    assert engine.graph is not None
    assert engine.vector is not None
