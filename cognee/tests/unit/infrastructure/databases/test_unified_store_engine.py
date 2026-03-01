"""Unit tests for the UnifiedStoreEngine and EngineCapability."""

import pytest

from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine


# ---------------------------------------------------------------------------
# EngineCapability flag tests
# ---------------------------------------------------------------------------


class TestEngineCapability:
    def test_none_is_falsy(self):
        assert not EngineCapability.NONE

    def test_single_flags(self):
        assert EngineCapability.GRAPH
        assert EngineCapability.VECTOR
        assert EngineCapability.HYBRID_WRITE
        assert EngineCapability.HYBRID_SEARCH

    def test_composition(self):
        combo = EngineCapability.GRAPH | EngineCapability.VECTOR
        assert combo & EngineCapability.GRAPH
        assert combo & EngineCapability.VECTOR
        assert not (combo & EngineCapability.HYBRID_WRITE)

    def test_full_hybrid(self):
        full = (
            EngineCapability.GRAPH
            | EngineCapability.VECTOR
            | EngineCapability.HYBRID_WRITE
            | EngineCapability.HYBRID_SEARCH
        )
        assert full & EngineCapability.GRAPH
        assert full & EngineCapability.VECTOR
        assert full & EngineCapability.HYBRID_WRITE
        assert full & EngineCapability.HYBRID_SEARCH


# ---------------------------------------------------------------------------
# UnifiedStoreEngine tests with separate engines
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal stand-in for GraphDBInterface."""

    pass


class _FakeVector:
    """Minimal stand-in for VectorDBInterface."""

    pass


class TestUnifiedStoreEngineSeparateBackends:
    def setup_method(self):
        self.graph = _FakeGraph()
        self.vector = _FakeVector()
        self.engine = UnifiedStoreEngine(
            graph_engine=self.graph,
            vector_engine=self.vector,
            capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
        )

    def test_capabilities(self):
        assert self.engine.has_capability(EngineCapability.GRAPH)
        assert self.engine.has_capability(EngineCapability.VECTOR)
        assert not self.engine.has_capability(EngineCapability.HYBRID_WRITE)
        assert not self.engine.has_capability(EngineCapability.HYBRID_SEARCH)

    def test_graph_property(self):
        assert self.engine.graph is self.graph

    def test_vector_property(self):
        assert self.engine.vector is self.vector

    def test_is_hybrid_false(self):
        assert not self.engine.is_hybrid

    def test_is_same_backend_false(self):
        assert not self.engine.is_same_backend


# ---------------------------------------------------------------------------
# UnifiedStoreEngine tests with a single hybrid engine
# ---------------------------------------------------------------------------


class _FakeHybridAdapter:
    """Simulates a backend that implements both graph and vector interfaces."""

    pass


class TestUnifiedStoreEngineHybridBackend:
    def setup_method(self):
        self.adapter = _FakeHybridAdapter()
        self.engine = UnifiedStoreEngine(
            graph_engine=self.adapter,
            vector_engine=self.adapter,
            capabilities=(
                EngineCapability.GRAPH
                | EngineCapability.VECTOR
                | EngineCapability.HYBRID_WRITE
                | EngineCapability.HYBRID_SEARCH
            ),
        )

    def test_capabilities(self):
        assert self.engine.has_capability(EngineCapability.GRAPH)
        assert self.engine.has_capability(EngineCapability.VECTOR)
        assert self.engine.has_capability(EngineCapability.HYBRID_WRITE)
        assert self.engine.has_capability(EngineCapability.HYBRID_SEARCH)

    def test_graph_and_vector_are_same_object(self):
        assert self.engine.graph is self.engine.vector

    def test_is_hybrid_true(self):
        assert self.engine.is_hybrid

    def test_is_same_backend_true(self):
        assert self.engine.is_same_backend


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestUnifiedStoreEngineErrors:
    def test_graph_raises_without_capability(self):
        engine = UnifiedStoreEngine(capabilities=EngineCapability.NONE)
        with pytest.raises(RuntimeError, match="no GRAPH capability"):
            _ = engine.graph

    def test_vector_raises_without_capability(self):
        engine = UnifiedStoreEngine(capabilities=EngineCapability.NONE)
        with pytest.raises(RuntimeError, match="no VECTOR capability"):
            _ = engine.vector

    def test_graph_raises_when_none_despite_capability(self):
        engine = UnifiedStoreEngine(
            graph_engine=None,
            capabilities=EngineCapability.GRAPH,
        )
        with pytest.raises(RuntimeError, match="no GRAPH capability"):
            _ = engine.graph

    def test_vector_raises_when_none_despite_capability(self):
        engine = UnifiedStoreEngine(
            vector_engine=None,
            capabilities=EngineCapability.VECTOR,
        )
        with pytest.raises(RuntimeError, match="no VECTOR capability"):
            _ = engine.vector

    def test_no_capabilities_is_not_hybrid(self):
        engine = UnifiedStoreEngine(capabilities=EngineCapability.NONE)
        assert not engine.is_hybrid
        assert not engine.is_same_backend
