"""Tests for extract_relevant_node_ids with max_distance filtering (issue #2720).

These tests validate the max_distance parameter added to
``NodeEdgeVectorSearch.extract_relevant_node_ids`` which prevents the graph
filter from becoming ineffective on small datasets where
``wide_search_top_k`` exceeds the collection size.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock

# --- Lightweight import of NodeEdgeVectorSearch without pulling in all of cognee ---

# Stub out heavy cognee submodules that the real import would trigger
_stub_modules = [
    "cognee.infrastructure.databases.vector",
    "cognee.infrastructure.databases.vector.exceptions",
    "cognee.modules.observability",
    "cognee.shared.logging_utils",
]

for mod_name in _stub_modules:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Provide minimal stubs for imported names
sys.modules["cognee.infrastructure.databases.vector.exceptions"].CollectionNotFoundError = Exception
sys.modules["cognee.infrastructure.databases.vector"].get_vector_engine = lambda: None

# Stub out new_span to be a no-op context manager
class _DummySpan:
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_attribute(self, *a): pass

def _noop_span(name):
    return _DummySpan()

sys.modules["cognee.modules.observability"].new_span = _noop_span
sys.modules["cognee.modules.observability"].COGNEE_VECTOR_COLLECTION = "cognee.vector.collection"

# Stub logging
class _DummyLogger:
    def info(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def warning(self, *a, **kw): pass

sys.modules["cognee.shared.logging_utils"].get_logger = lambda *a, **kw: _DummyLogger()
sys.modules["cognee.shared.logging_utils"].ERROR = 40

# Now import the module under test
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch


def _make_scored_result(node_id: str, score: float):
    """Create a mock ScoredResult-like object."""
    obj = MagicMock()
    obj.id = node_id
    obj.score = score
    return obj


class TestExtractRelevantNodeIds:
    """Tests for extract_relevant_node_ids distance-based filtering."""

    def test_no_filter_returns_all_ids(self):
        """When max_distance is None, all IDs are returned (backward compat)."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = None
        vs.node_distances = {
            "Entity_name": [
                _make_scored_result("id1", 0.5),
                _make_scored_result("id2", 1.8),
                _make_scored_result("id3", 0.9),
            ]
        }
        result = vs.extract_relevant_node_ids(max_distance=None)
        assert set(result) == {"id1", "id2", "id3"}

    def test_filter_removes_high_distance_ids(self):
        """IDs with cosine distance > max_distance should be excluded."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = None
        vs.node_distances = {
            "Entity_name": [
                _make_scored_result("id1", 0.5),   # passes (0.5 <= 1.5)
                _make_scored_result("id2", 1.8),   # filtered out (1.8 > 1.5)
                _make_scored_result("id3", 1.5),   # passes (1.5 <= 1.5, boundary)
            ]
        }
        result = vs.extract_relevant_node_ids(max_distance=1.5)
        assert set(result) == {"id1", "id3"}

    def test_filter_multiple_collections(self):
        """Filtering works across multiple collections."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = None
        vs.node_distances = {
            "Entity_name": [
                _make_scored_result("e1", 0.3),
                _make_scored_result("e2", 1.9),
            ],
            "DocumentChunk_text": [
                _make_scored_result("c1", 1.2),
                _make_scored_result("c2", 1.6),  # filtered out
            ],
        }
        result = vs.extract_relevant_node_ids(max_distance=1.5)
        assert set(result) == {"e1", "c1"}

    def test_batch_mode_returns_empty(self):
        """Batch mode should always return empty list."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = 3  # batch mode
        vs.node_distances = {
            "Entity_name": [_make_scored_result("id1", 0.5)]
        }
        result = vs.extract_relevant_node_ids(max_distance=1.5)
        assert result == []

    def test_empty_distances(self):
        """Empty node_distances should return empty list."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = None
        vs.node_distances = {}
        result = vs.extract_relevant_node_ids(max_distance=1.5)
        assert result == []

    def test_all_ids_filtered_out(self):
        """When all IDs are above threshold, returns empty list."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = None
        vs.node_distances = {
            "Entity_name": [
                _make_scored_result("id1", 1.8),
                _make_scored_result("id2", 1.9),
            ]
        }
        result = vs.extract_relevant_node_ids(max_distance=1.5)
        assert result == []

    def test_default_threshold_is_reasonable(self):
        """The default threshold of 1.5 should filter out clearly irrelevant results
        while keeping most relevant ones."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = None
        # Simulate a small dataset where wide_search_top_k=100 returns everything
        vs.node_distances = {
            "Entity_name": [
                _make_scored_result(f"entity_{i}", 0.3 + i * 0.1)
                for i in range(20)
            ],
            "DocumentChunk_text": [
                _make_scored_result(f"chunk_{i}", 0.5 + i * 0.1)
                for i in range(44)
            ],
        }
        result_no_filter = vs.extract_relevant_node_ids(max_distance=None)
        result_with_filter = vs.extract_relevant_node_ids(max_distance=1.5)
        assert len(result_no_filter) == 64
        assert len(result_with_filter) < 64
        assert len(result_with_filter) > 0

    def test_node_without_id_is_skipped(self):
        """Nodes with None id should be skipped gracefully."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = None
        node_no_id = MagicMock()
        node_no_id.id = None
        node_no_id.score = 0.5
        vs.node_distances = {
            "Entity_name": [
                _make_scored_result("id1", 0.5),
                node_no_id,
            ]
        }
        result = vs.extract_relevant_node_ids(max_distance=1.5)
        assert result == ["id1"]

    def test_node_without_score_included_when_filtering(self):
        """Nodes with no score attribute should be included (fail-open) when filtering."""
        vs = NodeEdgeVectorSearch.__new__(NodeEdgeVectorSearch)
        vs.query_list_length = None
        node_no_score = MagicMock()
        node_no_score.id = "id_no_score"
        del node_no_score.score
        vs.node_distances = {
            "Entity_name": [
                _make_scored_result("id1", 0.5),
                node_no_score,
            ]
        }
        result = vs.extract_relevant_node_ids(max_distance=1.5)
        assert set(result) == {"id1", "id_no_score"}
