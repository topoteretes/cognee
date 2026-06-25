"""Unit tests for graph-native marker detection (COG-5522).

Cover the two behaviors that keep destructive routing safe:
- both marker fields (delete_mode + provenance_version) are required;
- a missing-provenance backend reads as "not graph-native", but any *other*
  metadata error propagates so destructive ops fail closed instead of silently
  routing a real graph-native graph to the (empty) ledger path.
"""

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_NATIVE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
)
from cognee.infrastructure.databases.provenance.markers import (
    ensure_graph_native_for_new_graph,
    is_graph_native_graph,
)

pytestmark = pytest.mark.asyncio


class _FakeGraph:
    def __init__(self, metadata=None, *, empty=True, metadata_error=None, mark_error=None):
        self._metadata = dict(metadata or {})
        self._empty = empty
        self._metadata_error = metadata_error
        self._mark_error = mark_error
        self.marked_with = None

    async def get_graph_metadata(self):
        if self._metadata_error is not None:
            raise self._metadata_error
        return dict(self._metadata)

    async def set_graph_metadata(self, metadata):
        if self._mark_error is not None:
            raise self._mark_error
        self.marked_with = dict(metadata)
        self._metadata.update(metadata)

    async def is_empty(self):
        return self._empty


def _full_marker():
    return {
        GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_NATIVE,
        GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
    }


async def test_both_marker_fields_required():
    assert await is_graph_native_graph(_FakeGraph(_full_marker())) is True
    # Only delete_mode -> not graph-native.
    assert (
        await is_graph_native_graph(
            _FakeGraph({GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_NATIVE})
        )
        is False
    )
    # Only version -> not graph-native.
    assert (
        await is_graph_native_graph(
            _FakeGraph({GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION})
        )
        is False
    )
    # Wrong version -> not graph-native.
    assert (
        await is_graph_native_graph(
            _FakeGraph(
                {
                    GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_NATIVE,
                    GRAPH_PROVENANCE_VERSION_KEY: "999",
                }
            )
        )
        is False
    )


async def test_missing_provenance_reads_as_not_graph_native():
    graph = _FakeGraph(metadata_error=UnsupportedProvenanceCapability())
    assert await is_graph_native_graph(graph) is False


async def test_unexpected_metadata_error_fails_closed():
    graph = _FakeGraph(metadata_error=RuntimeError("db down"))
    with pytest.raises(RuntimeError, match="db down"):
        await is_graph_native_graph(graph)


async def test_ensure_marks_empty_graph():
    graph = _FakeGraph(empty=True)
    assert await ensure_graph_native_for_new_graph(graph) is True
    assert graph.marked_with == _full_marker()


async def test_ensure_skips_nonempty_graph():
    graph = _FakeGraph(empty=False)
    assert await ensure_graph_native_for_new_graph(graph) is False
    assert graph.marked_with is None


async def test_ensure_already_graph_native_is_idempotent():
    graph = _FakeGraph(_full_marker(), empty=False)
    assert await ensure_graph_native_for_new_graph(graph) is True
    assert graph.marked_with is None  # no re-mark


async def test_ensure_returns_false_when_backend_cannot_mark():
    graph = _FakeGraph(empty=True, mark_error=UnsupportedProvenanceCapability())
    assert await ensure_graph_native_for_new_graph(graph) is False
