"""Unit tests for graph-provenance marker detection (COG-5522).

Cover the two behaviors that keep destructive routing safe:
- both marker fields (delete_mode + provenance_version) are required;
- a missing-provenance backend reads as "not graph-provenance", but any *other*
  metadata error propagates so destructive ops fail closed instead of silently
  routing a real graph-provenance graph to the (empty) ledger path.
"""

import pytest

from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
)
from cognee.infrastructure.databases.provenance.markers import (
    mark_graph_provenance_if_empty,
    stores_provenance_in_graph,
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
        GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
        GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
    }


async def test_both_marker_fields_required():
    assert await stores_provenance_in_graph(_FakeGraph(_full_marker())) is True
    # Only delete_mode -> not graph-provenance.
    assert (
        await stores_provenance_in_graph(
            _FakeGraph({GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE})
        )
        is False
    )
    # Only version -> not graph-provenance.
    assert (
        await stores_provenance_in_graph(
            _FakeGraph({GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION})
        )
        is False
    )
    # Wrong version -> not graph-provenance.
    assert (
        await stores_provenance_in_graph(
            _FakeGraph(
                {
                    GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
                    GRAPH_PROVENANCE_VERSION_KEY: "999",
                }
            )
        )
        is False
    )


async def test_missing_provenance_reads_as_not_graph_provenance():
    graph = _FakeGraph(metadata_error=UnsupportedProvenanceCapability())
    assert await stores_provenance_in_graph(graph) is False


async def test_unexpected_metadata_error_fails_closed():
    graph = _FakeGraph(metadata_error=RuntimeError("db down"))
    with pytest.raises(RuntimeError, match="db down"):
        await stores_provenance_in_graph(graph)


async def test_ensure_marks_empty_graph():
    graph = _FakeGraph(empty=True)
    assert await mark_graph_provenance_if_empty(graph) is True
    assert graph.marked_with == _full_marker()


async def test_ensure_skips_nonempty_graph():
    graph = _FakeGraph(empty=False)
    assert await mark_graph_provenance_if_empty(graph) is False
    assert graph.marked_with is None


async def test_ensure_already_graph_provenance_is_idempotent():
    graph = _FakeGraph(_full_marker(), empty=False)
    assert await mark_graph_provenance_if_empty(graph) is True
    assert graph.marked_with is None  # no re-mark


async def test_ensure_returns_false_when_backend_cannot_mark():
    graph = _FakeGraph(empty=True, mark_error=UnsupportedProvenanceCapability())
    assert await mark_graph_provenance_if_empty(graph) is False
