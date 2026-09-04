"""graph_only threading through the migration import path.

``remember(source, index_vectors=False)`` must reach ``add_data_points`` as
``graph_only=True`` so a bundled archive restores without a vector engine,
an embedding provider, or an API key (the `cognee-cli demo` contract).
"""

from unittest.mock import AsyncMock, patch

import pytest

import importlib

from cognee.modules.migration.cogx import COGXEntity, COGXFact, COGXRawNode
from cognee.modules.migration.loader import store_imported_graph, stream_graph_from_source
from cognee.modules.migration.sources.base import MemorySource

# Bind the module object itself: the storage package's __init__ re-exports the
# add_data_points *function* under the same name, shadowing the module as a
# package attribute. Both mock.patch string targets (getattr walk on Python
# 3.10) and ``import ... as`` (parent-attribute binding) resolve to that
# function, so use importlib, which always returns the module from sys.modules.
add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")


class _StubSource(MemorySource):
    source_system = "cognee"
    replayable = True

    def __init__(self, records, mode="preserve"):
        super().__init__(mode=mode)
        self._records = records

    async def records(self):
        for record in self._records:
            yield record


def _sample_records():
    return [
        COGXEntity(external_id="alice", name="Alice", entity_type="Person"),
        COGXEntity(external_id="anthropic", name="Anthropic", entity_type="Organization"),
        COGXFact(
            external_id="f1",
            subject_ref="alice",
            predicate="works_at",
            object_ref="anthropic",
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_only", [True, False])
async def test_stream_graph_from_source_threads_graph_only(graph_only):
    add_mock = AsyncMock(return_value=[])
    stats = {"graph_nodes": 0, "graph_edges": 0, "skipped_facts": 0, "deduped_edges": 0}

    with patch.object(add_data_points_module, "add_data_points", add_mock):
        await stream_graph_from_source(_StubSource(_sample_records()), stats, graph_only=graph_only)

    assert add_mock.await_count >= 1
    for call in add_mock.await_args_list:
        assert call.kwargs["graph_only"] is graph_only
    assert stats["graph_nodes"] > 0
    assert stats["graph_edges"] == 1
    assert stats["skipped_facts"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_only", [True, False])
async def test_store_imported_graph_threads_graph_only(graph_only):
    add_mock = AsyncMock(return_value=[])
    batch = {"nodes": [], "edges": []}

    with patch.object(add_data_points_module, "add_data_points", add_mock):
        await store_imported_graph([batch], graph_only=graph_only)

    assert add_mock.await_count == 1
    assert add_mock.await_args.kwargs["graph_only"] is graph_only


@pytest.mark.asyncio
async def test_remember_maps_index_vectors_to_graph_only():
    """The public kwarg is index_vectors (same as the code route); the
    migration layer receives it inverted as graph_only."""
    from cognee.api.v1.remember import remember

    import_mock = AsyncMock(return_value="sentinel")
    source = _StubSource([COGXRawNode(properties={})])

    with patch("cognee.modules.migration.import_source.import_memory_source", import_mock):
        result = await remember(source, index_vectors=False)

    assert result == "sentinel"
    assert import_mock.await_args.kwargs["graph_only"] is True

    import_mock.reset_mock()
    with patch("cognee.modules.migration.import_source.import_memory_source", import_mock):
        await remember(source)
    assert import_mock.await_args.kwargs["graph_only"] is False
