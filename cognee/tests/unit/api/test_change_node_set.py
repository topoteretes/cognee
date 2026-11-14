"""Unit tests for ``cognee.change_node_set``.

The move touches three stores; these tests drive the orchestration with mocked
engines and assert the contract from issue #6628: the leaving set's tags and
edges are stripped (scoped to the document's nodes), the new set's NodeSet nodes
and edges are added, the chunk vectors are repointed via ``update_payload`` (which
preserves the vector, so no re-embed), and no cognify/LLM path is taken.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import cognee.api.v1.change_node_set.change_node_set as module
from cognee.api.v1.change_node_set import change_node_set
from cognee.modules.engine.models.node_set import InvalidNodeSetError

MODULE = "cognee.api.v1.change_node_set.change_node_set"


def _graph_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.remove_belongs_to_set_tags = AsyncMock()
    engine.add_node = AsyncMock()
    engine.add_edge = AsyncMock()
    engine.update_node = AsyncMock()
    return engine


def _vector_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.update_payload = AsyncMock()
    return engine


@pytest.fixture
def wired():
    """Patch every boundary of change_node_set and hand the mocks back.

    _document_chunk_ids and _publish_node_set are stubbed so the test exercises
    the orchestration, not graph connection-shape parsing or the SQLAlchemy write
    (the relational write mirrors the already-tested publish_updated_data path).
    """
    user = SimpleNamespace(id=uuid4())
    document_id = uuid4()
    dataset_id = uuid4()
    graph = _graph_engine()
    vector = _vector_engine()

    with (
        patch(f"{MODULE}.get_default_user", AsyncMock(return_value=user)),
        patch(f"{MODULE}.get_authorized_dataset", AsyncMock(return_value=SimpleNamespace())),
        patch(f"{MODULE}.resolve_data_id", AsyncMock(return_value=document_id)),
        patch(
            f"{MODULE}.get_data",
            AsyncMock(return_value=SimpleNamespace(node_set=json.dumps(["alpha"]))),
        ),
        patch(f"{MODULE}.get_graph_engine", AsyncMock(return_value=graph)),
        patch(f"{MODULE}.get_vector_engine_async", AsyncMock(return_value=vector)),
        patch(f"{MODULE}._document_chunk_ids", AsyncMock(return_value=["chunk-1", "chunk-2"])),
        patch(f"{MODULE}._publish_node_set", AsyncMock()) as publish,
    ):
        yield SimpleNamespace(
            user=user,
            document_id=document_id,
            dataset_id=dataset_id,
            graph=graph,
            vector=vector,
            publish=publish,
        )


@pytest.mark.asyncio
async def test_move_replaces_tags_edges_and_payloads(wired):
    result = await change_node_set(
        data_id=wired.document_id,
        dataset_id=wired.dataset_id,
        node_set=["beta"],
    )

    node_ids = [str(wired.document_id), "chunk-1", "chunk-2"]

    # The leaving set is stripped, scoped to exactly the document's nodes.
    wired.graph.remove_belongs_to_set_tags.assert_awaited_once_with(["alpha"], node_ids=node_ids)

    # The new NodeSet node exists and every node gets a belongs_to_set edge to it.
    wired.graph.add_node.assert_awaited_once()
    edge_targets = {call.args[0] for call in wired.graph.add_edge.await_args_list}
    assert edge_targets == set(node_ids)
    for call in wired.graph.add_edge.await_args_list:
        assert call.args[2] == "belongs_to_set"

    # The tag property is set to exactly the new set on every node.
    assert wired.graph.update_node.await_count == len(node_ids)
    for call in wired.graph.update_node.await_args_list:
        assert call.args[1]["belongs_to_set"] == ["beta"]

    # Chunk vectors are repointed in place — update_payload preserves the vector,
    # so no embedding call happens and the chunk ids are unchanged.
    wired.vector.update_payload.assert_awaited_once()
    collection, updates = wired.vector.update_payload.await_args.args
    assert collection == "DocumentChunk_text"
    assert set(updates) == {"chunk-1", "chunk-2"}
    assert updates["chunk-1"]["belongs_to_set"] == ["beta"]

    # The relational node set moves too.
    wired.publish.assert_awaited_once_with(wired.document_id, ["beta"])

    assert result["status"] == "retagged"
    assert result["node_set"] == ["beta"]
    assert result["previous_node_set"] == ["alpha"]
    assert result["chunks_retagged"] == 2


@pytest.mark.asyncio
async def test_clearing_node_set_removes_old_without_adding(wired):
    await change_node_set(
        data_id=wired.document_id,
        dataset_id=wired.dataset_id,
        node_set=[],
    )

    wired.graph.remove_belongs_to_set_tags.assert_awaited_once()
    # Clearing adds no NodeSet node or edge.
    wired.graph.add_node.assert_not_awaited()
    wired.graph.add_edge.assert_not_awaited()
    _, updates = wired.vector.update_payload.await_args.args
    assert updates["chunk-1"]["belongs_to_set"] == []
    wired.publish.assert_awaited_once_with(wired.document_id, [])


@pytest.mark.asyncio
async def test_no_chunks_still_moves_relational(wired):
    with patch(f"{MODULE}._document_chunk_ids", AsyncMock(return_value=[])):
        result = await change_node_set(
            data_id=wired.document_id,
            dataset_id=wired.dataset_id,
            node_set=["beta"],
        )

    # Nothing to repoint in the vector store when the document has no chunks.
    wired.vector.update_payload.assert_not_awaited()
    # The relational move still lands so a later cognify tags it with the new set.
    wired.publish.assert_awaited_once_with(wired.document_id, ["beta"])
    assert result["chunks_retagged"] == 0


@pytest.mark.asyncio
async def test_malformed_node_set_rejected_before_any_write(wired):
    with pytest.raises(InvalidNodeSetError):
        await change_node_set(
            data_id=wired.document_id,
            dataset_id=wired.dataset_id,
            node_set="beta",  # bare string, not a list of names
        )

    wired.graph.remove_belongs_to_set_tags.assert_not_awaited()
    wired.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_document_raises(wired):
    from cognee.api.v1.exceptions import UpdateTargetNotFoundError

    with patch(f"{MODULE}.resolve_data_id", AsyncMock(return_value=None)):
        with pytest.raises(UpdateTargetNotFoundError):
            await change_node_set(
                data_id=wired.document_id,
                dataset_id=wired.dataset_id,
                node_set=["beta"],
            )


def test_change_node_set_is_exported():
    import cognee

    assert cognee.change_node_set is module.change_node_set
