"""Incremental writes must remain searchable by the document's node sets, and
node-set-only updates must move a document without reprocessing it."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.chunking.chunk_policy import ChunkPlan
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types.TextDocument import TextDocument
from cognee.tasks.documents.classify_documents import update_node_set

incremental = importlib.import_module("cognee.api.v1.update.incremental")


def _document(tagged=True):
    document = TextDocument(
        id=uuid4(),
        title="brief.md",
        name="brief",
        raw_data_location="/tmp/brief.md",
        mime_type="text/markdown",
        external_metadata='{"node_set":["workflow:meeting-actions","kind:brief"]}'
        if tagged
        else "{}",
    )
    update_node_set(document)
    return document


@pytest.mark.asyncio
@pytest.mark.parametrize("tagged", [True, False])
async def test_fresh_chunks_reach_extraction_with_document_membership(monkeypatch, tagged):
    document = _document(tagged)
    fresh = [
        DocumentChunk(
            id=uuid4(),
            text=f"Changed action {i}",
            chunk_size=3,
            chunk_index=i,
            cut_type="paragraph_end",
            is_part_of=document,
        )
        for i in range(2)
    ]
    observed = []

    async def extract(batch, **kwargs):
        observed.extend(batch)
        assert kwargs["summary_method"] == config.summary_method
        for chunk in batch:
            assert chunk.belongs_to_set == document.belongs_to_set
            assert chunk.source_node_set == document.source_node_set
        return []

    config = SimpleNamespace(
        chunks_per_batch=1,
        triplet_embedding=False,
        summary_method="llm",
    )
    publish = AsyncMock()
    monkeypatch.setattr(incremental, "get_cognify_config", lambda: config)
    monkeypatch.setattr(incremental, "_resolve_extraction_config", lambda: None)
    monkeypatch.setattr(incremental, "extract_graph_and_summarize", extract)
    monkeypatch.setattr(incremental, "add_data_points", AsyncMock())
    monkeypatch.setattr(incremental, "publish_updated_data", publish)
    result = await incremental._write_and_publish(
        {
            "staged": object(),
            "document": document,
            "stored_chunks": [],
            "plan": ChunkPlan(fresh=fresh),
        },
        document.id,
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4()),
        None,
        object,
        None,
        uuid4(),
    )
    assert observed == fresh
    assert result["added_chunks"] == 2
    publish.assert_awaited_once()


@pytest.mark.parametrize("tagged", [True, False])
def test_repositioned_chunks_keep_document_membership(tagged):
    document = _document(tagged)
    chunk_id = uuid4()
    chunk = incremental._rehydrate_chunk(
        document,
        {"id": str(chunk_id), "text": "existing action", "chunk_size": 2},
        7,
    )
    assert chunk.belongs_to_set == document.belongs_to_set
    assert chunk.source_node_set == document.source_node_set
    assert chunk.id == chunk_id
    assert chunk.chunk_index == 7


# ----- Metadata-only node-set updates (update(data_id, None, ..., node_set=...)) -----

update_mod = importlib.import_module("cognee.api.v1.update.update")
node_sets_mod = importlib.import_module("cognee.api.v1.update.node_sets")
reassign = importlib.import_module("cognee.modules.graph.methods.reassign_document_node_sets")


@pytest.fixture
def authorized(monkeypatch):
    """A local update() that may write the dataset and resolves the id to ``resolved``."""
    import cognee.modules.data.methods as data_methods

    resolved = uuid4()
    monkeypatch.setattr(
        data_methods, "get_authorized_dataset", AsyncMock(return_value=SimpleNamespace())
    )
    monkeypatch.setattr(data_methods, "resolve_data_id", AsyncMock(return_value=resolved))

    async def no_processing(*args, **kwargs):
        raise AssertionError("a node_set-only update reached extraction or the rebuild")

    for name in ("ensure_extractor_runtime", "incremental_update", "add", "cognify", "forget"):
        monkeypatch.setattr(update_mod, name, no_processing)
    return resolved


@pytest.mark.asyncio
async def test_node_set_only_update_dispatches_before_any_processing(monkeypatch, authorized):
    user, dataset_id = SimpleNamespace(id=uuid4()), uuid4()
    reassigned = AsyncMock(return_value={"status": "metadata_updated"})
    monkeypatch.setattr(update_mod, "update_node_sets", reassigned)

    result = await update_mod.update(uuid4(), None, dataset_id, user, node_set=["beta"])

    assert result == {"status": "metadata_updated"}
    reassigned.assert_awaited_once_with(authorized, dataset_id, user, ["beta"])


@pytest.mark.asyncio
async def test_node_set_only_update_refuses_per_call_stores(monkeypatch, authorized):
    from cognee.modules.graph.exceptions import NodeSetReassignmentNotSupportedError

    monkeypatch.setattr(update_mod, "update_node_sets", AsyncMock())

    with pytest.raises(NodeSetReassignmentNotSupportedError):
        await update_mod.update(
            uuid4(), None, uuid4(), SimpleNamespace(id=uuid4()), node_set=[], graph_db_config={}
        )
    update_mod.update_node_sets.assert_not_awaited()


def test_replacement_node_sets_are_validated_and_deduplicated():
    from cognee.modules.engine.models.node_set import InvalidNodeSetError

    assert node_sets_mod.normalize_node_set(["Beta", "gamma", "beta"]) == ["Beta", "gamma"]
    assert node_sets_mod.normalize_node_set([]) == []
    with pytest.raises(InvalidNodeSetError):
        node_sets_mod.normalize_node_set(["beta", " "])


def test_retagging_keeps_other_contributors_and_unrelated_tags():
    from cognee.modules.engine.models.node_set import NodeSet

    alpha = str(NodeSet.id_for("alpha"))
    # alpha left (its membership edge is gone), the unrelated tag stays, beta joins once.
    assert reassign._retagged(["alpha", "project:x", "beta"], {alpha}, ["beta", "gamma"]) == [
        "project:x",
        "beta",
        "gamma",
    ]
    # alpha still has another contributor, so it is not among the removed sets.
    assert reassign._retagged(["alpha"], set(), ["beta"]) == ["alpha", "beta"]
    assert reassign._retagged(None, {alpha}, []) == []


def test_document_and_chunk_metadata_copies_follow_the_replacement():
    document = {
        "belongs_to_set": ["alpha"],
        "source_node_set": "alpha",
        "external_metadata": '{"source": "crm", "node_set": ["alpha"]}',
    }
    patch = reassign._node_patch(document, {reassign._set_id("alpha")}, ["beta"], True, False)
    assert patch["belongs_to_set"] == ["beta"]
    assert patch["source_node_set"] == "beta"
    assert json.loads(patch["external_metadata"]) == {"source": "crm", "node_set": ["beta"]}

    chunk = {"belongs_to_set": ["alpha"], "external_metadata": '{"node_set": ["alpha"]}'}
    cleared = reassign._node_patch(chunk, {reassign._set_id("alpha")}, [], False, True)
    assert cleared == {"belongs_to_set": [], "external_metadata": None}

    # Already consistent: nothing to write.
    assert reassign._node_patch({**document, **patch}, set(), ["beta"], True, False) == {}
