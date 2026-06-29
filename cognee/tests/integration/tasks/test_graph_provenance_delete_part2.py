"""End-to-end graph-provenance delete tests on the real default stack.

Runs the full add -> cognify -> delete pipeline against real Ladybug + LanceDB +
SQLite. The LLM (entity/summary extraction) is mocked for a deterministic graph,
and embeddings are stubbed to fixed zero vectors — these tests never *retrieve*,
they only need vector rows to exist and then verify they are deleted, so the
embedding values are irrelevant. No network/API calls.

All three documents are ingested in a SINGLE multi-document cognify() run, which
processes documents concurrently. Provenance is now folded into the graph write
(COG-5522 #4/#8): a node/edge is created and stamped in one atomic statement, so
concurrent stamps of a shared entity (e.g. one mentioned in two documents)
set-merge instead of racing — every owner ref survives. This is the realistic
ingestion path; before the fold it non-deterministically dropped owner refs.
"""

import os
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

import cognee
from cognee.api.v1.datasets import datasets
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
)
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.engine.operations.setup import setup as setup_cognee
from cognee.modules.users.methods import get_default_user
from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

DOC1 = "Alice knows Bob."
DOC2 = "Alice lives in New York. She is from Berlin."
DOC3 = "Bob lives in New York. Bob is from San Francisco."


def _kg(nodes, edges):
    return KnowledgeGraph(
        summary="s",
        description="s",
        nodes=[Node(id=n, name=n, type=t, description=f"{n} is a {t}", label=n) for n, t in nodes],
        edges=[Edge(source_node_id=s, target_node_id=d, relationship_name=r) for s, d, r in edges],
    )


def _mock_llm_output(text_input, system_prompt, response_model):
    if text_input == "test":
        return "test"
    if response_model == SummarizedContent:
        return SummarizedContent(summary="s", description="s")
    if response_model == KnowledgeGraph:
        if "Berlin" in text_input:  # DOC2
            return _kg(
                [("Alice", "Person"), ("New York", "City"), ("Berlin", "City")],
                [("Alice", "New York", "lives_in"), ("Alice", "Berlin", "from")],
            )
        if "San Francisco" in text_input:  # DOC3
            return _kg(
                [("Bob", "Person"), ("New York", "City"), ("San Francisco", "City")],
                [("Bob", "New York", "lives_in"), ("Bob", "San Francisco", "from")],
            )
        if "knows" in text_input:  # DOC1
            return _kg([("Alice", "Person"), ("Bob", "Person")], [("Alice", "Bob", "knows")])
    return _kg([], [])


async def _fake_embed(self, text):
    """Deterministic zero vectors — values don't matter, only presence/deletion."""
    return [[0.0] * (self.dimensions or 8) for _ in text]


async def _graph_names():
    graph = await get_graph_engine()
    nodes, _ = await graph.get_graph_data()
    return {((props or {}).get("name") or "").lower() for _nid, props in nodes}


async def _graph_relationship_names():
    graph = await get_graph_engine()
    _nodes, edges = await graph.get_graph_data()
    return {e[2] for e in edges}


async def _source_ref_keys_by_name(name):
    """Distinct source ref keys stamped on the graph node(s) with this name."""
    graph = await get_graph_engine()
    nodes, _edges = await graph.get_graph_data()
    ids = [
        str(nid)
        for nid, props in nodes
        if ((props or {}).get("name") or "").lower() == name.lower()
    ]
    snapshots = await graph.get_node_delete_data(ids)
    keys = set()
    for snapshot in snapshots.values():
        keys.update(snapshot.source_ref_keys)
    return keys


async def _setup(tmp_path):
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup_cognee()
    user = await get_default_user()
    await set_database_global_context_variables("main_dataset", user.id)
    return user


async def _ingest_single_run():
    """Add all three docs, then cognify ONCE over the whole dataset (the docs are
    processed concurrently in a single run). Returns (dataset_id, ids...)."""
    r1 = await cognee.add(DOC1)
    r2 = await cognee.add(DOC2)
    r3 = await cognee.add(DOC3)
    d1 = r1.data_ingestion_info[0]["data_id"]
    d2 = r2.data_ingestion_info[0]["data_id"]
    d3 = r3.data_ingestion_info[0]["data_id"]

    cognify_result = await cognee.cognify()
    dataset_id = list(cognify_result.keys())[0]

    return dataset_id, d1, d2, d3


@pytest.mark.asyncio
@patch.object(LiteLLMEmbeddingEngine, "embed_text", _fake_embed)
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def test_data_item_delete_shared_entities_survive(mock_struct, tmp_path):
    """Delete DOC2: its unowned entity (Berlin) and edges go, but entities shared
    with another document (Alice via DOC1, New York via DOC3) survive."""
    mock_struct.side_effect = _mock_llm_output
    user = await _setup(tmp_path)
    dataset_id, _d1, d2, _d3 = await _ingest_single_run()

    # The graph was marked graph-provenance (Part 1 markers are live on the default stack).
    graph = await get_graph_engine()
    metadata = await graph.get_graph_metadata()
    assert metadata.get(GRAPH_DELETE_MODE_KEY) == GRAPH_DELETE_MODE_GRAPH_PROVENANCE

    before = await _graph_names()
    assert {"alice", "bob", "new york", "berlin", "san francisco"} <= before

    # #8 fix: in one multi-doc run the shared entities keep BOTH owners' refs
    # (Alice is in DOC1+DOC2, New York in DOC2+DOC3). Folded stamping set-merges
    # the concurrent writes instead of losing one.
    assert len(await _source_ref_keys_by_name("Alice")) == 2
    assert len(await _source_ref_keys_by_name("New York")) == 2
    assert len(await _source_ref_keys_by_name("Berlin")) == 1

    await datasets.delete_data(dataset_id, d2, user)

    after = await _graph_names()
    assert "berlin" not in after, "Berlin (owned only by DOC2) must be deleted"
    assert "alice" in after, "Alice (also owned by DOC1) must survive"
    assert "new york" in after, "New York (also owned by DOC3) must survive"
    assert {"bob", "san francisco"} <= after

    relationships = await _graph_relationship_names()
    assert "from" not in relationships or "lives_in" in relationships
    # Alice's DOC2-only "from" edge to Berlin is gone; "knows" (DOC1) survives.
    assert "knows" in relationships


@pytest.mark.asyncio
@patch.object(LiteLLMEmbeddingEngine, "embed_text", _fake_embed)
@patch.object(LLMGateway, "acreate_structured_output", new_callable=AsyncMock)
async def test_unowned_edge_deleted_endpoints_survive(mock_struct, tmp_path):
    """Delete DOC1: its unique 'knows' edge is removed while its endpoint entities
    (Alice via DOC2, Bob via DOC3) survive."""
    mock_struct.side_effect = _mock_llm_output
    user = await _setup(tmp_path)
    dataset_id, d1, _d2, _d3 = await _ingest_single_run()

    assert "knows" in await _graph_relationship_names()

    await datasets.delete_data(dataset_id, d1, user)

    assert "knows" not in await _graph_relationship_names(), "DOC1-only edge must be deleted"
    after = await _graph_names()
    assert "alice" in after and "bob" in after, "Shared endpoints must survive"
