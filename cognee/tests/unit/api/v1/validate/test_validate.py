"""Unit tests for cognee.validate().

Graph-shaped fixtures mirror the real get_graph_data() contract — nodes are
(id, props) with a "type" field, edges are (src, tgt, relationship_name, props) —
the same shape test_graph_report_retriever.py's fixtures are pinned to.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

import importlib

from cognee.api.v1.validate.validate import (
    IssueSeverity,
    IssueType,
    ValidationStatus,
    _check_identity_ids,
    _check_orphaned_edges,
    _check_vector_sync,
    validate,
)
from cognee.modules.engine.models import Entity

# Patch targets use the module object rather than the dotted string: the
# package __init__ re-exports the ``validate`` function under the same name as
# the submodule, so both mock's getattr-based path resolution (Python 3.10)
# and ``import ... as`` land on the function instead of the module.
# importlib.import_module returns the real submodule from sys.modules.
validate_module = importlib.import_module("cognee.api.v1.validate.validate")


# ---------------------------------------------------------------------------
# Orphaned edges
# ---------------------------------------------------------------------------


def test_orphaned_edge_detected_when_target_missing():
    nodes = [("a", {"type": "Entity", "name": "A"})]
    edges = [("a", "ghost", "relates_to", {})]

    issues = _check_orphaned_edges(nodes, edges)

    assert len(issues) == 1
    assert issues[0].type == IssueType.ORPHANED_EDGE
    assert issues[0].severity == IssueSeverity.ERROR
    assert "ghost" in issues[0].detail


def test_orphaned_edge_detected_when_source_missing():
    nodes = [("b", {"type": "Entity", "name": "B"})]
    edges = [("ghost", "b", "relates_to", {})]

    issues = _check_orphaned_edges(nodes, edges)

    assert len(issues) == 1
    assert "ghost" in issues[0].detail


def test_no_orphaned_edges_on_clean_graph():
    nodes = [("a", {"type": "Entity"}), ("b", {"type": "Entity"})]
    edges = [("a", "b", "relates_to", {})]

    assert _check_orphaned_edges(nodes, edges) == []


def test_edge_missing_both_endpoints_reports_both_ids():
    issues = _check_orphaned_edges([], [("x", "y", "relates_to", {})])

    assert len(issues) == 1
    assert "'x'" in issues[0].detail and "'y'" in issues[0].detail


# ---------------------------------------------------------------------------
# Identity-id mismatches — Entity/EntityType's own dedup contract
# (DataPoint.id_for), see modules/migrations/versions/namespace_entity_type_node_ids.py
# for why the three stores drifting apart is a real, previously-hit bug class.
# ---------------------------------------------------------------------------


def test_identity_id_matches_no_issue():
    correct_id = str(Entity.id_for("Acme"))
    nodes = [(correct_id, {"type": "Entity", "name": "Acme"})]

    assert _check_identity_ids(nodes) == []


def test_identity_id_mismatch_detected():
    stale_id = str(uuid4())
    nodes = [(stale_id, {"type": "Entity", "name": "Acme"})]

    issues = _check_identity_ids(nodes)

    assert len(issues) == 1
    assert issues[0].type == IssueType.IDENTITY_ID_MISMATCH
    assert issues[0].severity == IssueSeverity.WARNING
    assert stale_id in issues[0].detail
    assert str(Entity.id_for("Acme")) in issues[0].detail


def test_identity_id_check_covers_entity_type_too():
    nodes = [(str(uuid4()), {"type": "EntityType", "name": "Company"})]

    issues = _check_identity_ids(nodes)

    assert len(issues) == 1
    assert "EntityType" in issues[0].detail


def test_identity_id_check_skips_types_without_a_dedup_contract():
    # DocumentChunk declares no identity_fields — a random uuid4 id is correct.
    nodes = [(str(uuid4()), {"type": "DocumentChunk", "text": "hello"})]

    assert _check_identity_ids(nodes) == []


def test_identity_id_check_skips_node_missing_the_identity_property():
    # Property absent from the graph read entirely — can't recompute, so this
    # check stays silent rather than guessing.
    nodes = [(str(uuid4()), {"type": "Entity"})]

    assert _check_identity_ids(nodes) == []


# ---------------------------------------------------------------------------
# Vector sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_vector_entry_detected():
    node_id = str(uuid4())
    nodes = [(node_id, {"type": "Entity", "name": "Acme"})]
    vector_engine = AsyncMock()
    vector_engine.retrieve.return_value = []

    issues = await _check_vector_sync(nodes, vector_engine)

    assert len(issues) == 1
    assert issues[0].type == IssueType.MISSING_VECTOR_ENTRY
    assert issues[0].severity == IssueSeverity.ERROR
    vector_engine.retrieve.assert_awaited_once_with("Entity_name", [node_id])


@pytest.mark.asyncio
async def test_vector_entry_present_no_issue():
    node_id = str(uuid4())
    nodes = [(node_id, {"type": "Entity", "name": "Acme"})]
    point = type("Point", (), {"id": node_id})()
    vector_engine = AsyncMock()
    vector_engine.retrieve.return_value = [point]

    assert await _check_vector_sync(nodes, vector_engine) == []


@pytest.mark.asyncio
async def test_vector_sync_groups_nodes_by_collection():
    entity_id = str(uuid4())
    chunk_id = str(uuid4())
    nodes = [
        (entity_id, {"type": "Entity", "name": "Acme"}),
        (chunk_id, {"type": "DocumentChunk", "text": "hello"}),
    ]
    vector_engine = AsyncMock()
    vector_engine.retrieve.return_value = []

    await _check_vector_sync(nodes, vector_engine)

    called_collections = {call.args[0] for call in vector_engine.retrieve.await_args_list}
    assert called_collections == {"Entity_name", "DocumentChunk_text"}


@pytest.mark.asyncio
async def test_vector_sync_skips_unchecked_types():
    # NodeSet has no index_fields — it's structural scaffolding, never embedded.
    nodes = [(str(uuid4()), {"type": "NodeSet", "name": "some-set"})]
    vector_engine = AsyncMock()

    issues = await _check_vector_sync(nodes, vector_engine)

    assert issues == []
    vector_engine.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Real LanceDB adapter — proves the one inferred contract most worth
# confirming against a genuine adapter rather than a mock: what retrieve()
# actually returns for a present vs. an absent point id.
# ---------------------------------------------------------------------------

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, _texts):
        raise AssertionError("this test seeds vectors directly, it must not embed")


class _EntityNamePayload(BaseModel):
    name: str


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_vector_sync_against_real_lancedb_adapter(tmp_path):
    adapter = LanceDBAdapter(
        url=str(tmp_path / "db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    synced_id = uuid4()
    missing_id = uuid4()

    await adapter.upsert_raw_vectors(
        "Entity_name",
        [{"id": synced_id, "vector": [1.0, 0.0, 0.0], "payload": {"name": "Synced"}}],
        payload_schema=_EntityNamePayload,
    )

    nodes = [
        (str(synced_id), {"type": "Entity", "name": "Synced"}),
        (str(missing_id), {"type": "Entity", "name": "Missing"}),
    ]

    issues = await _check_vector_sync(nodes, adapter)

    assert len(issues) == 1
    assert str(missing_id) in issues[0].detail
    assert str(synced_id) not in issues[0].detail


# ---------------------------------------------------------------------------
# Status rollup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_context(*_args, **_kwargs):
    yield


def _mock_dataset():
    return type("Dataset", (), {"id": "dataset-id", "owner_id": "owner-id"})()


@pytest.mark.asyncio
async def test_validate_reports_healthy_and_scopes_to_dataset():
    acme_id = str(Entity.id_for("Acme"))
    widget_id = str(Entity.id_for("Widget"))
    nodes = [
        (acme_id, {"type": "Entity", "name": "Acme"}),
        (widget_id, {"type": "Entity", "name": "Widget"}),
    ]
    edges = [(acme_id, widget_id, "produces", {})]

    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (nodes, edges)

    mock_vector_engine = AsyncMock()
    mock_vector_engine.retrieve.return_value = [
        type("Point", (), {"id": acme_id})(),
        type("Point", (), {"id": widget_id})(),
    ]

    with (
        patch.object(
            validate_module,
            "get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            validate_module,
            "get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch.object(
            validate_module,
            "get_authorized_existing_datasets",
            return_value=[_mock_dataset()],
        ),
        patch.object(
            validate_module,
            "get_default_user",
            return_value=object(),
        ),
        patch.object(
            validate_module,
            "set_database_global_context_variables",
            _noop_context,
        ),
    ):
        report = await validate(dataset="my_dataset", user=object())

    assert report.status == ValidationStatus.HEALTHY
    assert report.issues == []
    assert report.summary == {
        "graph_nodes": 2,
        "graph_edges": 1,
        "node_type_distribution": {"Entity": 2},
    }


@pytest.mark.asyncio
async def test_validate_reports_unhealthy_when_an_error_issue_exists():
    node_id = str(uuid4())
    nodes = [(node_id, {"type": "Entity", "name": "Acme"})]
    # A dangling edge is an ERROR-severity issue.
    edges = [(node_id, "ghost", "relates_to", {})]

    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (nodes, edges)
    mock_vector_engine = AsyncMock()
    mock_vector_engine.retrieve.return_value = [type("Point", (), {"id": node_id})()]

    with (
        patch.object(
            validate_module,
            "get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            validate_module,
            "get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch.object(
            validate_module,
            "get_authorized_existing_datasets",
            return_value=[_mock_dataset()],
        ),
        patch.object(
            validate_module,
            "get_default_user",
            return_value=object(),
        ),
        patch.object(
            validate_module,
            "set_database_global_context_variables",
            _noop_context,
        ),
    ):
        report = await validate(dataset="my_dataset", user=object())

    assert report.status == ValidationStatus.UNHEALTHY
    assert any(issue.type == IssueType.ORPHANED_EDGE for issue in report.issues)


@pytest.mark.asyncio
async def test_validate_reports_degraded_when_only_warning_issues_exist():
    # Identity-id mismatch is WARNING-severity, and nothing else is wrong.
    stale_id = str(uuid4())
    nodes = [(stale_id, {"type": "Entity", "name": "Acme"})]

    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (nodes, [])
    mock_vector_engine = AsyncMock()
    mock_vector_engine.retrieve.return_value = [type("Point", (), {"id": stale_id})()]

    with (
        patch.object(
            validate_module,
            "get_graph_engine",
            return_value=mock_graph_engine,
        ),
        patch.object(
            validate_module,
            "get_vector_engine_async",
            return_value=mock_vector_engine,
        ),
        patch.object(
            validate_module,
            "get_authorized_existing_datasets",
            return_value=[_mock_dataset()],
        ),
        patch.object(
            validate_module,
            "get_default_user",
            return_value=object(),
        ),
        patch.object(
            validate_module,
            "set_database_global_context_variables",
            _noop_context,
        ),
    ):
        report = await validate(dataset="my_dataset", user=object())

    assert report.status == ValidationStatus.DEGRADED
    assert len(report.issues) == 1
    assert report.issues[0].type == IssueType.IDENTITY_ID_MISMATCH
