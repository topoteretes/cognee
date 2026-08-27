"""Tests for ensure_graph_memory_cleared (COG-6335).

Covers the fallback forget(memory_only=True) uses for graph content that
carries no deletion provenance (e.g. remember(content_type="code")):

- no access control + no code_graph_pipeline run on record -> cleared
  (unchanged default behavior)
- no access control + a code_graph_pipeline run on record -> NOT cleared,
  named possibility, no reset attempted (nothing is safe to wipe when other
  datasets could share the same store)
- isolated + no DatasetDatabase row -> cleared, no reset attempted
- already-empty graph -> cleared without touching storage
- isolated + self-healing backend + graph still non-empty -> full store
  reset and cleared, REGARDLESS of whether the provenance-driven step found
  and removed content for this dataset or not (isolation guarantees nothing
  left behind can belong to another dataset) — this is the common
  documents+code-in-one-dataset case, since both remember(text) and
  remember(content_type="code") default to dataset_name="main_dataset"
- isolated + self-healing backend + graph still non-empty + a code
  snapshot marker present -> reset clears it too, so a later
  remember(content_type="code") does not skip re-ingestion (the COG-6335
  end-to-end claim)
- isolated + server-managed handler (e.g. neo4j) + graph still non-empty ->
  NOT cleared, no reset attempted (dropping it would leave the dataset with
  no backing database until something re-issues CREATE)
- get_graph_engine() resolves the dataset-scoped engine active in the
  caller's context, not some other dataset's or the process default
- unexpected error -> NOT cleared instead of raising, and logged with a
  traceback (exc_info=True)
- isolated + self-healing backend + EMPTY graph + a code_graph_pipeline run
  on record -> still resets (an empty graph does not prove the vector store
  is empty: index_vectors=True embeddings are never swept by the code-graph
  pipeline's own stale-content sweep)
- isolated + non-self-healing backend + empty graph + a code_graph_pipeline
  run on record -> NOT cleared, no reset attempted (same honesty behavior as
  the non-empty case)
- a reset captures the node/edge ids it is about to remove (via
  get_graph_data()) BEFORE wiping the store, and returns them as
  GraphMemoryStatus.deleted_elements
- a reset invalidates the CODE search snapshot cache for this dataset
- _UNPROVENANCED_PIPELINE_NAMES is a frozenset, not a single hardcoded string
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

module = importlib.import_module("cognee.modules.graph.methods.ensure_graph_memory_cleared")
graph_engine_module = importlib.import_module(
    "cognee.infrastructure.databases.graph.get_graph_engine"
)
pipelines_methods_module = importlib.import_module("cognee.modules.pipelines.methods")
code_retriever_module = importlib.import_module("cognee.modules.retrieval.code_retriever")

pytestmark = pytest.mark.asyncio

DATASET_ID = uuid4()
USER = SimpleNamespace(id=uuid4())


def _dataset_database(graph_handler="ladybug", vector_handler="lancedb"):
    return SimpleNamespace(
        graph_dataset_database_handler=graph_handler,
        vector_dataset_database_handler=vector_handler,
    )


class _FakeGraphEngine:
    """Minimal in-memory stand-in with just what this module touches."""

    def __init__(self, nodes=None, edges=None):
        self.nodes = dict(nodes or {})
        self.edges = list(edges or [])

    async def is_empty(self) -> bool:
        return not self.nodes

    async def get_node(self, node_id):
        return self.nodes.get(str(node_id))

    async def get_graph_data(self):
        return list(self.nodes.items()), list(self.edges)

    def wipe(self) -> None:
        self.nodes.clear()
        self.edges.clear()


# ---------------------------------------------------------------------------
# Non-isolated (ENABLE_BACKEND_ACCESS_CONTROL=false) — no reset is ever safe;
# only the read-only PipelineRun honesty check applies.
# ---------------------------------------------------------------------------


async def test_no_access_control_and_no_code_run_reports_cleared(monkeypatch):
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(module, "get_pipeline_run_by_dataset", AsyncMock(return_value=None))
    lookup = AsyncMock()
    monkeypatch.setattr(module, "get_existing_dataset_database", lookup)
    reset = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is True
    lookup.assert_not_called()
    reset.assert_not_called()


async def test_no_access_control_with_code_run_on_record_reports_not_cleared(monkeypatch):
    """Follow-up: a code_graph_pipeline PipelineRun row for this dataset means
    content with no deletion provenance may still be sitting in the (shared)
    graph — the same bug as the isolated case, just with no safe fallback, so
    the report must say so instead of the previous unconditional cleared=True.
    """
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: False)
    monkeypatch.setattr(
        module,
        "get_pipeline_run_by_dataset",
        AsyncMock(return_value=SimpleNamespace(pipeline_name="code_graph_pipeline")),
    )
    reset = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is False
    assert "may remain" in status.note
    reset.assert_not_called()


async def test_no_access_control_queries_the_right_pipeline_name(monkeypatch):
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: False)
    query = AsyncMock(return_value=None)
    monkeypatch.setattr(module, "get_pipeline_run_by_dataset", query)

    await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    query.assert_awaited_once_with(DATASET_ID, "code_graph_pipeline")


# ---------------------------------------------------------------------------
# Isolated store — no DatasetDatabase row, or already empty.
# ---------------------------------------------------------------------------


async def test_no_dataset_database_row_reports_cleared_and_never_resets(monkeypatch):
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(module, "get_existing_dataset_database", AsyncMock(return_value=None))
    engine_getter = AsyncMock()
    monkeypatch.setattr(module, "get_graph_engine", engine_getter)
    reset = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is True
    engine_getter.assert_not_called()
    reset.assert_not_called()


async def test_already_empty_graph_reports_cleared_without_reset(monkeypatch):
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module, "get_existing_dataset_database", AsyncMock(return_value=_dataset_database())
    )
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=_FakeGraphEngine()))
    query = AsyncMock(return_value=None)
    monkeypatch.setattr(module, "get_pipeline_run_by_dataset", query)
    reset = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is True
    assert status.deleted_elements is None
    # Finding 6: an empty graph alone is not trusted -- the same cheap
    # unprovenanced-pipeline signal used for the non-isolated honesty check
    # is consulted here too, before short-circuiting to cleared=True.
    query.assert_awaited_once_with(DATASET_ID, "code_graph_pipeline")
    reset.assert_not_called()


# ---------------------------------------------------------------------------
# Isolated store, still non-empty after the provenance-driven step — the
# decision no longer depends on what that step found (see module docstring).
# ---------------------------------------------------------------------------


async def test_isolated_self_healing_backend_resets_even_when_provenance_found_content(
    monkeypatch,
):
    """The documents+code-in-one-dataset case: the provenance-driven step
    already found and removed this dataset's document content — that is not
    relevant here any more, because isolation means whatever is left in the
    store afterward cannot belong to any other dataset either way, so it is
    always safe to reset once the store is confirmed isolated and self-healing.

    Also covers Findings 2/9 (deleted_elements captured before the wipe) and 7
    (CODE snapshot cache invalidated after a reset).
    """
    from cognee.modules.engine.utils import generate_edge_object_id

    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module, "get_existing_dataset_database", AsyncMock(return_value=_dataset_database())
    )
    monkeypatch.setattr(
        module,
        "get_graph_engine",
        AsyncMock(
            return_value=_FakeGraphEngine(
                nodes={"n1": {}, "n2": {}},
                edges=[("n1", "n2", "calls", {})],
            )
        ),
    )
    reset = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)
    query = AsyncMock(return_value=None)
    monkeypatch.setattr(module, "get_pipeline_run_by_dataset", query)
    invalidate_mock = MagicMock()
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", invalidate_mock
    )

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is True
    assert "reset" in status.note
    reset.assert_awaited_once()
    # The graph was already non-empty, so the extra unprovenanced-pipeline
    # check (only needed for an empty graph — see Finding 6) never runs.
    query.assert_not_called()

    assert status.deleted_elements is not None
    assert status.deleted_elements.node_ids == {"n1", "n2"}
    assert status.deleted_elements.edge_ids == {str(generate_edge_object_id("n1", "n2", "calls"))}
    invalidate_mock.assert_called_once_with(dataset_id=DATASET_ID)


async def test_unsupported_backend_reports_not_cleared_and_never_resets(monkeypatch):
    """The graph is non-empty (whether the provenance step found something for
    this dataset or not makes no difference — see above), but the per-dataset
    store is server-managed (e.g. Neo4j) — dropping it would leave the dataset
    with no backing database until something re-issues CREATE, so the reset is
    intentionally not attempted."""
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module,
        "get_existing_dataset_database",
        AsyncMock(return_value=_dataset_database(graph_handler="neo4j")),
    )
    monkeypatch.setattr(
        module, "get_graph_engine", AsyncMock(return_value=_FakeGraphEngine({"n1": {}}))
    )
    reset = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is False
    assert "neo4j" in status.note
    reset.assert_not_called()


async def test_unexpected_error_reports_not_cleared_instead_of_raising(monkeypatch):
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module,
        "get_existing_dataset_database",
        AsyncMock(side_effect=RuntimeError("relational engine unavailable")),
    )

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is False
    assert status.note


async def test_unexpected_error_is_logged_with_a_traceback(monkeypatch):
    """Finding 12 (COG-6335 review): the broad except block must log with
    exc_info=True so a traceback reaches the logs, instead of only str(error)."""
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module,
        "get_existing_dataset_database",
        AsyncMock(side_effect=RuntimeError("relational engine unavailable")),
    )
    mock_warning = MagicMock()
    monkeypatch.setattr(module.logger, "warning", mock_warning)

    await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    mock_warning.assert_called_once()
    _, kwargs = mock_warning.call_args
    assert kwargs.get("exc_info") is True


async def test_isolated_self_healing_backend_resets_and_clears_snapshot_marker(monkeypatch):
    """The COG-6335 end-to-end claim: a code dataset (isolated store,
    self-healing backend) gets a full store reset, and the CodeRepository
    snapshot marker that used to make extract_code_graph skip re-ingestion is
    gone afterward, so the next remember(content_type="code") does a full
    load instead of "already matches snapshot ...; skipping load."
    """
    from cognee.tasks.code_graph.extract_code_graph import _stored_snapshot_identity, fact_node_id

    repo = "demo_repo"
    node_id = str(fact_node_id(repo, "repository", repo))
    fake_graph = _FakeGraphEngine({node_id: {"last_snapshot_id": "sha256:old"}})

    # extract_code_graph's own snapshot lookup goes through get_graph_engine too
    # (a local import inside _stored_snapshot_identity), so patch it at the
    # source module to share the same fake engine.
    monkeypatch.setattr(graph_engine_module, "get_graph_engine", AsyncMock(return_value=fake_graph))
    assert await _stored_snapshot_identity(repo) == "sha256:old"

    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module, "get_existing_dataset_database", AsyncMock(return_value=_dataset_database())
    )
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=fake_graph))

    async def _wipe(_dataset_database):
        fake_graph.wipe()

    reset = AsyncMock(side_effect=_wipe)
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is True
    assert "reset" in status.note
    reset.assert_awaited_once()

    # The end-to-end claim: the marker is gone, so a later ingest will not skip.
    assert await _stored_snapshot_identity(repo) is None


async def test_get_graph_engine_resolves_the_active_per_dataset_context(monkeypatch):
    """_forget_dataset_memory always calls ensure_graph_memory_cleared from
    inside set_database_global_context_variables(resolved_dataset_id,
    user.id). Prove get_graph_engine() here actually picks up THAT ambient
    per-dataset context (via get_graph_context_config reading the
    graph_db_config ContextVar the same way set_database_global_context_variables
    sets it) rather than some other dataset's or the process-wide default."""
    from cognee.context_global_variables import graph_db_config
    from cognee.infrastructure.databases.graph.config import get_graph_context_config

    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module, "get_existing_dataset_database", AsyncMock(return_value=_dataset_database())
    )
    monkeypatch.setattr(module, "get_pipeline_run_by_dataset", AsyncMock(return_value=None))

    seen_graph_database_name = None

    async def _fake_get_graph_engine():
        nonlocal seen_graph_database_name
        seen_graph_database_name = get_graph_context_config().get("graph_database_name")
        return _FakeGraphEngine()

    monkeypatch.setattr(module, "get_graph_engine", _fake_get_graph_engine)

    this_dataset_marker = f"{DATASET_ID}.lbug"
    token = graph_db_config.set({"graph_database_name": this_dataset_marker})
    try:
        await module.ensure_graph_memory_cleared(DATASET_ID, USER)
    finally:
        graph_db_config.reset(token)

    assert seen_graph_database_name == this_dataset_marker


# ---------------------------------------------------------------------------
# Finding 6 (COG-6335 review): an empty graph does not prove the isolated
# vector store is empty too (remember(content_type="code", index_vectors=True)
# writes vector embeddings the code-graph pipeline's own stale-sweep never
# touches) — a code_graph_pipeline run on record routes an otherwise-empty
# graph into the same reset path used for a non-empty one.
# ---------------------------------------------------------------------------


async def test_isolated_self_healing_backend_resets_when_graph_empty_but_code_run_on_record(
    monkeypatch,
):
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module, "get_existing_dataset_database", AsyncMock(return_value=_dataset_database())
    )
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=_FakeGraphEngine()))
    monkeypatch.setattr(
        module,
        "get_pipeline_run_by_dataset",
        AsyncMock(return_value=SimpleNamespace(pipeline_name="code_graph_pipeline")),
    )
    reset = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)
    monkeypatch.setattr(code_retriever_module, "invalidate_code_graph_snapshot_cache", MagicMock())

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is True
    assert "vector" in status.note
    reset.assert_awaited_once()
    # Nothing was in the graph to capture, but the reset still ran (for the
    # vector store's sake) — deleted_elements reflects that: present, empty.
    assert status.deleted_elements is not None
    assert status.deleted_elements.node_ids == set()
    assert status.deleted_elements.edge_ids == set()


async def test_isolated_non_self_healing_backend_with_graph_empty_and_code_run_reports_not_cleared(
    monkeypatch,
):
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module,
        "get_existing_dataset_database",
        AsyncMock(return_value=_dataset_database(graph_handler="neo4j")),
    )
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=_FakeGraphEngine()))
    monkeypatch.setattr(
        module,
        "get_pipeline_run_by_dataset",
        AsyncMock(return_value=SimpleNamespace(pipeline_name="code_graph_pipeline")),
    )
    reset = AsyncMock()
    monkeypatch.setattr(module, "delete_isolated_dataset_storage", reset)

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is False
    assert "neo4j" in status.note
    assert status.deleted_elements is None
    reset.assert_not_called()


async def test_isolated_reset_skips_snapshot_cache_invalidation_when_no_reset_ran(monkeypatch):
    """invalidate_code_graph_snapshot_cache is only called in the branch where
    a reset actually happened — not on every call (would be wasted work on
    the far more common already-cleared path)."""
    monkeypatch.setattr(module, "backend_access_control_enabled", lambda: True)
    monkeypatch.setattr(
        module, "get_existing_dataset_database", AsyncMock(return_value=_dataset_database())
    )
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=_FakeGraphEngine()))
    monkeypatch.setattr(module, "get_pipeline_run_by_dataset", AsyncMock(return_value=None))
    invalidate_mock = MagicMock()
    monkeypatch.setattr(
        code_retriever_module, "invalidate_code_graph_snapshot_cache", invalidate_mock
    )

    status = await module.ensure_graph_memory_cleared(DATASET_ID, USER)

    assert status.cleared is True
    invalidate_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Finding 13 (COG-6335 review): a frozenset, not a single hardcoded string, so
# a future unprovenanced pipeline is a one-line addition.
# ---------------------------------------------------------------------------


async def test_unprovenanced_pipeline_names_is_a_frozenset_containing_code_graph_pipeline():
    assert isinstance(module._UNPROVENANCED_PIPELINE_NAMES, frozenset)
    assert "code_graph_pipeline" in module._UNPROVENANCED_PIPELINE_NAMES
