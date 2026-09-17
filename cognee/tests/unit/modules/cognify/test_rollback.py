from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.cognify import rollback as rollback_module
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
)


class _FakeScalarsResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class _FakeExecuteResult:
    def __init__(self, items=None, scalar_value=None):
        self._items = items if items is not None else []
        self._scalar_value = scalar_value

    def scalars(self):
        return _FakeScalarsResult(self._items)

    def scalar(self):
        return self._scalar_value


class _FakeSession:
    def __init__(self, execute_results, call_log=None):
        self._execute_results = list(execute_results)
        self._call_log = call_log if call_log is not None else []
        self.committed = False

    async def execute(self, statement):
        if getattr(statement, "is_delete", False):
            self._call_log.append("relational_delete")
        return self._execute_results.pop(0)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, sessions):
        self._sessions = list(sessions)
        self.calls = 0

    def get_async_session(self):
        self.calls += 1
        return self._sessions.pop(0)


@pytest.mark.asyncio
async def test_cognify_rollback_deletes_graph_before_relational(monkeypatch):
    pipeline_run_id = uuid4()
    dataset_id = uuid4()
    data_id = uuid4()
    node_id = uuid4()
    edge_id = uuid4()

    node = SimpleNamespace(id=node_id, slug="node-1", data_id=data_id)
    edge = SimpleNamespace(id=edge_id, slug="edge-1", data_id=data_id)

    call_log = []
    session_discovery = _FakeSession(
        [
            _FakeExecuteResult([node]),
            _FakeExecuteResult([edge]),
            _FakeExecuteResult(scalar_value=False),
            _FakeExecuteResult(scalar_value=False),
        ]
    )
    data_record = SimpleNamespace(
        id=data_id,
        pipeline_status={"cognify_pipeline": {str(dataset_id): "DATASET_PROCESSING_STARTED"}},
    )
    session_mutation = _FakeSession(
        [
            _FakeExecuteResult(),
            _FakeExecuteResult(),
            _FakeExecuteResult([data_record]),
        ],
        call_log=call_log,
    )
    engine = _FakeEngine([session_discovery, session_mutation])

    async def _delete_from_graph_and_vector(*_args, **_kwargs):
        call_log.append("graph_delete")

    async def _has_nodes_in_legacy_ledger(_nodes):
        return []

    async def _has_edges_in_legacy_ledger(_edges):
        return []

    # Pin the engine as non-graph-provenance so rollback takes the relational-ledger
    # path (the unified branch is gated on supports_graph_provenance_delete()).
    async def _get_unified_engine():
        return SimpleNamespace(supports_graph_provenance_delete=lambda: False)

    monkeypatch.setattr(rollback_module, "get_unified_engine", _get_unified_engine)
    monkeypatch.setattr(rollback_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(rollback_module, "multi_user_support_possible", lambda: False)
    monkeypatch.setattr(rollback_module, "has_nodes_in_legacy_ledger", _has_nodes_in_legacy_ledger)
    monkeypatch.setattr(rollback_module, "has_edges_in_legacy_ledger", _has_edges_in_legacy_ledger)
    monkeypatch.setattr(
        rollback_module, "delete_from_graph_and_vector", _delete_from_graph_and_vector
    )
    monkeypatch.setattr(rollback_module.orm_attributes, "flag_modified", lambda *_args: None)

    await rollback_module.cognify_rollback_handler(
        pipeline_run_id=pipeline_run_id,
        dataset=SimpleNamespace(id=dataset_id),
    )

    assert call_log == ["graph_delete", "relational_delete", "relational_delete"]
    assert str(dataset_id) not in data_record.pipeline_status["cognify_pipeline"]
    assert session_mutation.committed is True


@pytest.mark.asyncio
async def test_graph_provenance_rollback_resets_status_without_ingestion_info(monkeypatch):
    """Startup recovery passes no data_ingestion_info, so the graph-provenance branch
    must derive the affected data ids from graph provenance and still reset the
    per-data cognify status (otherwise re-cognify would skip the data).

    The run's refs carry BOTH key versions: chunk-produced artifacts are stamped
    with the chunk-scoped v2 key, so a v1-only parser here throws before
    rollback_by_pipeline_run_id ever runs — and run_tasks only logs rollback
    errors, so the failed cognify would silently keep its partial graph.
    """
    from cognee.infrastructure.databases.provenance import (
        make_chunk_source_ref_key,
        make_source_ref_key,
    )

    pipeline_run_id = uuid4()
    dataset_id = uuid4()
    data_id = uuid4()
    source_ref_key = make_source_ref_key(dataset_id, data_id)
    chunk_source_ref_key = make_chunk_source_ref_key(dataset_id, data_id, uuid4())

    rolled_back = []

    class _FakeGraph:
        async def find_node_source_refs_by_pipeline_run(self, _run):
            return {"n1": [source_ref_key], "n2": [chunk_source_ref_key]}

        async def find_edge_source_refs_by_pipeline_run(self, _run):
            return {}

    async def _rollback(run, *, keep_data_ids=None):
        # The inline failure path keeps nothing: whole-run rollback.
        assert keep_data_ids is None
        rolled_back.append(run)

    fake_unified = SimpleNamespace(
        supports_graph_provenance_delete=lambda: True,
        graph=_FakeGraph(),
        rollback_by_pipeline_run_id=_rollback,
    )

    data_record = SimpleNamespace(
        id=data_id,
        pipeline_status={"cognify_pipeline": {str(dataset_id): "DATASET_PROCESSING_STARTED"}},
    )
    session_mutation = _FakeSession([_FakeExecuteResult([data_record])])
    engine = _FakeEngine([session_mutation])

    async def _get_unified_engine():
        return fake_unified

    async def _stores_provenance_in_graph(_graph):
        return True

    monkeypatch.setattr(rollback_module, "get_unified_engine", _get_unified_engine)
    monkeypatch.setattr(rollback_module, "stores_provenance_in_graph", _stores_provenance_in_graph)
    monkeypatch.setattr(rollback_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(rollback_module.orm_attributes, "flag_modified", lambda *_args: None)

    # No data_ingestion_info — exactly how startup recovery calls it.
    await rollback_module.cognify_rollback_handler(
        pipeline_run_id=pipeline_run_id,
        dataset=SimpleNamespace(id=dataset_id),
    )

    assert rolled_back == [str(pipeline_run_id)]
    # Status was reset for the data id derived from the run's graph source refs.
    assert str(dataset_id) not in data_record.pipeline_status["cognify_pipeline"]
    assert session_mutation.committed is True


@pytest.mark.asyncio
async def test_cognify_rollback_keeps_relational_rows_if_graph_delete_fails(monkeypatch):
    pipeline_run_id = uuid4()
    dataset_id = uuid4()
    data_id = uuid4()
    node_id = uuid4()
    edge_id = uuid4()

    node = SimpleNamespace(id=node_id, slug="node-1", data_id=data_id)
    edge = SimpleNamespace(id=edge_id, slug="edge-1", data_id=data_id)

    session_discovery = _FakeSession(
        [
            _FakeExecuteResult([node]),
            _FakeExecuteResult([edge]),
            _FakeExecuteResult(scalar_value=False),
            _FakeExecuteResult(scalar_value=False),
        ]
    )
    engine = _FakeEngine([session_discovery])

    async def _failing_delete(*_args, **_kwargs):
        raise RuntimeError("graph delete failed")

    async def _has_nodes_in_legacy_ledger(_nodes):
        return []

    async def _has_edges_in_legacy_ledger(_edges):
        return []

    # Pin the engine as non-graph-provenance so rollback takes the relational-ledger
    # path (the unified branch is gated on supports_graph_provenance_delete()).
    async def _get_unified_engine():
        return SimpleNamespace(supports_graph_provenance_delete=lambda: False)

    monkeypatch.setattr(rollback_module, "get_unified_engine", _get_unified_engine)
    monkeypatch.setattr(rollback_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(rollback_module, "multi_user_support_possible", lambda: False)
    monkeypatch.setattr(rollback_module, "has_nodes_in_legacy_ledger", _has_nodes_in_legacy_ledger)
    monkeypatch.setattr(rollback_module, "has_edges_in_legacy_ledger", _has_edges_in_legacy_ledger)
    monkeypatch.setattr(rollback_module, "delete_from_graph_and_vector", _failing_delete)

    with pytest.raises(RuntimeError, match="graph delete failed"):
        await rollback_module.cognify_rollback_handler(
            pipeline_run_id=pipeline_run_id,
            dataset=SimpleNamespace(id=dataset_id),
        )

    assert engine.calls == 1


def test_extract_data_ids_skips_already_completed_items():
    """An item skipped as already-completed belongs to an EARLIER run, so the run
    being rolled back must not claim its marker."""
    completed_id = uuid4()
    errored_id = uuid4()
    skipped_id = uuid4()
    run_info_fields = {
        "pipeline_run_id": uuid4(),
        "dataset_id": uuid4(),
        "dataset_name": "some-dataset",
    }

    data_ingestion_info = [
        {"run_info": PipelineRunCompleted(**run_info_fields), "data_id": completed_id},
        {"run_info": PipelineRunAlreadyCompleted(**run_info_fields), "data_id": skipped_id},
        {
            "run_info": PipelineRunErrored(**run_info_fields, payload="boom"),
            "data_id": errored_id,
        },
    ]

    assert rollback_module._extract_data_ids(data_ingestion_info) == {completed_id, errored_id}


@pytest.mark.asyncio
async def test_rollback_preserves_markers_of_previously_extracted_data(monkeypatch):
    """One unprocessable document must not cost the whole dataset its markers.

    ``run_tasks`` raises as soon as any item errors and passes the rollback EVERY
    per-item result, including the already-completed ones an earlier run extracted.
    Clearing those markers strands records whose nodes are still in the graph — the
    node/edge deletion is scoped by pipeline_run_id and never removes them — so the
    next cognify re-extracts them at full LLM cost.
    """
    pipeline_run_id = uuid4()
    dataset_id = uuid4()
    errored_id = uuid4()
    previously_extracted_id = uuid4()

    # No nodes or edges from this run: the run died before writing any, so the only
    # source of target ids is data_ingestion_info — which is where the bug lived.
    session_discovery = _FakeSession([_FakeExecuteResult([]), _FakeExecuteResult([])])
    session_mutation = _FakeSession([])
    engine = _FakeEngine([session_discovery, session_mutation])

    reset_calls = []

    async def _capture_reset(_session, target_data_ids, _dataset_id):
        reset_calls.append(set(target_data_ids))

    async def _get_unified_engine():
        return SimpleNamespace(supports_graph_provenance_delete=lambda: False)

    monkeypatch.setattr(rollback_module, "get_unified_engine", _get_unified_engine)
    monkeypatch.setattr(rollback_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(rollback_module, "multi_user_support_possible", lambda: False)
    monkeypatch.setattr(rollback_module, "_reset_pipeline_status", _capture_reset)

    await rollback_module.cognify_rollback_handler(
        pipeline_run_id=pipeline_run_id,
        dataset=SimpleNamespace(id=dataset_id),
        data_ingestion_info=[
            {
                "run_info": PipelineRunErrored(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset_id,
                    dataset_name="some-dataset",
                    payload="boom",
                ),
                "data_id": errored_id,
            },
            {
                "run_info": PipelineRunAlreadyCompleted(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset_id,
                    dataset_name="some-dataset",
                ),
                "data_id": previously_extracted_id,
            },
        ],
    )

    assert reset_calls == [{errored_id}]
    assert previously_extracted_id not in reset_calls[0]


@pytest.mark.asyncio
async def test_graph_provenance_rollback_keeps_completed_documents_when_asked(monkeypatch):
    """Startup recovery asks for keep_completed_data: a document the run marked
    complete keeps its refs and its completed status; the unfinished one is
    rolled back and its status cleared."""
    from cognee.infrastructure.databases.provenance import make_source_ref_key
    from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

    pipeline_run_id = uuid4()
    dataset_id = uuid4()
    done_id, unfinished_id = uuid4(), uuid4()
    done_ref = make_source_ref_key(dataset_id, done_id)
    unfinished_ref = make_source_ref_key(dataset_id, unfinished_id)

    rollback_calls = []

    class _FakeGraph:
        async def find_node_source_refs_by_pipeline_run(self, _run):
            return {"n-done": [done_ref], "n-unfinished": [unfinished_ref]}

        async def find_edge_source_refs_by_pipeline_run(self, _run):
            return {}

    async def _rollback(run, *, keep_data_ids=None):
        rollback_calls.append((run, keep_data_ids))

    fake_unified = SimpleNamespace(
        supports_graph_provenance_delete=lambda: True,
        graph=_FakeGraph(),
        rollback_by_pipeline_run_id=_rollback,
    )
    done_record = SimpleNamespace(
        id=done_id,
        pipeline_status={
            "cognify_pipeline": {str(dataset_id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED}
        },
    )
    unfinished_record = SimpleNamespace(
        id=unfinished_id,
        pipeline_status={"cognify_pipeline": {str(dataset_id): "DATASET_PROCESSING_STARTED"}},
    )
    # First session answers the completed-status lookup for both ids; the second
    # is the status reset, whose query only covers the unfinished id.
    lookup_session = _FakeSession([_FakeExecuteResult([done_record, unfinished_record])])
    reset_session = _FakeSession([_FakeExecuteResult([unfinished_record])])
    engine = _FakeEngine([lookup_session, reset_session])

    async def _get_unified_engine():
        return fake_unified

    async def _stores_provenance_in_graph(_graph):
        return True

    monkeypatch.setattr(rollback_module, "get_unified_engine", _get_unified_engine)
    monkeypatch.setattr(rollback_module, "stores_provenance_in_graph", _stores_provenance_in_graph)
    monkeypatch.setattr(rollback_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(rollback_module.orm_attributes, "flag_modified", lambda *_args: None)

    await rollback_module.cognify_rollback_handler(
        pipeline_run_id=pipeline_run_id,
        dataset=SimpleNamespace(id=dataset_id),
        keep_completed_data=True,
    )

    assert rollback_calls == [(str(pipeline_run_id), {done_id})]
    assert done_record.pipeline_status["cognify_pipeline"][str(dataset_id)] == (
        DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
    )
    assert str(dataset_id) not in unfinished_record.pipeline_status["cognify_pipeline"]
    assert reset_session.committed is True


def test_without_kept_refs_drops_only_the_kept_documents_refs():
    from cognee.infrastructure.databases.provenance import make_source_ref_key
    from cognee.infrastructure.databases.unified.unified_store_engine import _without_kept_refs

    dataset_id, kept, other = uuid4(), uuid4(), uuid4()
    refs = {
        "shared": [make_source_ref_key(dataset_id, kept), make_source_ref_key(dataset_id, other)],
        "only-kept": [make_source_ref_key(dataset_id, kept)],
    }

    trimmed = _without_kept_refs(refs, {kept})

    assert trimmed == {"shared": [make_source_ref_key(dataset_id, other)]}
