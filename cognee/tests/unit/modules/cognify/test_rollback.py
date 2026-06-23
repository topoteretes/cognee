from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.cognify import rollback as rollback_module


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

    # Pin the engine as non-graph-native so rollback takes the relational-ledger
    # path (the unified branch is gated on supports_graph_native_delete()).
    async def _get_unified_engine():
        return SimpleNamespace(supports_graph_native_delete=lambda: False)

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

    # Pin the engine as non-graph-native so rollback takes the relational-ledger
    # path (the unified branch is gated on supports_graph_native_delete()).
    async def _get_unified_engine():
        return SimpleNamespace(supports_graph_native_delete=lambda: False)

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
