"""Tests for the memory_only option in cognee.forget().

Covers:
- _forget_dataset_memory: clears graph+vector, resets pipeline_status, preserves data
- _forget_data_memory: clears graph+vector for a single item, resets its pipeline_status
- Validation: memory_only without dataset raises ValueError
- Routing: memory_only with dataset vs memory_only with dataset+data_id
- Telemetry target labels for memory_only combinations
"""

import importlib
import pytest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, patch

# Import the actual module (not the function) to access private helpers
forget_module = importlib.import_module("cognee.api.v1.forget.forget")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")
delete_dataset_nodes_and_edges_module = importlib.import_module(
    "cognee.modules.graph.methods.delete_dataset_nodes_and_edges"
)
delete_data_nodes_and_edges_module = importlib.import_module(
    "cognee.modules.graph.methods.delete_data_nodes_and_edges"
)
reset_dataset_pipeline_run_status_module = importlib.import_module(
    "cognee.modules.pipelines.layers.reset_dataset_pipeline_run_status"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASET_ID = uuid4()
DATA_ID_A = uuid4()
DATA_ID_B = uuid4()
USER = SimpleNamespace(id=uuid4())


def _make_data_record(data_id, pipeline_status=None):
    """Create a fake Data record with mutable pipeline_status."""
    return SimpleNamespace(id=data_id, pipeline_status=pipeline_status)


class _FakeScalarsResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items

    def first(self):
        return self._items[0] if self._items else None


class _FakeExecuteResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _FakeScalarsResult(self._items)


class _FakeSession:
    def __init__(self, data_records):
        self._data_records = data_records
        self.committed = False

    async def execute(self, _query):
        return _FakeExecuteResult(self._data_records)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeEngine:
    def __init__(self, session):
        self._session = session

    def get_async_session(self):
        return self._session


class _NoOpAsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# Tests for _forget_dataset_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forget_dataset_memory_clears_graph_and_resets_pipeline(monkeypatch):
    """memory_only=True with dataset clears graph/vector and resets pipeline_status."""
    other_dataset_id = str(uuid4())
    pipeline_status_a = {
        "cognify": {str(DATASET_ID): "COMPLETE", other_dataset_id: "COMPLETE"},
    }
    pipeline_status_b = {
        "cognify": {str(DATASET_ID): "COMPLETE"},
    }
    data_a = _make_data_record(DATA_ID_A, pipeline_status_a)
    data_b = _make_data_record(DATA_ID_B, pipeline_status_b)

    session = _FakeSession([data_a, data_b])
    engine = _FakeEngine(session)

    mock_delete = AsyncMock()
    mock_reset_status = AsyncMock()
    monkeypatch.setattr(
        forget_module,
        "_resolve_dataset_id",
        AsyncMock(return_value=DATASET_ID),
    )

    with (
        patch.object(
            delete_dataset_nodes_and_edges_module,
            "delete_dataset_nodes_and_edges",
            mock_delete,
        ),
        patch(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            return_value=engine,
        ),
        patch.object(
            reset_dataset_pipeline_run_status_module,
            "reset_dataset_pipeline_run_status",
            mock_reset_status,
        ),
        patch("sqlalchemy.orm.attributes.flag_modified"),
    ):
        result = await forget_module._forget_dataset_memory(str(DATASET_ID), USER)

    assert result["status"] == "success"
    assert result["dataset_id"] == str(DATASET_ID)
    assert result["data_records_reset"] == 2

    mock_delete.assert_awaited_once_with(DATASET_ID, USER.id)
    mock_reset_status.assert_awaited_once_with(
        dataset_id=DATASET_ID,
        user=USER,
        pipeline_names=["cognify_pipeline"],
    )

    # pipeline_status should have dataset entry removed
    assert str(DATASET_ID) not in data_a.pipeline_status["cognify"]
    assert str(DATASET_ID) not in data_b.pipeline_status["cognify"]

    # Other dataset entries should be preserved
    assert len(data_a.pipeline_status["cognify"]) == 1

    assert session.committed


@pytest.mark.asyncio
async def test_forget_dataset_memory_skips_records_without_pipeline_status(monkeypatch):
    """Records with no pipeline_status should not cause errors."""
    data_no_status = _make_data_record(DATA_ID_A, None)
    data_empty_status = _make_data_record(DATA_ID_B, {})

    session = _FakeSession([data_no_status, data_empty_status])
    engine = _FakeEngine(session)

    monkeypatch.setattr(
        forget_module,
        "_resolve_dataset_id",
        AsyncMock(return_value=DATASET_ID),
    )
    mock_reset_status = AsyncMock()

    with (
        patch.object(
            delete_dataset_nodes_and_edges_module,
            "delete_dataset_nodes_and_edges",
            AsyncMock(),
        ),
        patch(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            return_value=engine,
        ),
        patch.object(
            reset_dataset_pipeline_run_status_module,
            "reset_dataset_pipeline_run_status",
            mock_reset_status,
        ),
    ):
        result = await forget_module._forget_dataset_memory(str(DATASET_ID), USER)

    assert result["status"] == "success"
    assert result["data_records_reset"] == 2
    mock_reset_status.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests for _forget_data_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forget_data_memory_clears_graph_and_resets_pipeline(monkeypatch):
    """memory_only=True with dataset+data_id clears graph/vector for the item."""
    other_dataset_id = str(uuid4())
    pipeline_status = {
        "add_pipeline": {str(DATASET_ID): "COMPLETE"},
        "cognify_pipeline": {str(DATASET_ID): "COMPLETE", other_dataset_id: "COMPLETE"},
    }
    data_record = _make_data_record(DATA_ID_A, pipeline_status)

    session = _FakeSession([data_record])
    engine = _FakeEngine(session)

    mock_delete = AsyncMock()
    monkeypatch.setattr(
        forget_module,
        "_resolve_dataset_id",
        AsyncMock(return_value=DATASET_ID),
    )

    with (
        patch.object(
            delete_data_nodes_and_edges_module,
            "delete_data_nodes_and_edges",
            mock_delete,
        ),
        patch(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            return_value=engine,
        ),
        patch("sqlalchemy.orm.attributes.flag_modified"),
    ):
        result = await forget_module._forget_data_memory(DATA_ID_A, str(DATASET_ID), USER)

    assert result["status"] == "success"
    assert result["data_id"] == str(DATA_ID_A)
    assert result["dataset_id"] == str(DATASET_ID)

    mock_delete.assert_awaited_once_with(DATASET_ID, DATA_ID_A, USER.id)

    # only cognify status should be reset for this dataset
    assert str(DATASET_ID) not in data_record.pipeline_status["cognify_pipeline"]
    # other cognify dataset entries should remain
    assert other_dataset_id in data_record.pipeline_status["cognify_pipeline"]
    # add pipeline status should remain untouched
    assert str(DATASET_ID) in data_record.pipeline_status["add_pipeline"]
    assert session.committed


@pytest.mark.asyncio
async def test_forget_data_memory_no_record_found(monkeypatch):
    """When data record is not found, should still succeed (graph cleanup done)."""
    session = _FakeSession([])  # No data records
    engine = _FakeEngine(session)

    monkeypatch.setattr(
        forget_module,
        "_resolve_dataset_id",
        AsyncMock(return_value=DATASET_ID),
    )

    with (
        patch.object(
            delete_data_nodes_and_edges_module,
            "delete_data_nodes_and_edges",
            AsyncMock(),
        ),
        patch(
            "cognee.infrastructure.databases.relational.get_relational_engine",
            return_value=engine,
        ),
    ):
        result = await forget_module._forget_data_memory(DATA_ID_A, str(DATASET_ID), USER)

    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Tests for forget() routing with memory_only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forget_memory_only_without_dataset_raises(monkeypatch):
    """memory_only=True without dataset should raise ValueError."""
    monkeypatch.setattr(
        forget_module,
        "_forget_dataset_memory",
        AsyncMock(),
    )

    with (
        patch("cognee.shared.utils.send_telemetry", lambda *a, **kw: None),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch("cognee.low_level.setup", AsyncMock()),
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=USER)),
        patch.object(
            forget_module, "set_database_global_context_variables", return_value=_NoOpAsyncContext()
        ),
    ):
        with pytest.raises(ValueError, match="memory_only requires dataset"):
            await forget_module.forget(memory_only=True)


@pytest.mark.asyncio
async def test_forget_routes_to_dataset_memory(monkeypatch):
    """memory_only=True with dataset routes to _forget_dataset_memory."""
    mock_forget_dataset_memory = AsyncMock(
        return_value={"status": "success", "dataset_id": str(DATASET_ID), "data_records_reset": 1}
    )
    monkeypatch.setattr(
        forget_module,
        "_forget_dataset_memory",
        mock_forget_dataset_memory,
    )

    with (
        patch("cognee.shared.utils.send_telemetry", lambda *a, **kw: None),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch("cognee.low_level.setup", AsyncMock()),
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=USER)),
        patch.object(
            forget_module, "set_database_global_context_variables", return_value=_NoOpAsyncContext()
        ),
    ):
        result = await forget_module.forget(dataset="my-dataset", memory_only=True)

    assert result["status"] == "success"
    mock_forget_dataset_memory.assert_awaited_once_with("my-dataset", USER)


@pytest.mark.asyncio
async def test_forget_routes_to_data_memory(monkeypatch):
    """memory_only=True with dataset+data_id routes to _forget_data_memory."""
    mock_forget_data_memory = AsyncMock(
        return_value={"status": "success", "data_id": str(DATA_ID_A), "dataset_id": str(DATASET_ID)}
    )
    monkeypatch.setattr(
        forget_module,
        "_forget_data_memory",
        mock_forget_data_memory,
    )

    with (
        patch("cognee.shared.utils.send_telemetry", lambda *a, **kw: None),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch("cognee.low_level.setup", AsyncMock()),
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=USER)),
        patch.object(
            forget_module, "set_database_global_context_variables", return_value=_NoOpAsyncContext()
        ),
    ):
        result = await forget_module.forget(
            dataset="my-dataset", data_id=DATA_ID_A, memory_only=True
        )

    assert result["status"] == "success"
    mock_forget_data_memory.assert_awaited_once_with(DATA_ID_A, "my-dataset", USER)


# ---------------------------------------------------------------------------
# Tests for telemetry target labels
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forget_telemetry_target_labels(monkeypatch):
    """Verify correct telemetry target strings for memory_only combinations."""
    captured_targets = []

    def fake_telemetry(_event, _user, additional_properties=None):
        if additional_properties:
            captured_targets.append(additional_properties.get("target"))

    monkeypatch.setattr(
        forget_module,
        "_forget_dataset_memory",
        AsyncMock(return_value={"status": "success", "dataset_id": "x", "data_records_reset": 0}),
    )
    monkeypatch.setattr(
        forget_module,
        "_forget_data_memory",
        AsyncMock(return_value={"status": "success", "data_id": "x", "dataset_id": "x"}),
    )

    with (
        patch("cognee.shared.utils.send_telemetry", fake_telemetry),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch("cognee.low_level.setup", AsyncMock()),
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=USER)),
        patch.object(
            forget_module, "set_database_global_context_variables", return_value=_NoOpAsyncContext()
        ),
    ):
        await forget_module.forget(dataset="ds", memory_only=True)
        await forget_module.forget(dataset="ds", data_id=DATA_ID_A, memory_only=True)

    assert captured_targets == ["dataset_memory_only", "data_item_memory_only"]
