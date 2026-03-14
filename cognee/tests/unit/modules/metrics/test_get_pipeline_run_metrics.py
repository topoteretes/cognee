import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

metrics_ops = importlib.import_module("cognee.modules.metrics.operations.get_pipeline_run_metrics")


class _ExecuteResult:
    def __init__(self, value):
        self._value = value

    def scalars(self):
        return self

    def first(self):
        return self._value


class _FakeSession:
    def __init__(self, existing_metrics):
        self._existing_metrics = existing_metrics
        self.committed = False
        self.added = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, _stmt):
        return _ExecuteResult(self._existing_metrics)

    def add(self, item):
        self.added.append(item)

    async def commit(self):
        self.committed = True


class _FakeEngine:
    def __init__(self, session):
        self._session = session

    def get_async_session(self):
        return self._session


@pytest.mark.asyncio
async def test_get_pipeline_run_metrics_logs_cache_hit(monkeypatch):
    existing_metrics = object()
    fake_session = _FakeSession(existing_metrics=existing_metrics)
    fake_db_engine = _FakeEngine(fake_session)
    fake_graph_engine = SimpleNamespace(get_graph_metrics=AsyncMock())
    mock_logger = MagicMock()

    monkeypatch.setattr(metrics_ops, "logger", mock_logger)
    monkeypatch.setattr(metrics_ops, "get_relational_engine", lambda: fake_db_engine)
    monkeypatch.setattr(metrics_ops, "get_graph_engine", AsyncMock(return_value=fake_graph_engine))

    pipeline_run = SimpleNamespace(pipeline_run_id="run-1")
    result = await metrics_ops.get_pipeline_run_metrics(pipeline_run, include_optional=False)

    assert result == [existing_metrics]
    assert fake_session.committed is True
    fake_graph_engine.get_graph_metrics.assert_not_awaited()
    assert any("Cache hit" in call.args[0] for call in mock_logger.debug.call_args_list)


@pytest.mark.asyncio
async def test_get_pipeline_run_metrics_logs_cache_miss(monkeypatch):
    fake_session = _FakeSession(existing_metrics=None)
    fake_db_engine = _FakeEngine(fake_session)
    fake_graph_engine = SimpleNamespace(
        get_graph_metrics=AsyncMock(
            return_value={
                "num_nodes": 3,
                "num_edges": 2,
                "mean_degree": 0.67,
                "edge_density": 0.5,
                "num_connected_components": 1,
                "sizes_of_connected_components": [3],
                "num_selfloops": 0,
                "diameter": 2,
                "avg_shortest_path_length": 1.33,
                "avg_clustering": 0.0,
            }
        )
    )
    mock_logger = MagicMock()
    mock_fetch_token_count = AsyncMock(return_value=42)

    monkeypatch.setattr(metrics_ops, "logger", mock_logger)
    monkeypatch.setattr(metrics_ops, "fetch_token_count", mock_fetch_token_count)
    monkeypatch.setattr(metrics_ops, "get_relational_engine", lambda: fake_db_engine)
    monkeypatch.setattr(metrics_ops, "get_graph_engine", AsyncMock(return_value=fake_graph_engine))

    pipeline_run = SimpleNamespace(pipeline_run_id="run-2")
    result = await metrics_ops.get_pipeline_run_metrics(pipeline_run, include_optional=True)

    assert fake_session.committed is True
    assert len(fake_session.added) == 1
    assert result == fake_session.added
    fake_graph_engine.get_graph_metrics.assert_awaited_once_with(True)
    mock_fetch_token_count.assert_awaited_once_with(fake_db_engine)
    assert any("Cache miss" in call.args[0] for call in mock_logger.warning.call_args_list)


@pytest.mark.asyncio
async def test_get_pipeline_run_metrics_logs_error(monkeypatch):
    fake_session = _FakeSession(existing_metrics=None)
    fake_db_engine = _FakeEngine(fake_session)
    fake_graph_engine = SimpleNamespace(get_graph_metrics=AsyncMock(side_effect=RuntimeError("boom")))
    mock_logger = MagicMock()

    monkeypatch.setattr(metrics_ops, "logger", mock_logger)
    monkeypatch.setattr(metrics_ops, "get_relational_engine", lambda: fake_db_engine)
    monkeypatch.setattr(metrics_ops, "get_graph_engine", AsyncMock(return_value=fake_graph_engine))

    pipeline_run = SimpleNamespace(pipeline_run_id="run-3")
    with pytest.raises(RuntimeError, match="boom"):
        await metrics_ops.get_pipeline_run_metrics(pipeline_run, include_optional=False)

    assert mock_logger.error.called
    assert mock_logger.error.call_args.kwargs.get("exc_info") is True
