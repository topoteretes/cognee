import importlib
import types
from uuid import uuid4

import pytest

from cognee.modules.data.models import GraphMetrics

metrics_mod = importlib.import_module("cognee.modules.metrics.operations.get_pipeline_run_metrics")


def _make_pipeline_run():
    return types.SimpleNamespace(pipeline_run_id=uuid4())


@pytest.mark.asyncio
async def test_get_pipeline_run_metrics_returns_cached_metrics(monkeypatch, caplog):
    pipeline_run = _make_pipeline_run()
    cached = GraphMetrics(
        id=pipeline_run.pipeline_run_id,
        num_tokens=10,
        num_nodes=3,
        num_edges=2,
        mean_degree=1.0,
        edge_density=0.5,
        num_connected_components=1,
        sizes_of_connected_components=[3],
        num_selfloops=0,
        diameter=2,
        avg_shortest_path_length=1.0,
        avg_clustering=0.0,
    )

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, _query):
            result = types.SimpleNamespace()
            result.scalars = lambda: types.SimpleNamespace(first=lambda: cached)
            return result

        async def commit(self):
            return None

        def add(self, _metrics):
            return None

    class DummyEngine:
        def get_async_session(self):
            return DummySession()

    async def fake_get_graph_engine():
        raise AssertionError("graph engine should not be queried on cache hit")

    monkeypatch.setattr(
        metrics_mod,
        "get_relational_engine",
        lambda: DummyEngine(),
    )
    monkeypatch.setattr(metrics_mod, "get_graph_engine", fake_get_graph_engine)

    with caplog.at_level("INFO"):
        result = await metrics_mod.get_pipeline_run_metrics(pipeline_run, include_optional=False)

    assert result == [cached]
    assert any("cache hit" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_get_pipeline_run_metrics_computes_and_persists_on_cache_miss(monkeypatch, caplog):
    pipeline_run = _make_pipeline_run()
    added = []

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, _query):
            result = types.SimpleNamespace()
            result.scalars = lambda: types.SimpleNamespace(first=lambda: None)
            return result

        async def commit(self):
            return None

        def add(self, metrics):
            added.append(metrics)

    class DummyEngine:
        def get_async_session(self):
            return DummySession()

    class DummyGraphEngine:
        async def get_graph_metrics(self, include_optional):
            assert include_optional is True
            return {
                "num_nodes": 4,
                "num_edges": 5,
                "mean_degree": 2.0,
                "edge_density": 0.4,
                "num_connected_components": 1,
                "sizes_of_connected_components": [4],
                "num_selfloops": 0,
                "diameter": 3,
                "avg_shortest_path_length": 1.5,
                "avg_clustering": 0.1,
            }

    async def fake_fetch_token_count(_engine):
        return 42

    monkeypatch.setattr(
        metrics_mod,
        "get_relational_engine",
        lambda: DummyEngine(),
    )

    async def fake_get_graph_engine():
        return DummyGraphEngine()

    monkeypatch.setattr(
        metrics_mod,
        "get_graph_engine",
        fake_get_graph_engine,
    )
    monkeypatch.setattr(
        metrics_mod,
        "fetch_token_count",
        fake_fetch_token_count,
    )

    with caplog.at_level("WARNING"):
        result = await metrics_mod.get_pipeline_run_metrics(pipeline_run, include_optional=True)

    assert len(result) == 1
    assert result[0].num_tokens == 42
    assert result[0].num_nodes == 4
    assert len(added) == 1
    assert any("cache miss" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_get_pipeline_run_metrics_logs_and_reraises_graph_engine_errors(monkeypatch, caplog):
    pipeline_run = _make_pipeline_run()

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, _query):
            result = types.SimpleNamespace()
            result.scalars = lambda: types.SimpleNamespace(first=lambda: None)
            return result

        async def commit(self):
            return None

    async def failing_get_graph_engine():
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(
        metrics_mod,
        "get_relational_engine",
        lambda: types.SimpleNamespace(get_async_session=lambda: DummySession()),
    )
    monkeypatch.setattr(
        metrics_mod,
        "get_graph_engine",
        failing_get_graph_engine,
    )

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="graph unavailable"):
            await metrics_mod.get_pipeline_run_metrics(pipeline_run, include_optional=False)

    assert any(record.levelname == "ERROR" for record in caplog.records)
