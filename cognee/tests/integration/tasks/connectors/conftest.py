"""Keep connector integration tests inside their disposable storage roots."""

import pytest


@pytest.fixture(autouse=True)
def isolated_connector_state(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    from cognee.context_global_variables import graph_db_config, vector_db_config
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.tasks.ingestion.get_dlt_destination import get_dlt_destination

    monkeypatch.setenv("DB_PATH", str(tmp_path / "databases"))
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    monkeypatch.setenv("PIPELINES_DIR", str(tmp_path / "dlt" / "pipelines"))

    def reset_cached_state():
        Container()[PipelineContext].deactivate()
        _create_graph_engine.cache_clear()
        _create_vector_engine.cache_clear()
        create_relational_engine.cache_clear()
        get_dlt_destination.cache_clear()
        graph_db_config.set(None)
        vector_db_config.set(None)

    reset_cached_state()
    yield
    reset_cached_state()
