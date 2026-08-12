"""Shared fixtures for pipeline-runner unit tests.

The run_tasks generator needs the same mock plumbing everywhere it is driven
directly: a relational session that returns the dataset, a no-op database
context, and captured run-lifecycle log calls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


@pytest.fixture
def runner_plumbing(monkeypatch):
    """Wire a runner module's collaborators; returns the captured log mocks.

    Usage: ``logs = runner_plumbing(module, dataset)`` — then drive the
    module's generator and assert on ``logs.start`` / ``logs.complete`` /
    ``logs.error`` await counts and arguments.
    """

    def _wire(module, dataset):
        session = MagicMock()
        session.get = AsyncMock(return_value=dataset)
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        engine = MagicMock(spec=["get_async_session"])  # no push_to_s3 attribute
        engine.get_async_session.return_value = session_ctx
        monkeypatch.setattr(module, "get_relational_engine", lambda: engine)

        db_ctx = MagicMock()
        db_ctx.return_value.__aenter__ = AsyncMock(return_value=None)
        db_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(module, "set_database_global_context_variables", db_ctx)

        logs = SimpleNamespace(
            start=AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4())),
            complete=AsyncMock(),
            error=AsyncMock(),
        )
        monkeypatch.setattr(module, "log_pipeline_run_start", logs.start)
        monkeypatch.setattr(module, "log_pipeline_run_complete", logs.complete)
        monkeypatch.setattr(module, "log_pipeline_run_error", logs.error)

        if hasattr(module, "get_graph_engine"):
            monkeypatch.setattr(
                module, "get_graph_engine", AsyncMock(return_value=SimpleNamespace())
            )

        async def _identity(data):
            return data

        monkeypatch.setattr(module, "resolve_data_directories", _identity)
        return logs

    return _wire
