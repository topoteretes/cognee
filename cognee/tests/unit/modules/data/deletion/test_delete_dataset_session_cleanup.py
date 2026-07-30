"""delete_dataset must clean up the dataset's sessions BEFORE deleting the dataset.

Session cleanup locates sessions through their binding rows
(session_records.dataset_id). If it ran after the dataset row was deleted,
a crash in between would leave orphans that are unfindable forever — the
dataset is gone, so nothing ever asks to clean its sessions again. Running
cleanup first keeps a crash retryable: the dataset survives and re-running
the delete finishes the job.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

delete_dataset_module = importlib.import_module("cognee.modules.data.methods.delete_dataset")
metrics_module = importlib.import_module("cognee.modules.session_lifecycle.metrics")


class _SessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        return False


def _relational_session_mock():
    session = MagicMock()
    session.execute = AsyncMock(
        return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        )
    )
    session.scalar = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.get = AsyncMock(return_value=None)
    return session


@pytest.mark.asyncio
async def test_session_cleanup_runs_before_any_relational_access():
    dataset = SimpleNamespace(id=uuid4())
    events = []

    engine = MagicMock()
    engine.engine.dialect.name = "postgresql"

    def get_async_session():
        events.append("relational")
        return _SessionContext(_relational_session_mock())

    engine.get_async_session = MagicMock(side_effect=get_async_session)

    async def record_session_cleanup(dataset_id):
        events.append("sessions")

    with (
        patch.object(delete_dataset_module, "get_relational_engine", return_value=engine),
        patch.object(
            metrics_module,
            "delete_sessions_for_dataset",
            AsyncMock(side_effect=record_session_cleanup),
        ) as cleanup,
    ):
        await delete_dataset_module.delete_dataset(dataset)

    cleanup.assert_awaited_once_with(dataset.id)
    assert "sessions" in events and "relational" in events
    assert events[0] == "sessions", (
        f"session cleanup must run before the dataset row can be deleted; observed order: {events}"
    )
