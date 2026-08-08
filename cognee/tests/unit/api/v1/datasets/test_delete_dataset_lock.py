"""Delete operations acquire the same per-dataset lock as pipeline runs.

While a pipeline run holds a dataset's lock (via
cognee.infrastructure.locks.get_dataset_lock, as run_pipeline_per_dataset
does), datasets.empty_dataset and datasets.delete_data on that dataset
must wait until the lock is released.
"""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import get_dataset_lock

datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
data_methods_module = importlib.import_module("cognee.modules.data.methods")

USER = SimpleNamespace(id=uuid4())


class _NoopContextManager:
    """Stand-in for set_database_global_context_variables."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _dataset_stub(dataset_id):
    return SimpleNamespace(id=dataset_id, owner_id=USER.id, name="test_dataset")


async def _assert_waits_for_pipeline_lock(dataset_id, run_delete, mutation_mock):
    """Hold the dataset's pipeline lock, assert run_delete blocks until release."""
    pipeline_lock = await get_dataset_lock(dataset_id)
    await pipeline_lock.acquire()

    delete_task = asyncio.create_task(run_delete())
    await asyncio.sleep(0.01)
    assert not mutation_mock.called, "delete must wait while the pipeline holds the lock"

    pipeline_lock.release()
    await asyncio.wait_for(delete_task, timeout=1)
    assert mutation_mock.called


@pytest.mark.asyncio
async def test_empty_dataset_waits_for_pipeline_lock():
    dataset_id = uuid4()
    delete_nodes_and_edges = AsyncMock()

    with (
        patch.object(
            datasets_module,
            "get_authorized_dataset",
            AsyncMock(return_value=_dataset_stub(dataset_id)),
        ),
        patch.object(datasets_module, "set_database_global_context_variables", _NoopContextManager),
        patch.object(datasets_module, "delete_dataset_nodes_and_edges", delete_nodes_and_edges),
        patch.object(datasets_module, "get_dataset_data", AsyncMock(return_value=[])),
        patch.object(
            data_methods_module,
            "delete_dataset",
            AsyncMock(return_value={"status": "success"}),
        ),
    ):
        await _assert_waits_for_pipeline_lock(
            dataset_id,
            lambda: datasets_module.datasets.empty_dataset(dataset_id, USER),
            delete_nodes_and_edges,
        )


@pytest.mark.asyncio
async def test_delete_data_waits_for_pipeline_lock():
    dataset_id = uuid4()
    delete_nodes_and_edges = AsyncMock()

    with (
        patch.object(
            datasets_module,
            "get_authorized_dataset",
            AsyncMock(return_value=_dataset_stub(dataset_id)),
        ),
        patch.object(datasets_module, "set_database_global_context_variables", _NoopContextManager),
        # No matching Data record -> custom-graph-model path, which only
        # calls delete_data_nodes_and_edges.
        patch.object(datasets_module, "get_dataset_data", AsyncMock(return_value=[])),
        patch.object(datasets_module, "delete_data_nodes_and_edges", delete_nodes_and_edges),
    ):
        await _assert_waits_for_pipeline_lock(
            dataset_id,
            lambda: datasets_module.datasets.delete_data(dataset_id, uuid4(), USER),
            delete_nodes_and_edges,
        )
