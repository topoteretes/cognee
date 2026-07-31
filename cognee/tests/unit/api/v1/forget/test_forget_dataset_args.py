"""Tests for explicit dataset/dataset_id handling in cognee.forget()."""

import importlib
import pytest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, patch

forget_module = importlib.import_module("cognee.api.v1.forget.forget")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")
low_level_module = importlib.import_module("cognee.low_level")


USER = SimpleNamespace(id=uuid4())


class _CapturingContextManager:
    """Stand-in for set_database_global_context_variables."""

    captured = []

    def __init__(self, dataset, user_id):
        type(self).captured.append(dataset)
        self.dataset = dataset
        self.user_id = user_id

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture(autouse=True)
def _reset_captures():
    _CapturingContextManager.captured = []
    yield


@pytest.mark.asyncio
async def test_forget_routes_dataset_id_without_name_inference():
    dataset_id = uuid4()
    forget_dataset = AsyncMock(return_value={"status": "success", "dataset_id": str(dataset_id)})

    with (
        patch.object(low_level_module, "setup", AsyncMock()),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(
            forget_module,
            "set_database_global_context_variables",
            _CapturingContextManager,
        ),
        patch.object(forget_module, "_forget_dataset", forget_dataset),
    ):
        await forget_module.forget(dataset_id=dataset_id, user=USER)

    assert _CapturingContextManager.captured == [dataset_id]
    forget_dataset.assert_awaited_once()
    assert forget_dataset.call_args.args[0] == dataset_id


@pytest.mark.asyncio
async def test_forget_routes_dataset_as_name():
    forget_dataset = AsyncMock(return_value={"status": "success", "dataset_id": str(uuid4())})

    with (
        patch.object(low_level_module, "setup", AsyncMock()),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(
            forget_module,
            "set_database_global_context_variables",
            _CapturingContextManager,
        ),
        patch.object(forget_module, "_forget_dataset", forget_dataset),
    ):
        await forget_module.forget(dataset="scientists", user=USER)

    assert _CapturingContextManager.captured == ["scientists"]
    forget_dataset.assert_awaited_once()
    assert forget_dataset.call_args.args[0] == "scientists"


@pytest.mark.asyncio
async def test_forget_passes_dataset_id_to_remote_client():
    dataset_id = uuid4()
    remote_client = SimpleNamespace(forget=AsyncMock(return_value={"status": "success"}))

    with patch.object(serve_state_module, "get_remote_client", return_value=remote_client):
        await forget_module.forget(dataset_id=dataset_id, user=USER)

    remote_client.forget.assert_awaited_once()
    kwargs = remote_client.forget.call_args.kwargs
    assert kwargs["dataset_id"] == dataset_id
    assert kwargs["dataset"] is None


@pytest.mark.asyncio
async def test_forget_rejects_both_dataset_and_dataset_id():
    with pytest.raises(ValueError, match="either dataset or dataset_id"):
        await forget_module.forget(dataset="scientists", dataset_id=uuid4(), user=USER)
