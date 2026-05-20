"""Tests for UUID-string coercion at the cognee.forget() boundary.

The /v1/forget REST endpoint receives `dataset` as JSON, so a UUID is
deserialized as `str`. forget() must coerce UUID-shaped strings to UUID
*before* opening the database context, otherwise:

  1. _resolve_dataset_id sees `str` and takes the name-lookup path, missing
     a dataset addressed by id.
  2. set_database_global_context_variables -> get_or_create_dataset_database
     sees `str` and calls create_authorized_dataset(<uuid-string>, user),
     manufacturing a phantom dataset whose *name* is the UUID string of the
     dataset the caller meant to delete.

These tests pin the boundary contract: by the time forget() opens the dataset
context, a UUID-shaped string has been coerced to UUID; a real name stays a
string.
"""

import importlib
import pytest
from types import SimpleNamespace
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, patch

forget_module = importlib.import_module("cognee.api.v1.forget.forget")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")
low_level_module = importlib.import_module("cognee.low_level")


USER = SimpleNamespace(id=uuid4())


class _CapturingContextManager:
    """Stand-in for set_database_global_context_variables.

    Records the `dataset` argument it was constructed with so tests can
    assert what the real context manager would have received.
    """

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


def _patches(forget_dataset_mock):
    """Common patches: skip setup, remote client, default user; capture context manager."""
    return (
        patch.object(low_level_module, "setup", AsyncMock()),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(
            forget_module,
            "set_database_global_context_variables",
            _CapturingContextManager,
        ),
        patch.object(forget_module, "_forget_dataset", forget_dataset_mock),
    )


@pytest.mark.asyncio
async def test_forget_coerces_uuid_string_before_context_and_routing():
    """A UUID-shaped string arrives at the context manager and _forget_dataset
    as a UUID — never as a string. This is the regression guard for the
    phantom-dataset side effect."""
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
        await forget_module.forget(dataset=str(dataset_id), user=USER)

    # Context manager saw a UUID, not a string.
    assert _CapturingContextManager.captured == [dataset_id]
    assert isinstance(_CapturingContextManager.captured[0], UUID)

    # _forget_dataset saw a UUID, not a string.
    forget_dataset.assert_awaited_once()
    passed_dataset = forget_dataset.call_args.args[0]
    assert passed_dataset == dataset_id
    assert isinstance(passed_dataset, UUID)


@pytest.mark.asyncio
async def test_forget_preserves_uuid_object_unchanged():
    """A UUID object passes through untouched."""
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
        await forget_module.forget(dataset=dataset_id, user=USER)

    assert _CapturingContextManager.captured == [dataset_id]
    assert isinstance(_CapturingContextManager.captured[0], UUID)
    passed_dataset = forget_dataset.call_args.args[0]
    assert isinstance(passed_dataset, UUID)


@pytest.mark.asyncio
async def test_forget_preserves_non_uuid_string_as_name():
    """A non-UUID string (real dataset name) stays a string so the name-lookup
    path downstream still works."""
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
    passed_dataset = forget_dataset.call_args.args[0]
    assert passed_dataset == "scientists"
    assert isinstance(passed_dataset, str)


@pytest.mark.asyncio
async def test_forget_passes_uuid_to_remote_client_when_present():
    """When a remote client is configured, the coerced UUID is forwarded to it —
    the remote endpoint must also receive a UUID, not the original string."""
    dataset_id = uuid4()
    remote_client = SimpleNamespace(forget=AsyncMock(return_value={"status": "success"}))

    with (
        patch.object(serve_state_module, "get_remote_client", return_value=remote_client),
    ):
        await forget_module.forget(dataset=str(dataset_id), user=USER)

    remote_client.forget.assert_awaited_once()
    forwarded_dataset = remote_client.forget.call_args.kwargs["dataset"]
    assert forwarded_dataset == dataset_id
    assert isinstance(forwarded_dataset, UUID)
