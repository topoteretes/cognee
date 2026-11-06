"""``forget(data_id=...)``: authors with ``write`` may forget their own items.

``_resolve_dataset_id`` requires ``delete`` on the dataset. Given a ``data_id``
it also accepts ``write`` when the caller added that item (``Data.owner_id``);
otherwise the original ``delete`` denial is raised unchanged. Dataset-wide
forgets and ``memory_only`` (which ``update()`` rebuilds through) stay
delete-only.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.users.exceptions import PermissionDeniedError

forget_module = importlib.import_module("cognee.api.v1.forget.forget")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")
low_level_module = importlib.import_module("cognee.low_level")

DATASET_NAME = "team"


class _Scenario:
    def __init__(self, granted, data_owner="caller"):
        self.user = SimpleNamespace(id=uuid4())
        self.dataset = SimpleNamespace(id=uuid4(), name=DATASET_NAME)
        self.data_id = uuid4()
        owner_id = self.user.id if data_owner == "caller" else uuid4()
        self.data = SimpleNamespace(id=self.data_id, dataset_id=self.dataset.id, owner_id=owner_id)
        self.granted = granted
        self.get_data = AsyncMock(return_value=self.data)

    async def _by_id(self, _user, _dataset_id, permission_type="read"):
        if permission_type in self.granted:
            return self.dataset
        raise PermissionDeniedError(f"no [{permission_type}] permission")

    async def _by_name(self, dataset_name, _user, permission_type):
        if dataset_name == DATASET_NAME and permission_type in self.granted:
            return self.dataset
        return None

    async def resolve(self, dataset_ref, data_id=None):
        with (
            patch(
                "cognee.modules.data.methods.get_authorized_dataset.get_authorized_dataset",
                AsyncMock(side_effect=self._by_id),
            ),
            patch(
                "cognee.modules.data.methods.get_authorized_dataset_by_name",
                AsyncMock(side_effect=self._by_name),
            ),
            patch("cognee.modules.data.methods.get_data", self.get_data),
        ):
            return await forget_module._resolve_dataset_id(dataset_ref, self.user, data_id=data_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("by_name", [False, True])
async def test_author_with_write_may_forget_own_item(by_name):
    scenario = _Scenario(granted={"read", "write"})
    dataset_ref = DATASET_NAME if by_name else scenario.dataset.id

    resolved = await scenario.resolve(dataset_ref, data_id=scenario.data_id)

    assert resolved == scenario.dataset.id
    scenario.get_data.assert_awaited_once_with(
        scenario.user.id, scenario.data_id, scenario.dataset.id, verify_owner=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("by_name", "denial"), [(False, PermissionDeniedError), (True, DatasetNotFoundError)]
)
async def test_write_without_delete_cannot_forget_another_users_item(by_name, denial):
    scenario = _Scenario(granted={"read", "write"}, data_owner="someone else")
    dataset_ref = DATASET_NAME if by_name else scenario.dataset.id

    # The original delete denial, unchanged: nothing says the item exists.
    with pytest.raises(denial):
        await scenario.resolve(dataset_ref, data_id=scenario.data_id)


@pytest.mark.asyncio
async def test_write_without_delete_cannot_forget_a_missing_item():
    scenario = _Scenario(granted={"read", "write"})
    scenario.get_data.return_value = None

    with pytest.raises(PermissionDeniedError):
        await scenario.resolve(scenario.dataset.id, data_id=scenario.data_id)


@pytest.mark.asyncio
async def test_read_only_cannot_forget_own_item():
    scenario = _Scenario(granted={"read"})

    with pytest.raises(PermissionDeniedError):
        await scenario.resolve(scenario.dataset.id, data_id=scenario.data_id)

    scenario.get_data.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("by_name", [False, True])
async def test_forgetting_a_dataset_still_requires_delete(by_name):
    scenario = _Scenario(granted={"read", "write"})
    dataset_ref = DATASET_NAME if by_name else scenario.dataset.id

    with pytest.raises((PermissionDeniedError, DatasetNotFoundError)):
        await scenario.resolve(dataset_ref)

    scenario.get_data.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("data_owner", ["caller", "someone else"])
async def test_delete_permission_holders_are_unaffected(data_owner):
    scenario = _Scenario(granted={"read", "write", "delete"}, data_owner=data_owner)

    assert await scenario.resolve(scenario.dataset.id, data_id=scenario.data_id) == (
        scenario.dataset.id
    )
    assert await scenario.resolve(scenario.dataset.id) == scenario.dataset.id
    # The ACL decides; the row is never consulted.
    scenario.get_data.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_only", [False, True])
async def test_forget_admits_authors_for_item_deletion_but_not_memory_only(memory_only):
    data_id, dataset_id = uuid4(), uuid4()
    resolve = AsyncMock(side_effect=PermissionDeniedError("denied"))

    with (
        patch.object(low_level_module, "setup", AsyncMock()),
        patch.object(serve_state_module, "get_remote_client", return_value=None),
        patch.object(forget_module, "_resolve_dataset_id", resolve),
        pytest.raises(PermissionDeniedError),
    ):
        await forget_module.forget(
            data_id=data_id,
            dataset_id=dataset_id,
            memory_only=memory_only,
            user=SimpleNamespace(id=uuid4()),
        )

    # memory_only resets an item for re-cognify (update() rebuilds through it),
    # so it keeps the plain delete check: no data_id, no author rule.
    assert resolve.await_args.kwargs["data_id"] == (None if memory_only else data_id)
