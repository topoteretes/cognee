"""``datasets.delete_data``: authors with ``write`` may delete their own items.

``delete`` on the dataset allows deleting any item. ``write`` alone allows
deleting only a ``Data`` row whose ``owner_id`` is the caller, so collaborators
on a shared dataset can remove what they added and nothing else. Dropping the
dataset (``delete_dataset_if_empty`` or ``empty_dataset``) still needs ``delete``.
"""

import importlib
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements
from cognee.modules.users.exceptions import PermissionDeniedError

datasets_module = importlib.import_module("cognee.api.v1.datasets.datasets")
data_methods_module = importlib.import_module("cognee.modules.data.methods")
search_operations_module = importlib.import_module("cognee.modules.search.operations")


@asynccontextmanager
async def _context(*_args, **_kwargs):
    yield


def _acl(dataset, granted):
    """``get_authorized_dataset`` stand-in honouring the granted permission types."""

    async def get_authorized_dataset(_user, _dataset_id, permission_type="read"):
        if permission_type in granted:
            return dataset
        raise PermissionDeniedError(f"no [{permission_type}] permission")

    return AsyncMock(side_effect=get_authorized_dataset)


class _Scenario:
    def __init__(self, granted, data_owner="caller", data_exists=True):
        self.user = SimpleNamespace(id=uuid4())
        self.dataset_id, self.data_id = uuid4(), uuid4()
        self.dataset = SimpleNamespace(id=self.dataset_id, owner_id=uuid4())
        owner_id = self.user.id if data_owner == "caller" else uuid4()
        self.data = (
            SimpleNamespace(id=self.data_id, dataset_id=self.dataset_id, owner_id=owner_id)
            if data_exists
            else None
        )
        self.get_authorized_dataset = _acl(self.dataset, granted)
        self.delete_rows = AsyncMock()
        self.delete_graph = AsyncMock(
            return_value=DeletedGraphElements(node_ids={"n1"}, edge_ids={"e1"})
        )
        self.delete_dataset = AsyncMock()

    async def delete_data(self, **kwargs):
        with ExitStack() as stack:
            for patcher in (
                patch.object(
                    datasets_module, "get_authorized_dataset", self.get_authorized_dataset
                ),
                patch.object(data_methods_module, "get_data", AsyncMock(return_value=self.data)),
                patch.object(datasets_module, "set_database_global_context_variables", _context),
                patch.object(
                    datasets_module, "has_data_related_nodes", AsyncMock(return_value=True)
                ),
                patch.object(datasets_module, "delete_data_nodes_and_edges", self.delete_graph),
                patch.object(
                    datasets_module, "retire_forgotten_decisions", AsyncMock(return_value=set())
                ),
                patch.object(
                    datasets_module, "_invalidate_sessions_for_deleted_data_nonfatal", AsyncMock()
                ),
                patch.object(data_methods_module, "delete_data", self.delete_rows),
                patch.object(datasets_module, "has_data", AsyncMock(return_value=False)),
                patch.object(datasets_module, "has_dataset_data", AsyncMock(return_value=False)),
                patch.object(data_methods_module, "delete_dataset", self.delete_dataset),
                patch.object(
                    search_operations_module, "delete_dataset_search_history", AsyncMock()
                ),
            ):
                stack.enter_context(patcher)
            return await datasets_module.datasets.delete_data(
                self.dataset_id, self.data_id, self.user, **kwargs
            )

    def assert_nothing_deleted(self):
        self.delete_graph.assert_not_awaited()
        self.delete_rows.assert_not_awaited()
        self.delete_dataset.assert_not_awaited()


@pytest.mark.asyncio
async def test_author_with_write_deletes_own_item():
    scenario = _Scenario(granted={"read", "write"})

    receipt = await scenario.delete_data()

    assert receipt["status"] == "success"
    assert receipt["data_remaining"] is False
    scenario.delete_rows.assert_awaited_once()


@pytest.mark.asyncio
async def test_write_without_delete_cannot_delete_another_users_item():
    scenario = _Scenario(granted={"read", "write"}, data_owner="someone else")

    with pytest.raises(UnauthorizedDataAccessError) as error:
        await scenario.delete_data()

    # Same answer as an inaccessible dataset: the denial reveals nothing about the item.
    assert str(scenario.dataset_id) in str(error.value.message)
    assert "not accessible" in error.value.message
    scenario.assert_nothing_deleted()


@pytest.mark.asyncio
async def test_write_without_delete_cannot_delete_an_id_with_no_data_row():
    # No Data row means no owner to check (custom graph model ids): delete only.
    scenario = _Scenario(granted={"read", "write"}, data_exists=False)

    with pytest.raises(UnauthorizedDataAccessError):
        await scenario.delete_data()

    scenario.assert_nothing_deleted()


@pytest.mark.asyncio
async def test_read_only_cannot_delete_own_item():
    scenario = _Scenario(granted={"read"})

    with pytest.raises(UnauthorizedDataAccessError):
        await scenario.delete_data()

    scenario.assert_nothing_deleted()


@pytest.mark.asyncio
async def test_author_with_write_cannot_drop_the_dataset_through_delete_dataset_if_empty():
    scenario = _Scenario(granted={"read", "write"})

    with pytest.raises(UnauthorizedDataAccessError):
        await scenario.delete_data(delete_dataset_if_empty=True)

    scenario.assert_nothing_deleted()


@pytest.mark.asyncio
@pytest.mark.parametrize("data_owner", ["caller", "someone else"])
async def test_delete_permission_still_deletes_any_item(data_owner):
    scenario = _Scenario(granted={"read", "write", "delete"}, data_owner=data_owner)

    receipt = await scenario.delete_data()

    assert receipt["status"] == "success"
    scenario.delete_rows.assert_awaited_once()
    assert [call.args[2] for call in scenario.get_authorized_dataset.await_args_list] == ["delete"]


@pytest.mark.asyncio
async def test_delete_permission_still_drops_the_empty_dataset():
    scenario = _Scenario(granted={"read", "write", "delete"}, data_owner="someone else")

    receipt = await scenario.delete_data(delete_dataset_if_empty=True)

    assert receipt["dataset_deleted"] is True
    scenario.delete_dataset.assert_awaited_once()


@pytest.mark.asyncio
async def test_emptying_a_dataset_still_requires_delete():
    scenario = _Scenario(granted={"read", "write"})

    with (
        patch.object(datasets_module, "get_authorized_dataset", scenario.get_authorized_dataset),
        pytest.raises(PermissionDeniedError),
    ):
        await datasets_module.datasets.empty_dataset(scenario.dataset_id, scenario.user)

    scenario.get_authorized_dataset.assert_awaited_once_with(
        scenario.user, scenario.dataset_id, "delete"
    )
