"""Tests for ``create_authorized_dataset`` parent-user auto-share.

When a user has ``parent_user_id`` set (typical for agent/service
identities owned by a human), the parent should receive full permissions
on any dataset the agent creates.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.data.methods import create_authorized_dataset

module_under_test = importlib.import_module(create_authorized_dataset.__module__)

PERMS = ("read", "write", "delete", "share")


def _install_fakes(monkeypatch, recorded_calls, parent_resolver):
    async def fake_create_dataset(_name, _user, _session):
        return SimpleNamespace(id=uuid4())

    async def fake_give_permission(principal, dataset_id, permission_name):
        recorded_calls.append((principal.id, dataset_id, permission_name))

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _FakeEngine:
        def get_async_session(self):
            return _FakeSession()

    def fake_get_engine():
        return _FakeEngine()

    monkeypatch.setattr(module_under_test, "create_dataset", fake_create_dataset)
    monkeypatch.setattr(module_under_test, "give_permission_on_dataset", fake_give_permission)
    monkeypatch.setattr(module_under_test, "get_relational_engine", fake_get_engine)
    monkeypatch.setattr(module_under_test, "get_user", parent_resolver)


@pytest.mark.asyncio
async def test_parent_user_receives_full_permissions(monkeypatch):
    parent_id = uuid4()
    agent_id = uuid4()
    recorded = []

    async def fake_get_user(user_id):
        assert user_id == parent_id
        return SimpleNamespace(id=parent_id)

    _install_fakes(monkeypatch, recorded, fake_get_user)

    agent = SimpleNamespace(id=agent_id, parent_user_id=parent_id)
    await module_under_test.create_authorized_dataset("demo", agent)

    agent_perms = sorted({perm for (pid, _, perm) in recorded if pid == agent_id})
    parent_perms = sorted({perm for (pid, _, perm) in recorded if pid == parent_id})

    assert agent_perms == sorted(PERMS)
    assert parent_perms == sorted(PERMS)


@pytest.mark.asyncio
async def test_no_parent_skips_extra_grants(monkeypatch):
    agent_id = uuid4()
    recorded = []

    async def fake_get_user(_user_id):  # should never be called
        raise AssertionError("get_user must not run when parent_user_id is None")

    _install_fakes(monkeypatch, recorded, fake_get_user)

    agent = SimpleNamespace(id=agent_id, parent_user_id=None)
    await module_under_test.create_authorized_dataset("demo", agent)

    principals = {pid for (pid, _, _) in recorded}
    assert principals == {agent_id}


@pytest.mark.asyncio
async def test_self_parent_is_ignored(monkeypatch):
    """A user that points parent_user_id at itself should not be double-granted."""
    user_id = uuid4()
    recorded = []

    async def fake_get_user(_user_id):
        raise AssertionError("get_user must not run when parent equals user")

    _install_fakes(monkeypatch, recorded, fake_get_user)

    user = SimpleNamespace(id=user_id, parent_user_id=user_id)
    await module_under_test.create_authorized_dataset("demo", user)

    perms = [perm for (pid, _, perm) in recorded if pid == user_id]
    assert sorted(perms) == sorted(PERMS)  # exactly one grant per permission


@pytest.mark.asyncio
async def test_missing_parent_does_not_block_creation(monkeypatch):
    """If parent_user_id points to a deleted user, dataset creation still succeeds."""
    parent_id = uuid4()
    agent_id = uuid4()
    recorded = []

    async def fake_get_user(_user_id):
        raise EntityNotFoundError(message="nope")

    _install_fakes(monkeypatch, recorded, fake_get_user)

    agent = SimpleNamespace(id=agent_id, parent_user_id=parent_id)
    dataset = await module_under_test.create_authorized_dataset("demo", agent)

    assert dataset is not None
    parent_perms = [perm for (pid, _, perm) in recorded if pid == parent_id]
    assert parent_perms == []
