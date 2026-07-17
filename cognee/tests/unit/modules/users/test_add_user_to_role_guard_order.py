"""Regression test: add_user_to_role must check existence before dereferencing.

add_user_to_role fetches ``user``, ``role`` and ``role``'s ``tenant`` and then
checks ``if not user`` / ``elif not role``. The fetch order used to dereference
``role.tenant_id`` (to look up the tenant) and ``user.awaitable_attrs`` *before*
those guards ran, so a non-existent role/user raised ``AttributeError`` (HTTP 500
through the permissions router) instead of the intended ``RoleNotFoundError`` /
``UserNotFoundError`` (HTTP 404). The guards are now evaluated right after each
fetch, before any attribute access.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.users.exceptions import RoleNotFoundError, UserNotFoundError

add_user_to_role_module = importlib.import_module(
    "cognee.modules.users.roles.methods.add_user_to_role"
)


class _Result:
    def __init__(self, value):
        self._value = value

    def scalars(self):
        return self

    def first(self):
        return self._value


class _FakeSession:
    """Returns the queued results in execute() call order (user, role, tenant)."""

    def __init__(self, results):
        self._results = list(results)
        self._i = 0

    async def execute(self, _stmt):
        result = _Result(self._results[self._i])
        self._i += 1
        return result

    async def commit(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, results):
        self._results = results

    def get_async_session(self):
        return _FakeSession(self._results)


@pytest.mark.asyncio
async def test_missing_user_raises_user_not_found(monkeypatch):
    # user -> None; role/tenant rows present so the buggy code would still reach
    # user.awaitable_attrs (and raise AttributeError) instead of UserNotFoundError.
    role_row = SimpleNamespace(tenant_id=uuid4())
    tenant_row = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    monkeypatch.setattr(
        add_user_to_role_module,
        "get_relational_engine",
        lambda: _FakeEngine([None, role_row, tenant_row]),
    )

    with pytest.raises(UserNotFoundError):
        await add_user_to_role_module.add_user_to_role(uuid4(), uuid4(), uuid4())


@pytest.mark.asyncio
async def test_missing_role_raises_role_not_found(monkeypatch):
    # user present, role -> None; the buggy code dereferences role.tenant_id first.
    user_row = SimpleNamespace(id=uuid4())
    tenant_row = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    monkeypatch.setattr(
        add_user_to_role_module,
        "get_relational_engine",
        lambda: _FakeEngine([user_row, None, tenant_row]),
    )

    with pytest.raises(RoleNotFoundError):
        await add_user_to_role_module.add_user_to_role(uuid4(), uuid4(), uuid4())
