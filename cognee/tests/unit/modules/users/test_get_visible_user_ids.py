"""Unit tests for cognee.modules.users.methods.get_visible_user_ids.

Shared across sessions, agents, and usage-reporting code — see the
review discussion on cognee/pull/4342 for why this was consolidated
out of three near-identical private copies.
"""

import sys
from uuid import uuid4

import pytest

import cognee.modules.users.methods.get_visible_user_ids  # noqa: F401 - registers the submodule

# `cognee.modules.users.methods.__init__` does
# `from .get_visible_user_ids import get_visible_user_ids`, which overwrites the
# `get_visible_user_ids` attribute on the package with the function — so an
# attribute-style import binds to the function, not the module. Go through
# sys.modules to get the actual module for patching.
module = sys.modules["cognee.modules.users.methods.get_visible_user_ids"]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _stmt):
        return _FakeResult(self._rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def get_async_session(self):
        return _FakeSession(self._rows)


class _Row:
    def __init__(self, id_):
        self.id = id_


@pytest.mark.asyncio
async def test_returns_only_self_when_no_child_agents(monkeypatch):
    user_id = uuid4()

    monkeypatch.setattr(module, "get_relational_engine", lambda: _FakeEngine([]))

    result = await module.get_visible_user_ids(user_id)

    assert result == [user_id]


@pytest.mark.asyncio
async def test_includes_child_agent_ids(monkeypatch):
    user_id = uuid4()
    child_id = uuid4()

    monkeypatch.setattr(module, "get_relational_engine", lambda: _FakeEngine([_Row(child_id)]))

    result = await module.get_visible_user_ids(user_id)

    assert result == [user_id, child_id]
