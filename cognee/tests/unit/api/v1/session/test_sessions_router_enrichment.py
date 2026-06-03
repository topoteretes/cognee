from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.api.v1.sessions.routers import get_sessions_router as sessions_router_module


@pytest.mark.asyncio
async def test_generated_by_for_user_ids_marks_agent_owners(monkeypatch):
    human_id = uuid4()
    agent_by_parent_id = uuid4()
    agent_by_email_id = uuid4()
    parent_id = uuid4()
    rows = [
        SimpleNamespace(id=human_id, parent_user_id=None, email="human@example.com"),
        SimpleNamespace(
            id=agent_by_parent_id,
            parent_user_id=parent_id,
            email="worker@example.com",
        ),
        SimpleNamespace(id=agent_by_email_id, parent_user_id=None, email="helper@cognee.agent"),
    ]

    class FakeResult:
        def all(self):
            return rows

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def execute(self, _stmt):
            return FakeResult()

    fake_engine = SimpleNamespace(get_async_session=lambda: FakeSession())
    monkeypatch.setattr(
        sessions_router_module,
        "get_relational_engine",
        lambda: fake_engine,
    )

    generated_by = await sessions_router_module._generated_by_for_user_ids(
        [str(human_id), str(agent_by_parent_id), str(agent_by_email_id), "not-a-uuid"]
    )

    assert generated_by[str(human_id)] == "user"
    assert generated_by[str(agent_by_parent_id)] == "agent"
    assert generated_by[str(agent_by_email_id)] == "agent"


@pytest.mark.asyncio
async def test_generated_by_for_user_ids_ignores_invalid_ids(monkeypatch):
    fake_engine = SimpleNamespace(get_async_session=lambda: None)
    monkeypatch.setattr(
        sessions_router_module,
        "get_relational_engine",
        lambda: fake_engine,
    )

    generated_by = await sessions_router_module._generated_by_for_user_ids(["not-a-uuid"])

    assert generated_by == {}
