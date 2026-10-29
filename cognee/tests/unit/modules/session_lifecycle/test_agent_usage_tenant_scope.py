"""A tenant admin's cost view stops at the tenant's datasets (SDK-1237).

Runs against a real database: the fix is a SQL predicate (the member's
sessions joined to ``Dataset.tenant_id``), and the fake engine the other
agent_usage tests use ignores the statement it is handed.
"""

import asyncio
import os
import pathlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee
from cognee.modules.session_lifecycle import agent_usage

_SYSTEM_ROOT = str(
    pathlib.Path(
        os.path.join(
            pathlib.Path(__file__).parent.parent.parent,
            ".cognee_system/test_agent_usage_tenant_scope",
        )
    ).resolve()
)


@pytest.fixture(autouse=True, scope="module")
def _isolated_db():
    """Own database for this module, and the shared one back afterwards: the
    rows seeded here (one with no dataset) would otherwise be counted by the
    session-invalidation tests that scan every unattributed session."""
    from cognee.base_config import get_base_config
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    previous_root = get_base_config().system_root_directory
    cognee.config.system_root_directory(_SYSTEM_ROOT)
    create_relational_engine.cache_clear()

    async def _run():
        import cognee.modules.data.models
        import cognee.modules.session_lifecycle.models
        import cognee.modules.users.models
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())
    create_relational_engine.cache_clear()
    yield
    cognee.config.system_root_directory(previous_root)
    create_relational_engine.cache_clear()


async def _seed():
    """Admin of tenant A; a member of both tenants with one session in each and one unattributed."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.modules.session_lifecycle.models import SessionRecord

    admin_id, member_id = uuid4(), uuid4()
    tenant_a, tenant_b = uuid4(), uuid4()
    dataset_a, dataset_b = uuid4(), uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Dataset(id=dataset_a, name="ds_a", owner_id=member_id, tenant_id=tenant_a))
        session.add(Dataset(id=dataset_b, name="ds_b", owner_id=member_id, tenant_id=tenant_b))
        session.add_all(
            [
                SessionRecord(
                    session_id="member_in_a", user_id=member_id, dataset_id=dataset_a, cost_usd=1.0
                ),
                SessionRecord(
                    session_id="member_in_b", user_id=member_id, dataset_id=dataset_b, cost_usd=2.0
                ),
                SessionRecord(session_id="member_nowhere", user_id=member_id, cost_usd=4.0),
                SessionRecord(
                    session_id="admin_own", user_id=admin_id, dataset_id=dataset_a, cost_usd=8.0
                ),
            ]
        )
        await session.commit()

    return SimpleNamespace(
        admin_id=admin_id, member_id=member_id, tenant_a=tenant_a, tenant_b=tenant_b
    )


@pytest.mark.asyncio
async def test_tenant_admin_sees_members_sessions_on_the_tenants_datasets_only(monkeypatch):
    ids = await _seed()
    admin = SimpleNamespace(id=ids.admin_id, tenant_id=ids.tenant_a, email="admin@a.example")

    async def fake_get_users_in_tenant(_tenant_id, _user):
        return [
            {"id": str(ids.admin_id), "email": "admin@a.example", "roles": []},
            {"id": str(ids.member_id), "email": "member@example", "roles": []},
        ]

    async def no_persisted(_user_ids, active_only=False):
        return []

    monkeypatch.setattr(agent_usage, "get_users_in_tenant", fake_get_users_in_tenant)
    monkeypatch.setattr(agent_usage, "list_persisted_agent_connections", no_persisted)
    monkeypatch.setattr(agent_usage, "list_registered_agent_connections", list)

    result = await agent_usage.compute_cost_by_user_agent(
        user=admin, visible_user_ids=[ids.admin_id], permitted_dataset_ids=[], since=None
    )

    by_user = {row["user_id"]: row for row in result}
    assert by_user[str(ids.admin_id)]["cost_usd"] == pytest.approx(8.0)
    # Only the tenant-A session: not the tenant-B one, not the unattributed one.
    assert by_user[str(ids.member_id)]["session_count"] == 1
    assert by_user[str(ids.member_id)]["cost_usd"] == pytest.approx(1.0)
