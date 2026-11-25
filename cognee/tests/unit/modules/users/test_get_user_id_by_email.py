"""POST /users/get-user-id resolves only users the caller may see (SDK-1237).

Runs against a real database: the rule is a SQL predicate over
``user_tenants``. A caller in tenants finds their tenants' members; a caller
in no tenant finds users in no tenant; everyone finds themselves; nobody
confirms an address outside that.
"""

import asyncio
import os
import pathlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee
from cognee.modules.users.methods.get_user_id_by_email import get_user_id_by_email

_SYSTEM_ROOT = str(
    pathlib.Path(
        os.path.join(
            pathlib.Path(__file__).parent.parent.parent,
            ".cognee_system/test_get_user_id_by_email",
        )
    ).resolve()
)


@pytest.fixture(autouse=True, scope="module")
def _isolated_db():
    from cognee.base_config import get_base_config
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    previous_root = get_base_config().system_root_directory
    cognee.config.system_root_directory(_SYSTEM_ROOT)
    create_relational_engine.cache_clear()

    async def _run():
        import cognee.modules.data.models
        import cognee.modules.users.models
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())
    create_relational_engine.cache_clear()
    yield
    cognee.config.system_root_directory(previous_root)
    create_relational_engine.cache_clear()


async def _seed():
    """Tenant X with members a and b; tenant Y with member c; d and e in no tenant."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Tenant, User, UserTenant

    users = {name: uuid4() for name in "abcde"}
    tenant_x, tenant_y = uuid4(), uuid4()
    suffix = uuid4().hex[:8]

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Tenant(id=tenant_x, name=f"X-{suffix}", owner_id=users["a"]))
        session.add(Tenant(id=tenant_y, name=f"Y-{suffix}", owner_id=users["c"]))
        for name, user_id in users.items():
            session.add(
                User(
                    id=user_id,
                    email=f"{name}-{suffix}@example.com",
                    hashed_password="x",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                )
            )
        await session.flush()
        session.add_all(
            [
                UserTenant(user_id=users["a"], tenant_id=tenant_x),
                UserTenant(user_id=users["b"], tenant_id=tenant_x),
                UserTenant(user_id=users["c"], tenant_id=tenant_y),
            ]
        )
        await session.commit()

    emails = {name: f"{name}-{suffix}@example.com" for name in users}
    return users, emails


def _requester(user_id):
    return SimpleNamespace(id=user_id)


@pytest.mark.asyncio
async def test_a_tenant_member_finds_members_of_their_tenant_only():
    users, emails = await _seed()
    a = _requester(users["a"])

    assert await get_user_id_by_email(emails["b"], a) == users["b"]
    assert await get_user_id_by_email(emails["a"], a) == users["a"]
    assert await get_user_id_by_email(emails["c"], a) is None
    assert await get_user_id_by_email(emails["d"], a) is None


@pytest.mark.asyncio
async def test_a_user_in_no_tenant_finds_users_in_no_tenant_only():
    users, emails = await _seed()
    d = _requester(users["d"])

    assert await get_user_id_by_email(emails["e"], d) == users["e"]
    assert await get_user_id_by_email(emails["d"], d) == users["d"]
    assert await get_user_id_by_email(emails["a"], d) is None


@pytest.mark.asyncio
async def test_an_unknown_address_resolves_to_nothing():
    users, _ = await _seed()

    assert await get_user_id_by_email("nobody@example.com", _requester(users["a"])) is None
