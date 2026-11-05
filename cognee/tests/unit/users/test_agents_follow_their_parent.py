"""An agent's access ends when its parent's does (SDK-925).

An agent is its own user with its own API keys and ACLs, and nothing that ended
the parent's access reached it: a member removed from a tenant kept reading and
writing through the agents they had created, and a deleted user left agents with
live keys and no parent. Runs against a real database, with agents created the
way production creates them, because the point is that a real key stops
resolving and a real ACL is gone.
"""

import asyncio
import os
import pathlib
from uuid import uuid4

import pytest

import cognee

_SYSTEM_ROOT = str(
    pathlib.Path(
        os.path.join(
            pathlib.Path(__file__).parent.parent.parent,
            ".cognee_system/test_agents_follow_their_parent",
        )
    ).resolve()
)


@pytest.fixture(autouse=True, scope="module")
def _isolated_db():
    """Point cognee at a database of its own so these rows never touch dev data."""
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    cognee.config.system_root_directory(_SYSTEM_ROOT)

    # The engine is process-global (@lru_cache), so another module in the same
    # run leaves one cached against its own system root.
    create_relational_engine.cache_clear()

    async def _run():
        import cognee.modules.users.models
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())

    # The engine built above is bound to the event loop asyncio.run() just
    # closed. Drop the cache again so each test gets a fresh one.
    create_relational_engine.cache_clear()


async def _seed():
    """A tenant with its owner and a member, and a dataset the tenant holds.

    Both people have the tenant as their current tenant, which is the tenant
    create_agent copies onto an agent. A second tenant with a dataset of its own
    stands for everything an agent may legitimately keep.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models.Dataset import Dataset
    from cognee.modules.users.models import Tenant, User, UserTenant

    tenant_id = uuid4()
    other_tenant_id = uuid4()
    dataset_id = uuid4()
    other_dataset_id = uuid4()
    owner_id = uuid4()
    member_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        for user_id in (owner_id, member_id):
            session.add(
                User(
                    id=user_id,
                    email=f"{user_id}@example.com",
                    hashed_password="x",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                )
            )
        for id_ in (tenant_id, other_tenant_id):
            session.add(Tenant(id=id_, name=f"t-{id_}", owner_id=owner_id))
        await session.flush()

        session.add(Dataset(id=dataset_id, name="shared", owner_id=owner_id, tenant_id=tenant_id))
        session.add(
            Dataset(
                id=other_dataset_id, name="elsewhere", owner_id=owner_id, tenant_id=other_tenant_id
            )
        )
        for user_id in (owner_id, member_id):
            session.add(UserTenant(user_id=user_id, tenant_id=tenant_id))
            (await session.get(User, user_id)).tenant_id = tenant_id
        await session.commit()

    return {
        "tenant_id": tenant_id,
        "other_tenant_id": other_tenant_id,
        "dataset_id": dataset_id,
        "other_dataset_id": other_dataset_id,
        "owner_id": owner_id,
        "member_id": member_id,
    }


async def _agent_with_access(name, parent_id, dataset_id):
    """An agent of ``parent_id`` holding a key and read access to the dataset."""
    from cognee.modules.agents.create_agent import create_agent
    from cognee.modules.users.methods import get_user
    from cognee.modules.users.permissions.methods import give_permission_on_dataset

    parent = await get_user(parent_id)
    if parent.parent_user_id is None:
        agent, key = await create_agent(f"{name}-{uuid4()}", parent)
    else:
        agent, key = await _agent_of_an_agent(f"{name}-{uuid4()}", parent)
    await give_permission_on_dataset(agent, dataset_id, "read")
    return agent.id, key


async def _agent_of_an_agent(name, parent):
    """An agent whose parent is an agent, as one created before create_agent refused
    agents as parents still exists: create_agent's own steps, without that check."""
    from sqlalchemy import update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.api_key.create_api_key import create_api_key
    from cognee.modules.users.methods import create_user
    from cognee.modules.users.models import User

    agent = await create_user(
        email=f"{name}+{parent.id}@cognee.agent", password="!", parent_user_id=parent.id
    )
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            update(User).where(User.id == agent.id).values(tenant_id=parent.tenant_id)
        )
        await session.commit()
    return agent, (await create_api_key(agent, name)).api_key


async def _resolve(key):
    """The user a key authenticates as, or None, resolved the way a request resolves it."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.get_user_db import get_user_db_context
    from cognee.modules.users.get_user_manager import get_user_manager_context

    async with (
        get_relational_engine().get_async_session() as session,
        get_user_db_context(session) as user_db,
        get_user_manager_context(user_db) as user_manager,
    ):
        return await user_manager.get_by_token(key)


async def _count(model, **filters):
    from sqlalchemy import func, select

    from cognee.infrastructure.databases.relational import get_relational_engine

    async with get_relational_engine().get_async_session() as session:
        query = select(func.count()).select_from(model)
        for column, value in filters.items():
            query = query.where(getattr(model, column) == value)
        return (await session.execute(query)).scalar_one()


@pytest.mark.asyncio
async def test_removing_a_member_revokes_the_keys_and_access_of_their_agents():
    from cognee.modules.users.models import ACL
    from cognee.modules.users.tenants.methods import remove_user_from_tenant

    seed = await _seed()
    agent_id, agent_key = await _agent_with_access("agent", seed["member_id"], seed["dataset_id"])
    sub_agent_id, sub_agent_key = await _agent_with_access(
        "sub-agent", agent_id, seed["dataset_id"]
    )
    assert await _resolve(agent_key) is not None
    assert await _resolve(sub_agent_key) is not None

    await remove_user_from_tenant(
        user_id=seed["member_id"], tenant_id=seed["tenant_id"], owner_id=seed["owner_id"]
    )

    assert await _resolve(agent_key) is None
    assert await _resolve(sub_agent_key) is None
    assert await _count(ACL, principal_id=agent_id) == 0
    assert await _count(ACL, principal_id=sub_agent_id) == 0


@pytest.mark.asyncio
async def test_removing_a_member_clears_the_tenant_they_and_their_agents_were_working_in():
    """A stale current tenant lets a new key or a same-named dataset put them back in it."""
    from cognee.modules.users.models import User

    seed = await _seed()
    agent_id, _ = await _agent_with_access("agent", seed["member_id"], seed["dataset_id"])

    from cognee.modules.users.tenants.methods import remove_user_from_tenant

    await remove_user_from_tenant(
        user_id=seed["member_id"], tenant_id=seed["tenant_id"], owner_id=seed["owner_id"]
    )

    assert await _count(User, id=seed["member_id"], tenant_id=seed["tenant_id"]) == 0
    assert await _count(User, id=agent_id, tenant_id=seed["tenant_id"]) == 0
    assert await _count(User, id=seed["owner_id"], tenant_id=seed["tenant_id"]) == 1


@pytest.mark.asyncio
async def test_removing_a_member_removes_the_agents_membership_and_roles_in_the_tenant():
    """An agent an owner added to the tenant reaches its data through role and capabilities."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import PrincipalCapability, Role, UserRole, UserTenant
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS
    from cognee.modules.users.tenants.methods import remove_user_from_tenant

    seed = await _seed()
    agent_id, _ = await _agent_with_access("agent", seed["member_id"], seed["dataset_id"])
    role_id = uuid4()
    async with get_relational_engine().get_async_session() as session:
        session.add(Role(id=role_id, name=f"editors-{role_id}", tenant_id=seed["tenant_id"]))
        await session.flush()
        session.add(UserTenant(user_id=agent_id, tenant_id=seed["tenant_id"]))
        session.add(UserRole(user_id=agent_id, role_id=role_id))
        await session.commit()
    await grant_capability(agent_id, seed["tenant_id"], MANAGE_USERS)

    await remove_user_from_tenant(
        user_id=seed["member_id"], tenant_id=seed["tenant_id"], owner_id=seed["owner_id"]
    )

    assert await _count(UserTenant, user_id=agent_id) == 0
    assert await _count(UserRole, user_id=agent_id) == 0
    assert await _count(PrincipalCapability, principal_id=agent_id) == 0


@pytest.mark.asyncio
async def test_removing_a_member_leaves_the_agents_of_other_people_alone():
    from cognee.modules.users.models import ACL
    from cognee.modules.users.tenants.methods import remove_user_from_tenant

    seed = await _seed()
    owners_agent_id, owners_agent_key = await _agent_with_access(
        "owners-agent", seed["owner_id"], seed["dataset_id"]
    )

    await remove_user_from_tenant(
        user_id=seed["member_id"], tenant_id=seed["tenant_id"], owner_id=seed["owner_id"]
    )

    assert await _resolve(owners_agent_key) is not None
    assert await _count(ACL, principal_id=owners_agent_id) == 1


@pytest.mark.asyncio
async def test_an_agent_working_in_another_tenant_keeps_its_key_and_its_access_there():
    """Removal from one tenant must not switch off an agent that belongs to a different one."""
    from sqlalchemy import update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import (
        ACL,
        PrincipalCapability,
        Role,
        User,
        UserRole,
        UserTenant,
    )
    from cognee.modules.users.permissions.methods import give_permission_on_dataset
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS
    from cognee.modules.users.tenants.methods import remove_user_from_tenant

    seed = await _seed()
    agent_id, agent_key = await _agent_with_access("agent", seed["member_id"], seed["dataset_id"])
    role_id = uuid4()
    async with get_relational_engine().get_async_session() as session:
        await session.execute(
            update(User).where(User.id == agent_id).values(tenant_id=seed["other_tenant_id"])
        )
        session.add(Role(id=role_id, name=f"editors-{role_id}", tenant_id=seed["other_tenant_id"]))
        await session.flush()
        session.add(UserTenant(user_id=agent_id, tenant_id=seed["other_tenant_id"]))
        session.add(UserRole(user_id=agent_id, role_id=role_id))
        await session.commit()
        agent = await session.get(User, agent_id)
        await give_permission_on_dataset(agent, seed["other_dataset_id"], "read")
    await grant_capability(agent_id, seed["other_tenant_id"], MANAGE_USERS)

    await remove_user_from_tenant(
        user_id=seed["member_id"], tenant_id=seed["tenant_id"], owner_id=seed["owner_id"]
    )

    assert await _resolve(agent_key) is not None
    assert await _count(User, id=agent_id, tenant_id=seed["other_tenant_id"]) == 1
    assert await _count(ACL, principal_id=agent_id, dataset_id=seed["dataset_id"]) == 0
    assert await _count(ACL, principal_id=agent_id, dataset_id=seed["other_dataset_id"]) == 1
    assert await _count(UserTenant, user_id=agent_id, tenant_id=seed["other_tenant_id"]) == 1
    assert await _count(UserRole, user_id=agent_id, role_id=role_id) == 1
    assert await _count(PrincipalCapability, principal_id=agent_id) == 1


@pytest.mark.asyncio
async def test_deleting_a_user_deletes_their_agents_and_the_agents_of_those():
    from cognee.modules.users.methods import delete_user, get_user
    from cognee.modules.users.models import User
    from cognee.modules.users.models.UserApiKey import UserApiKey

    seed = await _seed()
    agent_id, agent_key = await _agent_with_access("agent", seed["member_id"], seed["dataset_id"])
    sub_agent_id, sub_agent_key = await _agent_with_access(
        "sub-agent", agent_id, seed["dataset_id"]
    )

    await delete_user((await get_user(seed["member_id"])).email)

    assert await _resolve(agent_key) is None
    assert await _resolve(sub_agent_key) is None
    assert await _count(User, id=agent_id) == 0
    assert await _count(User, id=sub_agent_id) == 0
    assert await _count(UserApiKey, user_id=agent_id) == 0
    assert await _count(UserApiKey, user_id=sub_agent_id) == 0
    assert await _count(User, id=seed["owner_id"]) == 1


@pytest.mark.asyncio
async def test_deleting_a_user_holds_one_connection_however_deep_the_agents_go():
    """Each nested session held a pooled connection while waiting for the next (#4197)."""
    from sqlalchemy import event

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.methods import delete_user, get_user

    seed = await _seed()
    agent_id, _ = await _agent_with_access("agent", seed["member_id"], seed["dataset_id"])
    sub_agent_id, _ = await _agent_with_access("sub-agent", agent_id, seed["dataset_id"])
    await _agent_with_access("sub-sub-agent", sub_agent_id, seed["dataset_id"])
    email = (await get_user(seed["member_id"])).email

    held = {"now": 0, "peak": 0}
    pool = get_relational_engine().engine.sync_engine

    def _checked_out(*_):
        held["now"] += 1
        held["peak"] = max(held["peak"], held["now"])

    def _checked_in(*_):
        held["now"] -= 1

    event.listen(pool, "checkout", _checked_out)
    event.listen(pool, "checkin", _checked_in)
    try:
        await delete_user(email)
    finally:
        event.remove(pool, "checkout", _checked_out)
        event.remove(pool, "checkin", _checked_in)

    assert held["peak"] == 1
