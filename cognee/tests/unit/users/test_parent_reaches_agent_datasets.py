"""A user reaches the datasets their agents reach, by id (SDK-925).

The user holds their agents' keys, so this grants nothing new; it makes the
access usable without the key. Runs against a real database with agents created
the way production creates them. Each agent's dataset is granted to the agent
only, never to the parent, so a pass proves the user's agents were consulted rather
than the creation-time grant create_authorized_dataset hands the direct parent.
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
            ".cognee_system/test_parent_reaches_agent_datasets",
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
    create_relational_engine.cache_clear()

    async def _run():
        import cognee.modules.users.models
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())
    create_relational_engine.cache_clear()


async def _user(tenant_id=None):
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.methods import get_user
    from cognee.modules.users.models import Tenant, User, UserTenant

    user_id = uuid4()
    async with get_relational_engine().get_async_session() as session:
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
        await session.flush()
        if tenant_id is not None:
            if await session.get(Tenant, tenant_id) is None:
                session.add(Tenant(id=tenant_id, name=f"t-{tenant_id}", owner_id=user_id))
                await session.flush()
            session.add(UserTenant(user_id=user_id, tenant_id=tenant_id))
            (await session.get(User, user_id)).tenant_id = tenant_id
        await session.commit()
    return await get_user(user_id)


async def _agent(parent):
    from cognee.modules.agents.create_agent import create_agent
    from cognee.modules.users.methods import get_user

    agent, _ = await create_agent(f"agent-{uuid4()}", parent)
    return await get_user(agent.id)


async def _dataset_of(agent, name, tenant_id=None):
    """A dataset the agent owns with every permission, granted to the agent alone."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models.Dataset import Dataset
    from cognee.modules.users.permissions.methods import give_permission_on_dataset

    dataset_id = uuid4()
    async with get_relational_engine().get_async_session() as session:
        session.add(Dataset(id=dataset_id, name=name, owner_id=agent.id, tenant_id=tenant_id))
        await session.commit()
    for permission in ("read", "write", "delete", "share"):
        await give_permission_on_dataset(agent, dataset_id, permission)
    return dataset_id


async def _readable_ids(user):
    from cognee.modules.users.permissions.methods import get_all_user_permission_datasets

    return {dataset.id for dataset in await get_all_user_permission_datasets(user, "read")}


@pytest.mark.asyncio
async def test_parent_reaches_its_agents_datasets():
    from cognee.modules.users.permissions.methods import get_specific_user_permission_datasets

    parent = await _user()
    agent = await _agent(parent)
    agent_dataset = await _dataset_of(agent, f"a-{uuid4()}")

    assert agent_dataset in await _readable_ids(parent)
    for permission in ("read", "write", "delete", "share"):
        assert await get_specific_user_permission_datasets(parent.id, permission, [agent_dataset])


@pytest.mark.asyncio
async def test_access_does_not_flow_to_other_users_or_up_from_agent_to_parent():
    parent = await _user()
    agent = await _agent(parent)
    stranger = await _user()
    parent_only = await _dataset_of(parent, f"p-{uuid4()}")
    agent_dataset = await _dataset_of(agent, f"a-{uuid4()}")

    assert agent_dataset not in await _readable_ids(stranger)
    # Only downwards: the agent does not gain what its parent holds.
    assert parent_only not in await _readable_ids(agent)


@pytest.mark.asyncio
async def test_parent_does_not_reach_an_agent_dataset_in_another_tenant():
    tenant_id = uuid4()
    parent = await _user(tenant_id)
    agent = await _agent(parent)
    elsewhere = await _dataset_of(agent, f"e-{uuid4()}", tenant_id=uuid4())

    assert elsewhere not in await _readable_ids(parent)


@pytest.mark.asyncio
async def test_parent_reaches_an_agent_dataset_by_id_not_by_name():
    """Names resolve among the user's own datasets only, as for any dataset another
    user owns: the agent's dataset is reached by passing its id."""
    from cognee.modules.data.exceptions import DatasetNotFoundError
    from cognee.modules.data.methods import get_authorized_existing_datasets

    parent = await _user()
    agent = await _agent(parent)
    name = f"x-{uuid4()}"
    agent_dataset = await _dataset_of(agent, name)

    by_id = await get_authorized_existing_datasets([agent_dataset], "read", parent)
    assert [dataset.id for dataset in by_id] == [agent_dataset]
    with pytest.raises(DatasetNotFoundError):
        await get_authorized_existing_datasets([name], "read", parent, strict=True)
