"""Looking a role up by id runs against a real database.

The capability endpoints take a role id straight from the URL and have to
establish which tenant owns it before they can authorize anything. That lookup
used to call get_role, which takes (tenant_id, role_name), so both endpoints
raised TypeError before reaching their permission check. A signature mismatch
is invisible to a mocked collaborator, hence the real session here.
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
            pathlib.Path(__file__).parent.parent.parent.parent,
            ".cognee_system/test_get_role_by_id",
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
        from cognee.infrastructure.databases.relational import get_relational_engine

        # create_database() rather than run_migrations(): the latter is
        # once-per-process and its relational step logs failures instead of
        # raising, so in a run with more than one module it is a silent no-op
        # and this database is never created. create_database() makes the
        # directory and the tables outright.
        import cognee.modules.users.models  # noqa: F401  register the tables

        await get_relational_engine().create_database()

    asyncio.run(_run())

    # The engine built above is bound to the event loop asyncio.run() just
    # closed. Drop the cache again so each test gets a fresh one.
    create_relational_engine.cache_clear()


async def _seed_role():
    """Create a tenant with one role in it."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role, Tenant

    tenant_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Tenant(id=tenant_id, name=f"t-{tenant_id}", owner_id=uuid4()))
        await session.flush()

        role = Role(id=uuid4(), name="reviewers", tenant_id=tenant_id)
        session.add(role)
        await session.commit()

        return {"tenant_id": tenant_id, "role_id": role.id}


@pytest.mark.asyncio
async def test_returns_the_role_and_the_tenant_that_owns_it():
    """The tenant_id is the point: authorization is checked against it."""
    from cognee.modules.users.permissions.methods import get_role_by_id

    seed = await _seed_role()

    role = await get_role_by_id(seed["role_id"])

    assert role.id == seed["role_id"]
    assert role.tenant_id == seed["tenant_id"]
    assert role.name == "reviewers"


@pytest.mark.asyncio
async def test_unknown_id_raises_role_not_found():
    from cognee.modules.users.exceptions import RoleNotFoundError
    from cognee.modules.users.permissions.methods import get_role_by_id

    await _seed_role()

    with pytest.raises(RoleNotFoundError):
        await get_role_by_id(uuid4())
