"""Capability resolution runs against a real database.

The rest of this package monkeypatches its collaborators, which cannot cover
what matters here: every level of the resolution has to be scoped to the tenant
being asked about, and to actual membership in it. A missing WHERE clause is
invisible to a mocked session but hands user management to anyone who knows a
tenant id, since has_user_management_permission guards listing and removing
users, roles and role membership.
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
            ".cognee_system/test_effective_capabilities",
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


async def _seed(*, tenant_owner_id=None):
    """Create a tenant with one member, one outsider and a "manage_users" permission."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Tenant, User, UserTenant
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    owner_id = tenant_owner_id or uuid4()
    member_id = uuid4()
    outsider_id = uuid4()
    tenant_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        for user_id in (owner_id, member_id, outsider_id):
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
        session.add(Tenant(id=tenant_id, name=f"t-{tenant_id}", owner_id=owner_id))
        await session.flush()

        session.add(UserTenant(user_id=owner_id, tenant_id=tenant_id))
        session.add(UserTenant(user_id=member_id, tenant_id=tenant_id))

        # Permission.name is unique, so the row is shared across tests in this module.
        permission_id = await _permission_id(session, MANAGE_USERS)
        await session.commit()

        return {
            "tenant_id": tenant_id,
            "owner_id": owner_id,
            "member_id": member_id,
            "outsider_id": outsider_id,
            "permission_id": permission_id,
        }


async def _permission_id(session, name: str):
    """Return the id of the named permission, creating the row the first time."""
    from sqlalchemy import select

    from cognee.modules.users.models import Permission

    existing = (
        (await session.execute(select(Permission).where(Permission.name == name))).scalars().first()
    )
    if existing is not None:
        return existing.id

    permission = Permission(id=uuid4(), name=name)
    session.add(permission)
    await session.flush()
    return permission.id


@pytest.mark.asyncio
async def test_owner_holds_the_whole_catalog():
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import CAPABILITY_TYPES

    seed = await _seed()

    result = await get_effective_capabilities(seed["owner_id"], seed["tenant_id"])

    assert result == set(CAPABILITY_TYPES)


@pytest.mark.asyncio
async def test_owner_short_circuits_before_the_membership_gate():
    """Ownership alone is enough, with no row in user_tenants.

    The order matters: if the membership gate ran first, an owner who is not
    listed as a member would be locked out of their own tenant.
    """
    from sqlalchemy import delete

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import UserTenant
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import CAPABILITY_TYPES

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        await session.execute(
            delete(UserTenant).where(
                UserTenant.user_id == seed["owner_id"],
                UserTenant.tenant_id == seed["tenant_id"],
            )
        )
        await session.commit()

    result = await get_effective_capabilities(seed["owner_id"], seed["tenant_id"])

    assert result == set(CAPABILITY_TYPES)


@pytest.mark.asyncio
async def test_tenant_default_reaches_a_member():
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import TenantDefaultPermissions
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(
            TenantDefaultPermissions(
                tenant_id=seed["tenant_id"], permission_id=seed["permission_id"]
            )
        )
        await session.commit()

    result = await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    assert MANAGE_USERS in result


@pytest.mark.asyncio
async def test_tenant_default_does_not_reach_a_non_member():
    """The bug this test exists for: a tenant-level grant answering for outsiders."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import TenantDefaultPermissions
    from cognee.modules.users.permissions.methods import get_effective_capabilities

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(
            TenantDefaultPermissions(
                tenant_id=seed["tenant_id"], permission_id=seed["permission_id"]
            )
        )
        await session.commit()

    result = await get_effective_capabilities(seed["outsider_id"], seed["tenant_id"])

    assert result == set()


@pytest.mark.asyncio
async def test_role_capability_reaches_its_member():
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role, RoleDefaultPermissions, UserRole
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()
    role_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Role(id=role_id, name="managers", tenant_id=seed["tenant_id"]))
        await session.flush()
        session.add(UserRole(user_id=seed["member_id"], role_id=role_id))
        session.add(RoleDefaultPermissions(role_id=role_id, permission_id=seed["permission_id"]))
        await session.commit()

    result = await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    assert MANAGE_USERS in result


@pytest.mark.asyncio
async def test_role_in_another_tenant_does_not_leak():
    """A role carries capabilities only inside the tenant that owns it."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role, RoleDefaultPermissions, UserRole, UserTenant
    from cognee.modules.users.permissions.methods import get_effective_capabilities

    home = await _seed()
    other = await _seed()
    role_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # The user is a member of both tenants but only privileged in "home".
        session.add(UserTenant(user_id=home["member_id"], tenant_id=other["tenant_id"]))
        session.add(Role(id=role_id, name="managers", tenant_id=home["tenant_id"]))
        await session.flush()
        session.add(UserRole(user_id=home["member_id"], role_id=role_id))
        session.add(RoleDefaultPermissions(role_id=role_id, permission_id=home["permission_id"]))
        await session.commit()

    result = await get_effective_capabilities(home["member_id"], other["tenant_id"])

    assert result == set()


@pytest.mark.asyncio
async def test_membership_is_required_and_leaks_nothing():
    """A tenant you are not in and one that does not exist must be indistinguishable.

    Otherwise any authenticated caller can tell real tenant ids from invented
    ones by comparing the responses.
    """
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.permissions.methods import require_tenant_membership

    seed = await _seed()

    assert await require_tenant_membership(seed["owner_id"], seed["tenant_id"]) is True
    assert await require_tenant_membership(seed["member_id"], seed["tenant_id"]) is True

    with pytest.raises(PermissionDeniedError) as not_a_member:
        await require_tenant_membership(seed["outsider_id"], seed["tenant_id"])

    with pytest.raises(PermissionDeniedError) as no_such_tenant:
        await require_tenant_membership(seed["member_id"], uuid4())

    assert str(not_a_member.value) == str(no_such_tenant.value)


@pytest.mark.asyncio
async def test_granting_and_revoking_on_a_role_moves_the_member():
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role, UserRole
    from cognee.modules.users.permissions.methods import (
        get_effective_capabilities,
        give_default_permission_to_role,
        revoke_default_permission_from_role,
    )
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()
    role_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Role(id=role_id, name=f"r-{role_id}", tenant_id=seed["tenant_id"]))
        await session.flush()
        session.add(UserRole(user_id=seed["member_id"], role_id=role_id))
        await session.commit()

    assert await get_effective_capabilities(seed["member_id"], seed["tenant_id"]) == set()

    await give_default_permission_to_role(role_id, MANAGE_USERS)
    assert MANAGE_USERS in await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    await revoke_default_permission_from_role(role_id, MANAGE_USERS)
    assert await get_effective_capabilities(seed["member_id"], seed["tenant_id"]) == set()


@pytest.mark.asyncio
async def test_granting_and_revoking_on_a_tenant_moves_the_member():
    from cognee.modules.users.permissions.methods import (
        get_effective_capabilities,
        give_default_permission_to_tenant,
        revoke_default_permission_from_tenant,
    )
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()

    await give_default_permission_to_tenant(seed["tenant_id"], MANAGE_USERS)
    assert MANAGE_USERS in await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    # Still nothing for someone outside the tenant.
    assert await get_effective_capabilities(seed["outsider_id"], seed["tenant_id"]) == set()

    await revoke_default_permission_from_tenant(seed["tenant_id"], MANAGE_USERS)
    assert await get_effective_capabilities(seed["member_id"], seed["tenant_id"]) == set()


@pytest.mark.asyncio
async def test_revoking_what_was_never_granted_is_a_no_op():
    """Revoke has to be safe to retry, so an absent grant is not an error."""
    from cognee.modules.users.permissions.methods import (
        revoke_default_permission_from_tenant,
    )
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()

    await revoke_default_permission_from_tenant(seed["tenant_id"], MANAGE_USERS)
    await revoke_default_permission_from_tenant(seed["tenant_id"], MANAGE_USERS)


@pytest.mark.asyncio
async def test_dataset_permissions_are_not_capabilities():
    """read/write/delete/share live in the same table but are not capabilities."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import TenantDefaultPermissions
    from cognee.modules.users.permissions.methods import get_effective_capabilities

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        read_permission_id = await _permission_id(session, "read")
        session.add(
            TenantDefaultPermissions(tenant_id=seed["tenant_id"], permission_id=read_permission_id)
        )
        await session.commit()

    result = await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    assert result == set()
