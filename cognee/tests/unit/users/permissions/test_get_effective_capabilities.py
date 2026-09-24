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
        # create_database() rather than run_migrations(): the latter is
        # once-per-process and its relational step logs failures instead of
        # raising, so in a run with more than one module it is a silent no-op
        # and this database is never created. create_database() makes the
        # directory and the tables outright. The models import registers the
        # tables.
        import cognee.modules.users.models
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())

    # The engine built above is bound to the event loop asyncio.run() just
    # closed. Drop the cache again so each test gets a fresh one.
    create_relational_engine.cache_clear()


async def _seed(*, tenant_owner_id=None):
    """Create a tenant with one member and one outsider."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Tenant, User, UserTenant

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
        await session.commit()

        return {
            "tenant_id": tenant_id,
            "owner_id": owner_id,
            "member_id": member_id,
            "outsider_id": outsider_id,
        }


async def _add_role_with_member(tenant_id, member_id, name="managers"):
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role, UserRole

    role_id = uuid4()
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Role(id=role_id, name=f"{name}-{role_id}", tenant_id=tenant_id))
        await session.flush()
        session.add(UserRole(user_id=member_id, role_id=role_id))
        await session.commit()
    return role_id


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
async def test_tenant_grant_reaches_a_member():
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()

    await grant_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)

    result = await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    assert MANAGE_USERS in result


@pytest.mark.asyncio
async def test_tenant_grant_does_not_reach_a_non_member():
    """The bug this test exists for: a tenant-level grant answering for outsiders."""
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()

    await grant_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)

    result = await get_effective_capabilities(seed["outsider_id"], seed["tenant_id"])

    assert result == set()


@pytest.mark.asyncio
async def test_role_capability_reaches_its_member():
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()
    role_id = await _add_role_with_member(seed["tenant_id"], seed["member_id"])

    await grant_capability(role_id, seed["tenant_id"], MANAGE_USERS)

    result = await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    assert MANAGE_USERS in result


@pytest.mark.asyncio
async def test_role_in_another_tenant_does_not_leak():
    """A role carries capabilities only inside the tenant that owns it."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import UserTenant
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    home = await _seed()
    other = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # The user is a member of both tenants but only privileged in "home".
        session.add(UserTenant(user_id=home["member_id"], tenant_id=other["tenant_id"]))
        await session.commit()

    role_id = await _add_role_with_member(home["tenant_id"], home["member_id"])
    await grant_capability(role_id, home["tenant_id"], MANAGE_USERS)

    result = await get_effective_capabilities(home["member_id"], other["tenant_id"])

    assert result == set()


@pytest.mark.asyncio
async def test_user_grant_is_scoped_to_its_tenant():
    """The reason the table carries tenant_id: a personal grant stays put.

    The old storage could not represent this at all — UserDefaultPermissions
    had no tenant column, which is why per-user grants used to be refused.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import UserTenant
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    home = await _seed()
    other = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(UserTenant(user_id=home["member_id"], tenant_id=other["tenant_id"]))
        await session.commit()

    await grant_capability(home["member_id"], home["tenant_id"], MANAGE_USERS)

    assert MANAGE_USERS in await get_effective_capabilities(home["member_id"], home["tenant_id"])
    assert await get_effective_capabilities(home["member_id"], other["tenant_id"]) == set()


@pytest.mark.asyncio
async def test_grant_rejects_a_principal_from_another_tenant():
    """A role or a tenant principal can only be granted capabilities inside its
    own tenant. The API always derives that tenant from the principal, so only
    a direct SDK caller passing the wrong tenant_id reaches this."""
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    home = await _seed()
    other = await _seed()
    role_id = await _add_role_with_member(home["tenant_id"], home["member_id"])

    with pytest.raises(PermissionDeniedError):
        await grant_capability(role_id, other["tenant_id"], MANAGE_USERS)
    with pytest.raises(PermissionDeniedError):
        await grant_capability(home["tenant_id"], other["tenant_id"], MANAGE_USERS)


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
    from cognee.modules.users.capabilities.methods import grant_capability, revoke_capability
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()
    role_id = await _add_role_with_member(seed["tenant_id"], seed["member_id"])

    assert await get_effective_capabilities(seed["member_id"], seed["tenant_id"]) == set()

    await grant_capability(role_id, seed["tenant_id"], MANAGE_USERS)
    assert MANAGE_USERS in await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    await revoke_capability(role_id, seed["tenant_id"], MANAGE_USERS)
    assert await get_effective_capabilities(seed["member_id"], seed["tenant_id"]) == set()


@pytest.mark.asyncio
async def test_granting_and_revoking_on_a_tenant_moves_the_member():
    from cognee.modules.users.capabilities.methods import grant_capability, revoke_capability
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()

    await grant_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)
    assert MANAGE_USERS in await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    # Still nothing for someone outside the tenant.
    assert await get_effective_capabilities(seed["outsider_id"], seed["tenant_id"]) == set()

    await revoke_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)
    assert await get_effective_capabilities(seed["member_id"], seed["tenant_id"]) == set()


@pytest.mark.asyncio
async def test_granting_twice_and_revoking_the_absent_are_no_ops():
    """Both directions have to be safe to retry."""
    from cognee.modules.users.capabilities.methods import grant_capability, revoke_capability
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()

    await grant_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)
    await grant_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)

    await revoke_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)
    await revoke_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)


@pytest.mark.asyncio
async def test_names_outside_the_catalog_do_not_resolve():
    """Grants are validated when written, but the catalog is code and can shrink.

    A row whose name is no longer (or never was) in the catalog must stop
    resolving rather than gate nothing under a stale name. Written directly to
    storage because the write path correctly refuses to produce such a row.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import PrincipalCapability
    from cognee.modules.users.permissions.methods import get_effective_capabilities

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(
            PrincipalCapability(
                principal_id=seed["tenant_id"],
                tenant_id=seed["tenant_id"],
                capability="read",
            )
        )
        await session.commit()

    result = await get_effective_capabilities(seed["member_id"], seed["tenant_id"])

    assert result == set()


@pytest.mark.asyncio
async def test_granting_and_revoking_each_need_their_own_capability():
    """End to end through the authorized methods, against real rows.

    A member trusted with MANAGE_USERS must not be able to hand out or take
    away capabilities, or managing members would be a path to every other
    capability in the tenant. Granting and revoking are then separate again.
    """
    from cognee.modules.users.capabilities.methods import (
        authorized_grant_capability,
        authorized_revoke_capability,
        grant_capability,
    )
    from cognee.modules.users.exceptions import CapabilityDeniedError
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
        REVOKE_CAPABILITIES,
    )

    seed = await _seed()
    tenant_id = seed["tenant_id"]
    member_id = seed["member_id"]
    role_id = await _add_role_with_member(tenant_id, member_id)

    await grant_capability(member_id, tenant_id, MANAGE_USERS)

    with pytest.raises(CapabilityDeniedError):
        await authorized_grant_capability(role_id, MANAGE_USERS, member_id)
    with pytest.raises(CapabilityDeniedError):
        await authorized_revoke_capability(member_id, MANAGE_USERS, member_id, tenant_id)

    await authorized_grant_capability(member_id, GRANT_CAPABILITIES, seed["owner_id"], tenant_id)

    await authorized_grant_capability(role_id, MANAGE_USERS, member_id)
    assert MANAGE_USERS in await get_effective_capabilities(member_id, tenant_id)

    with pytest.raises(CapabilityDeniedError):
        await authorized_revoke_capability(role_id, MANAGE_USERS, member_id)

    await authorized_grant_capability(member_id, REVOKE_CAPABILITIES, seed["owner_id"], tenant_id)
    await authorized_revoke_capability(role_id, MANAGE_USERS, member_id)


@pytest.mark.asyncio
async def test_an_outsider_cannot_grant_even_with_the_capability_elsewhere():
    """GRANT_CAPABILITIES held in one tenant says nothing about another."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import (
        authorized_grant_capability,
        grant_capability,
    )
    from cognee.modules.users.exceptions import CapabilityDeniedError
    from cognee.modules.users.models import UserTenant
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
    )

    home = await _seed()
    other = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(UserTenant(user_id=home["member_id"], tenant_id=other["tenant_id"]))
        await session.commit()

    await grant_capability(home["member_id"], home["tenant_id"], GRANT_CAPABILITIES)

    with pytest.raises(CapabilityDeniedError):
        await authorized_grant_capability(other["tenant_id"], MANAGE_USERS, home["member_id"])


async def _granted_by(principal_id, tenant_id):
    """Map each capability the principal holds in the tenant to its granter."""
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import PrincipalCapability

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = await session.execute(
            select(PrincipalCapability.capability, PrincipalCapability.granted_by).where(
                PrincipalCapability.principal_id == principal_id,
                PrincipalCapability.tenant_id == tenant_id,
            )
        )
        return dict(rows.all())


@pytest.mark.asyncio
async def test_a_batch_grant_writes_each_capability_once_and_records_the_granter():
    """Duplicates in a request collapse to one row, and a re-grant keeps the
    original granter: the record answers who gave it first, not who repeated it."""
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
        REVOKE_CAPABILITIES,
    )

    seed = await _seed()
    tenant_id = seed["tenant_id"]

    await grant_capability(
        tenant_id,
        tenant_id,
        [MANAGE_USERS, MANAGE_USERS, GRANT_CAPABILITIES],
        granted_by=seed["owner_id"],
    )
    assert await _granted_by(tenant_id, tenant_id) == {
        MANAGE_USERS: seed["owner_id"],
        GRANT_CAPABILITIES: seed["owner_id"],
    }

    await grant_capability(
        tenant_id,
        tenant_id,
        [GRANT_CAPABILITIES, REVOKE_CAPABILITIES],
        granted_by=seed["member_id"],
    )
    assert await _granted_by(tenant_id, tenant_id) == {
        MANAGE_USERS: seed["owner_id"],
        GRANT_CAPABILITIES: seed["owner_id"],
        REVOKE_CAPABILITIES: seed["member_id"],
    }


@pytest.mark.asyncio
async def test_a_grant_without_a_requester_has_no_granter():
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()

    await grant_capability(seed["tenant_id"], seed["tenant_id"], MANAGE_USERS)

    assert await _granted_by(seed["tenant_id"], seed["tenant_id"]) == {MANAGE_USERS: None}


@pytest.mark.asyncio
async def test_a_batch_revoke_removes_only_the_named_capabilities():
    from cognee.modules.users.capabilities.methods import grant_capability, revoke_capability
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
        REVOKE_CAPABILITIES,
    )

    seed = await _seed()
    tenant_id = seed["tenant_id"]

    await grant_capability(
        tenant_id, tenant_id, [MANAGE_USERS, GRANT_CAPABILITIES, REVOKE_CAPABILITIES]
    )
    await revoke_capability(tenant_id, tenant_id, [MANAGE_USERS, GRANT_CAPABILITIES])

    assert set(await _granted_by(tenant_id, tenant_id)) == {REVOKE_CAPABILITIES}


@pytest.mark.asyncio
async def test_the_authorized_grant_records_the_requester():
    from cognee.modules.users.capabilities.methods import authorized_grant_capability
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
    )

    seed = await _seed()
    tenant_id = seed["tenant_id"]

    await authorized_grant_capability(
        seed["member_id"], [MANAGE_USERS, GRANT_CAPABILITIES], seed["owner_id"], tenant_id
    )

    assert await _granted_by(seed["member_id"], tenant_id) == {
        MANAGE_USERS: seed["owner_id"],
        GRANT_CAPABILITIES: seed["owner_id"],
    }


async def _set_current_tenant(user_id, tenant_id):
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import User

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        user = await session.get(User, user_id)
        user.tenant_id = tenant_id
        await session.commit()


@pytest.mark.asyncio
async def test_concurrent_grants_of_the_same_capability_all_succeed():
    """Two requests granting the same row both used to see it missing, and the
    second insert failed on the primary key, which the API answered with 500."""
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
        REVOKE_CAPABILITIES,
    )

    seed = await _seed()
    tenant_id = seed["tenant_id"]
    member_id = seed["member_id"]

    await asyncio.gather(
        *(grant_capability(member_id, tenant_id, MANAGE_USERS) for _ in range(8)),
        grant_capability(member_id, tenant_id, [MANAGE_USERS, GRANT_CAPABILITIES]),
        grant_capability(member_id, tenant_id, [GRANT_CAPABILITIES, REVOKE_CAPABILITIES]),
    )

    assert set(await _granted_by(member_id, tenant_id)) == {
        MANAGE_USERS,
        GRANT_CAPABILITIES,
        REVOKE_CAPABILITIES,
    }


@pytest.mark.asyncio
async def test_removing_a_member_drops_their_personal_capabilities():
    """Left in place, a personal grant came back when the person was added
    again, so a requester with only MANAGE_USERS could remove and re-add
    someone to restore a capability they could not grant themselves."""
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
    )
    from cognee.modules.users.tenants.methods import add_user_to_tenant, remove_user_from_tenant

    seed = await _seed()
    tenant_id = seed["tenant_id"]
    member_id = seed["member_id"]
    manager_id = seed["outsider_id"]
    await add_user_to_tenant(user_id=manager_id, tenant_id=tenant_id, owner_id=seed["owner_id"])
    await grant_capability(manager_id, tenant_id, MANAGE_USERS)
    await grant_capability(member_id, tenant_id, GRANT_CAPABILITIES)

    await remove_user_from_tenant(user_id=member_id, tenant_id=tenant_id, owner_id=manager_id)
    assert await _granted_by(member_id, tenant_id) == {}

    await add_user_to_tenant(user_id=member_id, tenant_id=tenant_id, owner_id=manager_id)
    assert await get_effective_capabilities(member_id, tenant_id) == set()


@pytest.mark.asyncio
async def test_removing_a_member_keeps_their_grants_in_other_tenants():
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import UserTenant
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS
    from cognee.modules.users.tenants.methods import remove_user_from_tenant

    home = await _seed()
    other = await _seed()
    member_id = home["member_id"]

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(UserTenant(user_id=member_id, tenant_id=other["tenant_id"]))
        await session.commit()

    await grant_capability(member_id, home["tenant_id"], MANAGE_USERS)
    await grant_capability(member_id, other["tenant_id"], MANAGE_USERS)

    await remove_user_from_tenant(
        user_id=member_id, tenant_id=home["tenant_id"], owner_id=home["owner_id"]
    )

    assert await _granted_by(member_id, home["tenant_id"]) == {}
    assert set(await _granted_by(member_id, other["tenant_id"])) == {MANAGE_USERS}


@pytest.mark.asyncio
async def test_deleting_a_role_drops_its_capabilities():
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS
    from cognee.modules.users.roles.methods import delete_role

    seed = await _seed()
    role_id = await _add_role_with_member(seed["tenant_id"], seed["member_id"])
    await grant_capability(role_id, seed["tenant_id"], MANAGE_USERS)

    await delete_role(role_id=role_id, owner_id=seed["owner_id"])

    assert await _granted_by(role_id, seed["tenant_id"]) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["grant", "revoke"])
async def test_a_made_up_tenant_and_a_foreign_one_are_refused_alike(action):
    """Otherwise the 404 for a missing tenant against the 403 for a foreign one
    tells anyone which tenant ids are real."""
    from cognee.modules.users.capabilities.methods import (
        authorized_grant_capability,
        authorized_revoke_capability,
    )
    from cognee.modules.users.exceptions import CapabilityDeniedError
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    authorized = authorized_grant_capability if action == "grant" else authorized_revoke_capability
    seed = await _seed()
    outsider_id = seed["outsider_id"]

    with pytest.raises(CapabilityDeniedError) as foreign:
        await authorized(outsider_id, MANAGE_USERS, outsider_id, seed["tenant_id"])
    with pytest.raises(CapabilityDeniedError) as made_up:
        await authorized(outsider_id, MANAGE_USERS, outsider_id, uuid4())

    assert made_up.value.message == foreign.value.message
    assert made_up.value.status_code == foreign.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("requester", ["member", "outsider"])
async def test_a_real_user_and_a_made_up_id_are_refused_alike_without_a_tenant(requester):
    """A missing tenant_id used to answer 400 for a real user id and 403 for a
    made-up one. It now defaults to the requester's current tenant, and a
    requester with none, or without the capability there, is refused the same
    way either way."""
    from cognee.modules.users.capabilities.methods import authorized_grant_capability
    from cognee.modules.users.exceptions import CapabilityDeniedError
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()
    requester_id = seed[f"{requester}_id"]
    if requester == "member":
        await _set_current_tenant(requester_id, seed["tenant_id"])

    with pytest.raises(CapabilityDeniedError) as real:
        await authorized_grant_capability(seed["owner_id"], MANAGE_USERS, requester_id)
    with pytest.raises(CapabilityDeniedError) as made_up:
        await authorized_grant_capability(uuid4(), MANAGE_USERS, requester_id)

    assert real.value.message == made_up.value.message


@pytest.mark.asyncio
async def test_a_user_principal_defaults_to_the_requesters_current_tenant():
    from cognee.modules.users.capabilities.methods import authorized_grant_capability
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()
    await _set_current_tenant(seed["owner_id"], seed["tenant_id"])

    await authorized_grant_capability(seed["member_id"], MANAGE_USERS, seed["owner_id"])

    assert await _granted_by(seed["member_id"], seed["tenant_id"]) == {
        MANAGE_USERS: seed["owner_id"]
    }


async def _all_capability_rows(tenant_id):
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import PrincipalCapability

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = await session.execute(
            select(PrincipalCapability.principal_id, PrincipalCapability.capability).where(
                PrincipalCapability.tenant_id == tenant_id
            )
        )
        return set(rows.all())


@pytest.mark.asyncio
async def test_a_granter_can_only_pass_on_what_they_hold():
    """GRANT_CAPABILITIES passes on what the granter has, the same rule role
    assignment follows. Without it, holding only GRANT_CAPABILITIES reached the
    whole catalog: grant yourself MANAGE_USERS, or the whole tenant."""
    from cognee.modules.users.capabilities.methods import (
        authorized_grant_capability,
        grant_capability,
    )
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
        REVOKE_CAPABILITIES,
    )

    seed = await _seed()
    tenant_id = seed["tenant_id"]
    granter_id = seed["member_id"]
    await grant_capability(granter_id, tenant_id, GRANT_CAPABILITIES)
    before = await _all_capability_rows(tenant_id)

    for principal_id in (granter_id, tenant_id, seed["owner_id"]):
        with pytest.raises(PermissionDeniedError) as denied:
            await authorized_grant_capability(principal_id, MANAGE_USERS, granter_id, tenant_id)
        assert "manage_users" in denied.value.message

    # All or nothing: one unheld name in a batch writes none of it.
    with pytest.raises(PermissionDeniedError) as denied:
        await authorized_grant_capability(
            seed["owner_id"], [GRANT_CAPABILITIES, REVOKE_CAPABILITIES], granter_id, tenant_id
        )
    assert "revoke_capabilities" in denied.value.message
    assert "grant_capabilities" not in denied.value.message
    assert await _all_capability_rows(tenant_id) == before

    # What they do hold, they can pass on.
    await authorized_grant_capability(seed["owner_id"], GRANT_CAPABILITIES, granter_id, tenant_id)
    assert (seed["owner_id"], GRANT_CAPABILITIES) in await _all_capability_rows(tenant_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("granter", ["owner", "admin role member"])
async def test_the_owner_and_the_admin_role_can_grant_the_whole_catalog(granter):
    """Both pass every capability check, so the hold rule never stops them,
    which is how the first grants in a tenant get made."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import authorized_grant_capability
    from cognee.modules.users.models import Role, UserRole
    from cognee.modules.users.permissions.permission_types import CAPABILITY_TYPES

    seed = await _seed()
    tenant_id = seed["tenant_id"]
    if granter == "owner":
        granter_id = seed["owner_id"]
    else:
        granter_id = seed["member_id"]
        admin_role_id = uuid4()
        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            session.add(Role(id=admin_role_id, name="admin", tenant_id=tenant_id))
            await session.flush()
            session.add(UserRole(user_id=granter_id, role_id=admin_role_id))
            await session.commit()

    await authorized_grant_capability(tenant_id, sorted(CAPABILITY_TYPES), granter_id, tenant_id)

    assert {capability for _, capability in await _all_capability_rows(tenant_id)} == set(
        CAPABILITY_TYPES
    )


@pytest.mark.asyncio
async def test_a_role_whose_tenant_is_gone_is_refused_like_an_unknown_id():
    """SQLite does not enforce roles.tenant_id, so a role can outlive its
    tenant row. Resolution would answer 404 for it; it must read like any
    other refusal."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import authorized_grant_capability
    from cognee.modules.users.exceptions import CapabilityDeniedError
    from cognee.modules.users.models import Role
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()
    orphan_role_id = uuid4()
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Role(id=orphan_role_id, name=f"orphan-{orphan_role_id}", tenant_id=uuid4()))
        await session.commit()

    with pytest.raises(CapabilityDeniedError) as orphan:
        await authorized_grant_capability(orphan_role_id, MANAGE_USERS, seed["owner_id"])
    with pytest.raises(CapabilityDeniedError) as unknown:
        await authorized_grant_capability(uuid4(), MANAGE_USERS, seed["owner_id"])

    assert orphan.value.message == unknown.value.message


@pytest.mark.asyncio
async def test_deleting_a_user_drops_their_capabilities_but_keeps_what_they_granted():
    """The CASCADE on principal_id is not enforced on SQLite, so the rows go
    with the user through the ORM, as ACL rows do. Grants the user made to
    others stay: a grant outlives the person who made it."""
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.methods import delete_user
    from cognee.modules.users.permissions.methods import get_effective_capabilities
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    seed = await _seed()
    tenant_id = seed["tenant_id"]
    leaver_id = seed["member_id"]
    await grant_capability(leaver_id, tenant_id, MANAGE_USERS)
    await grant_capability(seed["outsider_id"], tenant_id, MANAGE_USERS, granted_by=leaver_id)

    await delete_user(f"{leaver_id}@example.com")

    assert await _granted_by(leaver_id, tenant_id) == {}
    assert set(await _granted_by(seed["outsider_id"], tenant_id)) == {MANAGE_USERS}
    assert await get_effective_capabilities(seed["owner_id"], tenant_id)
