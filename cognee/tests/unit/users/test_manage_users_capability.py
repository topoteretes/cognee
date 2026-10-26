"""Creating roles, assigning them and adding people to a tenant need the
MANAGE_USERS capability, not tenant ownership.

These three used to compare the requester against tenant.owner_id, so an admin
who could already list, remove and reassign members still could not add one.
Runs against a real database: the point is that a real grant, resolved the way
production resolves it, opens exactly these doors and no others.
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
            ".cognee_system/test_manage_users_capability",
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
        # once-per-process, so in a run with more than one module it is a
        # no-op and this database is never created. The models import
        # registers the tables.
        import cognee.modules.users.models
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())

    # The engine built above is bound to the event loop asyncio.run() just
    # closed. Drop the cache again so each test gets a fresh one.
    create_relational_engine.cache_clear()


async def _seed():
    """A tenant with its owner, a manager, a plain member, and a person outside it.

    Everyone in the tenant has it as their current tenant, which is the tenant
    create_role acts on. The manager holds MANAGE_USERS personally.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import Role, Tenant, User, UserTenant
    from cognee.modules.users.permissions.permission_types import MANAGE_USERS

    tenant_id = uuid4()
    ids = {name: uuid4() for name in ("owner", "manager", "member", "outsider")}
    role_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        for name, user_id in ids.items():
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
        session.add(Tenant(id=tenant_id, name=f"t-{tenant_id}", owner_id=ids["owner"]))
        await session.flush()

        for name in ("owner", "manager", "member"):
            session.add(UserTenant(user_id=ids[name], tenant_id=tenant_id))
        session.add(Role(id=role_id, name=f"editors-{role_id}", tenant_id=tenant_id))
        await session.flush()

        for name in ("owner", "manager", "member"):
            user = await session.get(User, ids[name])
            user.tenant_id = tenant_id
        await session.commit()

    await grant_capability(ids["manager"], tenant_id, MANAGE_USERS)

    return {"tenant_id": tenant_id, "role_id": role_id, **ids}


async def _count(model, **filters):
    from sqlalchemy import func, select

    from cognee.infrastructure.databases.relational import get_relational_engine

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        query = select(func.count()).select_from(model)
        for column, value in filters.items():
            query = query.where(getattr(model, column) == value)
        return (await session.execute(query)).scalar_one()


@pytest.mark.asyncio
@pytest.mark.parametrize("requester", ["owner", "manager"])
async def test_owner_and_manager_can_create_a_role(requester):
    from cognee.modules.users.models import Role
    from cognee.modules.users.roles.methods import create_role

    seed = await _seed()

    role_id = await create_role(role_name=f"new-{uuid4()}", owner_id=seed[requester])

    assert await _count(Role, id=role_id, tenant_id=seed["tenant_id"]) == 1


@pytest.mark.asyncio
async def test_a_plain_member_cannot_create_a_role():
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.models import Role
    from cognee.modules.users.roles.methods import create_role

    seed = await _seed()

    with pytest.raises(PermissionDeniedError):
        await create_role(role_name="should-not-exist", owner_id=seed["member"])

    assert await _count(Role, tenant_id=seed["tenant_id"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("requester", ["owner", "manager"])
async def test_owner_and_manager_can_assign_a_role(requester):
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.roles.methods import add_user_to_role

    seed = await _seed()

    await add_user_to_role(
        user_id=seed["member"], role_id=seed["role_id"], owner_id=seed[requester]
    )

    assert await _count(UserRole, user_id=seed["member"], role_id=seed["role_id"]) == 1


@pytest.mark.asyncio
async def test_a_plain_member_cannot_assign_a_role():
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.roles.methods import add_user_to_role

    seed = await _seed()

    with pytest.raises(PermissionDeniedError):
        await add_user_to_role(
            user_id=seed["member"], role_id=seed["role_id"], owner_id=seed["member"]
        )

    assert await _count(UserRole, role_id=seed["role_id"]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("requester", ["owner", "manager"])
async def test_owner_and_manager_can_add_someone_to_the_tenant(requester):
    from cognee.modules.users.models import UserTenant
    from cognee.modules.users.tenants.methods import add_user_to_tenant

    seed = await _seed()

    await add_user_to_tenant(
        user_id=seed["outsider"], tenant_id=seed["tenant_id"], owner_id=seed[requester]
    )

    assert await _count(UserTenant, user_id=seed["outsider"], tenant_id=seed["tenant_id"]) == 1


@pytest.mark.asyncio
async def test_a_plain_member_cannot_add_someone_to_the_tenant():
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.models import UserTenant
    from cognee.modules.users.tenants.methods import add_user_to_tenant

    seed = await _seed()

    with pytest.raises(PermissionDeniedError):
        await add_user_to_tenant(
            user_id=seed["outsider"], tenant_id=seed["tenant_id"], owner_id=seed["member"]
        )

    assert await _count(UserTenant, user_id=seed["outsider"]) == 0


@pytest.mark.asyncio
async def test_managing_users_in_one_tenant_does_not_reach_another():
    """The check runs against the tenant being changed, not the requester's own."""
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.models import UserTenant
    from cognee.modules.users.tenants.methods import add_user_to_tenant

    home = await _seed()
    other = await _seed()

    with pytest.raises(PermissionDeniedError):
        await add_user_to_tenant(
            user_id=home["outsider"], tenant_id=other["tenant_id"], owner_id=home["manager"]
        )

    assert await _count(UserTenant, user_id=home["outsider"]) == 0


async def _add_role(tenant_id, name, capabilities=()):
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import Role

    role_id = uuid4()
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Role(id=role_id, name=name, tenant_id=tenant_id))
        await session.commit()

    if capabilities:
        await grant_capability(role_id, tenant_id, list(capabilities))
    return role_id


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["manager", "member"])
async def test_a_manager_cannot_hand_out_a_role_carrying_grant_capabilities(target):
    """Managing users must not be a way to reach granting, for yourself or for
    an accomplice."""
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.permissions.permission_types import GRANT_CAPABILITIES
    from cognee.modules.users.roles.methods import add_user_to_role

    seed = await _seed()
    role_id = await _add_role(seed["tenant_id"], f"granters-{uuid4()}", [GRANT_CAPABILITIES])

    with pytest.raises(PermissionDeniedError) as denied:
        await add_user_to_role(user_id=seed[target], role_id=role_id, owner_id=seed["manager"])

    assert GRANT_CAPABILITIES in denied.value.message
    assert await _count(UserRole, role_id=role_id) == 0


@pytest.mark.asyncio
async def test_a_manager_cannot_create_an_admin_role_and_join_it():
    """The deprecated fallback treats any role named "admin" as holding every
    capability, so a manager who could create one and join it would own the
    tenant in two calls."""
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.roles.methods import add_user_to_role, create_role

    seed = await _seed()
    admin_role_id = await create_role(role_name="admin", owner_id=seed["manager"])

    with pytest.raises(PermissionDeniedError):
        await add_user_to_role(
            user_id=seed["manager"], role_id=admin_role_id, owner_id=seed["manager"]
        )

    assert await _count(UserRole, role_id=admin_role_id) == 0


@pytest.mark.asyncio
async def test_holding_what_the_role_carries_is_enough():
    """The rule caps what can be handed out; it does not reserve roles for the owner."""
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.permissions.permission_types import (
        GRANT_CAPABILITIES,
        MANAGE_USERS,
    )
    from cognee.modules.users.roles.methods import add_user_to_role

    seed = await _seed()
    managers_role = await _add_role(seed["tenant_id"], f"managers-{uuid4()}", [MANAGE_USERS])
    granters_role = await _add_role(seed["tenant_id"], f"granters-{uuid4()}", [GRANT_CAPABILITIES])

    # MANAGE_USERS is already the manager's, so passing it on is fine.
    await add_user_to_role(user_id=seed["member"], role_id=managers_role, owner_id=seed["manager"])

    await grant_capability(seed["manager"], seed["tenant_id"], GRANT_CAPABILITIES)
    await add_user_to_role(user_id=seed["member"], role_id=granters_role, owner_id=seed["manager"])

    assert await _count(UserRole, user_id=seed["member"]) == 2


@pytest.mark.asyncio
async def test_a_capability_granted_to_the_tenant_counts_as_held():
    from cognee.modules.users.capabilities.methods import grant_capability
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.permissions.permission_types import GRANT_CAPABILITIES
    from cognee.modules.users.roles.methods import add_user_to_role

    seed = await _seed()
    role_id = await _add_role(seed["tenant_id"], f"granters-{uuid4()}", [GRANT_CAPABILITIES])
    await grant_capability(seed["tenant_id"], seed["tenant_id"], GRANT_CAPABILITIES)

    await add_user_to_role(user_id=seed["member"], role_id=role_id, owner_id=seed["manager"])

    assert await _count(UserRole, role_id=role_id) == 1


@pytest.mark.asyncio
async def test_the_owner_and_an_existing_admin_can_still_fill_the_admin_role():
    """Both pass every capability check, so the rule never stops them."""
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.roles.methods import add_user_to_role

    seed = await _seed()
    admin_role_id = await _add_role(seed["tenant_id"], "admin")

    await add_user_to_role(user_id=seed["manager"], role_id=admin_role_id, owner_id=seed["owner"])
    await add_user_to_role(user_id=seed["member"], role_id=admin_role_id, owner_id=seed["manager"])

    assert await _count(UserRole, role_id=admin_role_id) == 2


@pytest.mark.asyncio
async def test_a_role_deleted_during_assignment_leaves_no_membership(monkeypatch):
    """add_user_to_role checks permission between two sessions. A role deleted
    in that window used to get a membership row anyway on SQLite, which does
    not enforce the foreign key, and the call reported success."""
    import importlib

    from cognee.modules.users.exceptions import RoleNotFoundError
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.roles.methods import add_user_to_role, delete_role

    add_user_to_role_module = importlib.import_module(
        "cognee.modules.users.roles.methods.add_user_to_role"
    )
    real_check = add_user_to_role_module.require_role_capabilities
    seed = await _seed()

    async def check_then_delete_the_role(requester_id, role_id, tenant_id, role_name):
        await real_check(requester_id, role_id, tenant_id, role_name)
        await delete_role(role_id=role_id, owner_id=seed["owner"])

    monkeypatch.setattr(
        add_user_to_role_module, "require_role_capabilities", check_then_delete_the_role
    )

    with pytest.raises(RoleNotFoundError):
        await add_user_to_role(
            user_id=seed["member"], role_id=seed["role_id"], owner_id=seed["owner"]
        )

    assert await _count(UserRole, user_id=seed["member"]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("leaves_by", ["removal from the tenant", "account deletion"])
async def test_a_user_who_leaves_during_assignment_gets_no_membership(monkeypatch, leaves_by):
    """The INSERT ... SELECT re-checks the tenant membership, not only the role.
    Otherwise a removal racing an assignment left a role membership for a
    non-member, and re-adding them later brought the role's capabilities back."""
    import importlib

    from cognee.modules.users.exceptions import TenantNotFoundError, UserNotFoundError
    from cognee.modules.users.methods import delete_user
    from cognee.modules.users.models import UserRole
    from cognee.modules.users.roles.methods import add_user_to_role
    from cognee.modules.users.tenants.methods import remove_user_from_tenant

    add_user_to_role_module = importlib.import_module(
        "cognee.modules.users.roles.methods.add_user_to_role"
    )
    real_check = add_user_to_role_module.require_role_capabilities
    seed = await _seed()

    async def check_then_the_user_leaves(requester_id, role_id, tenant_id, role_name):
        await real_check(requester_id, role_id, tenant_id, role_name)
        if leaves_by == "account deletion":
            await delete_user(f"{seed['member']}@example.com")
        else:
            await remove_user_from_tenant(
                user_id=seed["member"], tenant_id=seed["tenant_id"], owner_id=seed["owner"]
            )

    monkeypatch.setattr(
        add_user_to_role_module, "require_role_capabilities", check_then_the_user_leaves
    )

    expected = UserNotFoundError if leaves_by == "account deletion" else TenantNotFoundError
    with pytest.raises(expected):
        await add_user_to_role(
            user_id=seed["member"], role_id=seed["role_id"], owner_id=seed["owner"]
        )

    assert await _count(UserRole, user_id=seed["member"]) == 0
