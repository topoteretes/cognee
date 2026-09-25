"""_read_roles_and_grants runs against a real database.

Mocking cannot cover the failure this exists to catch: CLO-399's own
dataset-groups endpoint first read Principal.id/Principal.type through an
outer join across User/Role/Tenant and got a user labelled "tenant" back,
because the ORM tried to resolve one polymorphic identity across three
subclass tables at once. That only shows up against a real schema.
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
            ".cognee_system/test_memory_provenance_role_grants",
        )
    ).resolve()
)


@pytest.fixture(autouse=True, scope="module")
def _isolated_db():
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    cognee.config.system_root_directory(_SYSTEM_ROOT)
    create_relational_engine.cache_clear()

    async def _run():
        import cognee.modules.data.models
        import cognee.modules.users.models
        from cognee.infrastructure.databases.relational import get_relational_engine

        await get_relational_engine().create_database()

    asyncio.run(_run())
    create_relational_engine.cache_clear()


async def _permission_id(session, name: str):
    """Return the id of the named permission, creating the row the first time.

    Permission.name is unique, and this module's DB is shared across every
    test in the file (module-scoped fixture), so a plain insert on the second
    call collides with the first.
    """
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


async def _seed():
    """One tenant, two users, one role with one member, one dataset, and a
    grant on each of the three principal kinds — role, user, tenant."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.modules.users.models import ACL, Permission, Role, Tenant, User, UserRole

    tenant_id = uuid4()
    owner_id = uuid4()
    member_id = uuid4()
    role_id = uuid4()
    dataset_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Tenant(id=tenant_id, name="Acme", owner_id=owner_id))
        for user_id in (owner_id, member_id):
            session.add(
                User(
                    id=user_id,
                    email=f"{user_id}@example.com",
                    hashed_password="x",
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                    tenant_id=tenant_id,
                )
            )
        await session.flush()

        session.add(Role(id=role_id, name="billing", tenant_id=tenant_id))
        await session.flush()
        session.add(UserRole(user_id=member_id, role_id=role_id))

        session.add(Dataset(id=dataset_id, name="ds", owner_id=owner_id, tenant_id=tenant_id))
        await session.flush()

        read_id = await _permission_id(session, "read")
        write_id = await _permission_id(session, "write")

        session.add_all(
            [
                ACL(principal_id=role_id, permission_id=read_id, dataset_id=dataset_id),
                ACL(principal_id=member_id, permission_id=read_id, dataset_id=dataset_id),
                ACL(principal_id=tenant_id, permission_id=write_id, dataset_id=dataset_id),
            ]
        )
        await session.commit()

    return {
        "tenant_id": tenant_id,
        "owner_id": owner_id,
        "member_id": member_id,
        "role_id": role_id,
        "dataset_id": dataset_id,
    }


@pytest.mark.asyncio
async def test_role_membership_and_grant_kinds_are_labelled_correctly():
    from cognee.api.v1.visualize.memory_provenance import _read_roles_and_grants

    seed = await _seed()

    roles, grants = await _read_roles_and_grants(
        tenant_ids=[seed["tenant_id"]],
        dataset_ids=[str(seed["dataset_id"])],
    )

    assert len(roles) == 1
    assert roles[0]["id"] == str(seed["role_id"])
    assert roles[0]["user_ids"] == [str(seed["member_id"])]

    by_principal = {(g["principal_id"], g["permission"]): g["principal_kind"] for g in grants}
    assert by_principal[(str(seed["role_id"]), "read")] == "role"
    assert by_principal[(str(seed["member_id"]), "read")] == "user"
    assert by_principal[(str(seed["tenant_id"]), "write")] == "tenant"
    # The bug this test exists for: none of the three may be mislabelled as
    # one of the others.
    assert len(by_principal) == 3


@pytest.mark.asyncio
async def test_role_in_another_tenant_is_excluded_by_tenant_scope():
    from cognee.api.v1.visualize.memory_provenance import _read_roles_and_grants

    seed = await _seed()

    roles, _ = await _read_roles_and_grants(
        tenant_ids=[uuid4()],
        dataset_ids=[str(seed["dataset_id"])],
    )

    assert roles == []


@pytest.mark.asyncio
async def test_no_tenant_scope_keeps_only_roles_the_scoped_users_belong_to():
    """The OSS/single-user path: no tenant to filter by, so membership does
    the scoping instead."""
    from cognee.api.v1.visualize.memory_provenance import _read_roles_and_grants

    seed = await _seed()

    in_scope, _ = await _read_roles_and_grants(
        tenant_ids=None,
        dataset_ids=[str(seed["dataset_id"])],
        scope_user_ids=[str(seed["member_id"])],
    )
    out_of_scope, _ = await _read_roles_and_grants(
        tenant_ids=None,
        dataset_ids=[str(seed["dataset_id"])],
        scope_user_ids=[str(seed["owner_id"])],
    )

    assert len(in_scope) == 1
    assert in_scope[0]["id"] == str(seed["role_id"])
    assert out_of_scope == []


@pytest.mark.asyncio
async def test_get_memory_provenance_graph_includes_the_role_and_its_grant():
    """End-to-end through the public entry point, not just the private reader."""
    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_graph

    seed = await _seed()

    nodes, edges = await get_memory_provenance_graph(scope_tenant_ids=[seed["tenant_id"]])

    node_types = {nid: props["type"] for nid, props in nodes}
    edge_set = {(s, t, rel) for s, t, rel, _ in edges}

    role_id = str(seed["role_id"])
    tenant_id = str(seed["tenant_id"])
    dataset_id = str(seed["dataset_id"])
    member_id = str(seed["member_id"])

    assert node_types[f"role:{role_id}"] == "Role"
    assert (f"tenant:{tenant_id}", f"role:{role_id}", "has_role") in edge_set
    assert (f"role:{role_id}", f"user:{member_id}", "has_member") in edge_set
    assert (f"role:{role_id}", f"dataset:{dataset_id}", "reads") in edge_set
    assert (f"tenant:{tenant_id}", f"dataset:{dataset_id}", "writes") in edge_set


@pytest.mark.asyncio
async def test_get_memory_provenance_payload_is_json_safe_and_carries_the_role():
    """CLO-401's provenance-as-JSON, routed through the same
    build_visualization_payload as the dataset-visualization JSON endpoint —
    proof the two share one preprocess() call rather than two shapes that
    could drift."""
    import json

    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_payload

    seed = await _seed()

    payload = await get_memory_provenance_payload(scope_tenant_ids=[seed["tenant_id"]])

    json.dumps(payload)  # must not raise
    for field in ("nodes", "links", "color_maps", "provenance_index", "search_events"):
        assert field in payload

    role_id = str(seed["role_id"])
    node_names = {n["id"]: n.get("name") for n in payload["nodes"]}
    assert f"role:{role_id}" in node_names


@pytest.mark.asyncio
async def test_dataset_scope_narrows_the_graph_to_the_callers_readable_set():
    """A tenant scope answers "what exists here", not "what may this member
    read" — the workspace's other datasets used to arrive as named nodes."""
    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_graph

    seed = await _seed()

    nodes, edges = await get_memory_provenance_graph(
        scope_tenant_ids=[seed["tenant_id"]],
        scope_dataset_ids=[uuid4()],
    )

    node_types = {nid: props["type"] for nid, props in nodes}
    assert f"dataset:{seed['dataset_id']}" not in node_types
    # The ACL edges around a dropped dataset go with it, or the grant itself
    # discloses that the dataset exists.
    assert not [e for e in edges if str(seed["dataset_id"]) in e[1]]


@pytest.mark.asyncio
async def test_dataset_scope_keeps_a_dataset_the_caller_may_read():
    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_graph

    seed = await _seed()

    nodes, _ = await get_memory_provenance_graph(
        scope_tenant_ids=[seed["tenant_id"]],
        scope_dataset_ids=[seed["dataset_id"]],
    )

    node_types = {nid: props["type"] for nid, props in nodes}
    assert node_types[f"dataset:{seed['dataset_id']}"] == "Dataset"


@pytest.mark.asyncio
async def test_empty_dataset_scope_shows_no_datasets_rather_than_all_of_them():
    """The failure mode this guards: treating "may read nothing" as "no filter"
    is exactly how the tenant-wide list got published in the first place."""
    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_graph

    seed = await _seed()

    nodes, _ = await get_memory_provenance_graph(
        scope_tenant_ids=[seed["tenant_id"]],
        scope_dataset_ids=[],
    )

    assert [nid for nid, props in nodes if props["type"] == "Dataset"] == []


@pytest.mark.asyncio
async def test_provenance_scope_gives_the_tenant_owner_the_whole_workspace():
    from sqlalchemy import select

    from cognee.api.v1.visualize.routers.get_schema_router import _provenance_scope
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import User

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        owner = (
            (await session.execute(select(User).where(User.id == seed["owner_id"]))).scalars().one()
        )
        tenant_ids, user_ids, dataset_ids = await _provenance_scope(owner)

    assert tenant_ids == [seed["tenant_id"]]
    assert user_ids is None
    assert dataset_ids is None


@pytest.mark.asyncio
async def test_provenance_scope_narrows_a_plain_member_to_their_grants():
    from sqlalchemy import select

    from cognee.api.v1.visualize.routers.get_schema_router import _provenance_scope
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import User

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        member = (
            (await session.execute(select(User).where(User.id == seed["member_id"])))
            .scalars()
            .one()
        )
        tenant_ids, _, dataset_ids = await _provenance_scope(member)

    assert tenant_ids == [seed["tenant_id"]]
    # Not None: a member is scoped to a list, and this seed grants them read on
    # the one dataset it creates.
    assert dataset_ids is not None
    assert seed["dataset_id"] in dataset_ids


@pytest.mark.asyncio
async def test_provenance_scope_drops_a_workspace_dataset_the_member_has_no_grant_on():
    """The reported leak, end to end: a second dataset in the same tenant that
    the member holds nothing on must not reach their scope."""
    from sqlalchemy import select

    from cognee.api.v1.visualize.routers.get_schema_router import _provenance_scope
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset
    from cognee.modules.users.models import User

    seed = await _seed()
    private_dataset_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(
            Dataset(
                id=private_dataset_id,
                name="payroll",
                owner_id=seed["owner_id"],
                tenant_id=seed["tenant_id"],
            )
        )
        await session.commit()

    async with db_engine.get_async_session() as session:
        member = (
            (await session.execute(select(User).where(User.id == seed["member_id"])))
            .scalars()
            .one()
        )
        _, _, dataset_ids = await _provenance_scope(member)

    assert private_dataset_id not in dataset_ids


@pytest.mark.asyncio
async def test_provenance_scope_gives_a_tenant_admin_the_whole_workspace():
    """Administering a tenant is decided in one place for the whole API
    (``has_user_management_permission``: the owner plus the admin role names).
    A deployment that promotes an admin must not have to remember this
    endpoint separately."""
    from sqlalchemy import select

    from cognee.api.v1.visualize.routers.get_schema_router import _provenance_scope
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role, User, UserRole

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        admin_role_id = uuid4()
        session.add(Role(id=admin_role_id, name="admin", tenant_id=seed["tenant_id"]))
        await session.flush()
        session.add(UserRole(user_id=seed["member_id"], role_id=admin_role_id))
        await session.commit()

    async with db_engine.get_async_session() as session:
        member = (
            (await session.execute(select(User).where(User.id == seed["member_id"])))
            .scalars()
            .one()
        )
        tenant_ids, user_ids, dataset_ids = await _provenance_scope(member)

    assert tenant_ids == [seed["tenant_id"]]
    assert user_ids is None
    assert dataset_ids is None


@pytest.mark.asyncio
async def test_dataset_scope_drops_agents_and_sessions_on_out_of_scope_datasets(monkeypatch):
    """The dataset scope narrows the datasets, but the users it reads agents
    and sessions for are still every user in the tenant, so the work done on a
    dataset has to follow the dataset out of the graph."""
    from unittest.mock import AsyncMock

    from cognee.api.v1.visualize import memory_provenance as provenance_module

    seed = await _seed()
    other_dataset_id = str(uuid4())

    monkeypatch.setattr(
        provenance_module,
        "_read_agents",
        AsyncMock(
            return_value=[
                {
                    "id": "agent-in",
                    "name": "in",
                    "user_id": str(seed["member_id"]),
                    "datasets": [{"dataset_id": str(seed["dataset_id"]), "role": "read"}],
                },
                {
                    "id": "agent-out",
                    "name": "out",
                    "user_id": str(seed["owner_id"]),
                    "datasets": [{"dataset_id": other_dataset_id, "role": "read"}],
                },
                {
                    "id": "agent-unattributed",
                    "name": "unattributed",
                    "user_id": str(seed["owner_id"]),
                    "datasets": [],
                },
            ]
        ),
    )
    monkeypatch.setattr(
        provenance_module,
        "_read_sessions",
        AsyncMock(
            return_value=[
                {
                    "id": "session-in",
                    "name": "session-in",
                    "user_id": str(seed["member_id"]),
                    "dataset_id": str(seed["dataset_id"]),
                },
                {
                    "id": "session-out",
                    "name": "session-out",
                    "user_id": str(seed["owner_id"]),
                    "dataset_id": other_dataset_id,
                },
            ]
        ),
    )

    nodes, _ = await provenance_module.get_memory_provenance_graph(
        scope_tenant_ids=[seed["tenant_id"]],
        scope_dataset_ids=[seed["dataset_id"]],
    )

    node_ids = {nid for nid, _props in nodes}
    assert "agent:agent-in" in node_ids
    assert "session:session-in" in node_ids
    assert "agent:agent-out" not in node_ids
    assert "session:session-out" not in node_ids
    # No dataset to check a grant against is the fail-closed half of the rule.
    assert "agent:agent-unattributed" not in node_ids


@pytest.mark.asyncio
async def test_no_dataset_scope_keeps_every_agent_and_session(monkeypatch):
    """The narrowing is opt-in: a tenant administrator asks for the whole
    workspace and must still get the agents and sessions in it."""
    from unittest.mock import AsyncMock

    from cognee.api.v1.visualize import memory_provenance as provenance_module

    seed = await _seed()

    monkeypatch.setattr(
        provenance_module,
        "_read_agents",
        AsyncMock(
            return_value=[
                {
                    "id": "agent-unattributed",
                    "name": "unattributed",
                    "user_id": str(seed["owner_id"]),
                    "datasets": [],
                }
            ]
        ),
    )
    monkeypatch.setattr(
        provenance_module,
        "_read_sessions",
        AsyncMock(
            return_value=[
                {
                    "id": "session-unattributed",
                    "name": "session-unattributed",
                    "user_id": str(seed["owner_id"]),
                    "dataset_id": None,
                }
            ]
        ),
    )

    nodes, _ = await provenance_module.get_memory_provenance_graph(
        scope_tenant_ids=[seed["tenant_id"]],
    )

    node_ids = {nid for nid, _props in nodes}
    assert "agent:agent-unattributed" in node_ids
    assert "session:session-unattributed" in node_ids


@pytest.mark.asyncio
async def test_dataset_scope_drops_a_role_with_no_grant_on_an_in_scope_dataset():
    """A tenant scope alone still handed a plain member every role in the
    tenant, including one with zero relation to any dataset they can read.
    The repo's own GET /permissions/tenants/{id}/roles already withholds
    exactly this from a non-administrator."""
    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_graph
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role, UserRole

    seed = await _seed()
    other_role_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Role(id=other_role_id, name="engineering", tenant_id=seed["tenant_id"]))
        await session.flush()
        session.add(UserRole(user_id=seed["owner_id"], role_id=other_role_id))
        await session.commit()

    nodes, edges = await get_memory_provenance_graph(
        scope_tenant_ids=[seed["tenant_id"]],
        scope_dataset_ids=[seed["dataset_id"]],
    )

    node_ids = {nid for nid, _props in nodes}
    # "billing" holds a read grant on the in-scope dataset (see _seed()) and
    # must survive; "engineering" holds no grant on anything and must not.
    assert f"role:{seed['role_id']}" in node_ids
    assert f"role:{other_role_id}" not in node_ids
    assert not [e for e in edges if str(other_role_id) in e[0] or str(other_role_id) in e[1]]


@pytest.mark.asyncio
async def test_no_dataset_scope_keeps_every_role_regardless_of_grants():
    """The role narrowing is opt-in: a tenant administrator's whole-workspace
    view must not lose a role just because it holds no dataset grant."""
    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_graph
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role

    seed = await _seed()
    ungranted_role_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(Role(id=ungranted_role_id, name="no-grants", tenant_id=seed["tenant_id"]))
        await session.commit()

    nodes, _ = await get_memory_provenance_graph(scope_tenant_ids=[seed["tenant_id"]])

    node_ids = {nid for nid, _props in nodes}
    assert f"role:{ungranted_role_id}" in node_ids


@pytest.mark.asyncio
async def test_dataset_scope_drops_a_tenant_user_with_no_tie_to_an_in_scope_dataset():
    """Same leak, for the user roster: a tenant-only user with no ownership,
    ACL grant, or role membership on an in-scope dataset must not appear,
    matching what GET /permissions/tenants/{id}/users withholds from a
    non-administrator."""
    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_graph
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import User

    seed = await _seed()
    bystander_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(
            User(
                id=bystander_id,
                email=f"{bystander_id}@example.com",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
                tenant_id=seed["tenant_id"],
            )
        )
        await session.commit()

    nodes, _ = await get_memory_provenance_graph(
        scope_tenant_ids=[seed["tenant_id"]],
        scope_dataset_ids=[seed["dataset_id"]],
    )

    node_ids = {nid for nid, _props in nodes}
    # owner (dataset owner) and member (direct ACL + role grant) both have a
    # tie to the in-scope dataset and must survive; the bystander has none.
    assert f"user:{seed['owner_id']}" in node_ids
    assert f"user:{seed['member_id']}" in node_ids
    assert f"user:{bystander_id}" not in node_ids


@pytest.mark.asyncio
async def test_dataset_scope_keeps_a_user_who_only_belongs_to_a_role_with_a_grant():
    """A user reachable only through role membership (no direct ACL row of
    their own) must still surface, or the kept role's has_member edge would
    silently vanish (add_edge drops an edge whose endpoint node is missing)."""
    from cognee.api.v1.visualize.memory_provenance import get_memory_provenance_graph
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import Role, User, UserRole

    seed = await _seed()
    role_only_user_id = uuid4()
    role_only_role_id = uuid4()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(
            User(
                id=role_only_user_id,
                email=f"{role_only_user_id}@example.com",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
                tenant_id=seed["tenant_id"],
            )
        )
        session.add(Role(id=role_only_role_id, name="readers", tenant_id=seed["tenant_id"]))
        await session.flush()
        session.add(UserRole(user_id=role_only_user_id, role_id=role_only_role_id))

        read_id = await _permission_id(session, "read")
        from cognee.modules.users.models import ACL

        session.add(
            ACL(
                principal_id=role_only_role_id,
                permission_id=read_id,
                dataset_id=seed["dataset_id"],
            )
        )
        await session.commit()

    nodes, edges = await get_memory_provenance_graph(
        scope_tenant_ids=[seed["tenant_id"]],
        scope_dataset_ids=[seed["dataset_id"]],
    )

    node_ids = {nid for nid, _props in nodes}
    edge_set = {(s, t, rel) for s, t, rel, _ in edges}
    assert f"user:{role_only_user_id}" in node_ids
    assert (
        f"role:{role_only_role_id}",
        f"user:{role_only_user_id}",
        "has_member",
    ) in edge_set


@pytest.mark.asyncio
async def test_administers_tenant_denial_logs_at_debug_not_error(caplog):
    """A plain member failing the administrator check is the expected common
    case for this read-only view, not a denial worth an ERROR log line on
    every request (has_user_management_permission's other callers keep their
    ERROR default; only this call site opts into DEBUG)."""
    import logging

    from sqlalchemy import select

    from cognee.api.v1.visualize.routers.get_schema_router import _administers_tenant
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import User

    seed = await _seed()

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        member = (
            (await session.execute(select(User).where(User.id == seed["member_id"])))
            .scalars()
            .one()
        )
        with caplog.at_level(logging.ERROR):
            result = await _administers_tenant(member, seed["tenant_id"])

    assert result is False
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
