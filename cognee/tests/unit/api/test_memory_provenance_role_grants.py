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
        from cognee.infrastructure.databases.relational import get_relational_engine

        import cognee.modules.data.models  # noqa: F401
        import cognee.modules.users.models  # noqa: F401

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
