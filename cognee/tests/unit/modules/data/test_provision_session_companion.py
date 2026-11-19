"""The session companion copies the primary's ACL snapshot once and verifies it after (SDK-303).

Runs on in-memory SQLite with the real models. The owner, or an agent user
whose parent is the owner, may provision; anyone else is refused. Drift between
the companion's and the primary's permissions is a 409, never a silent change.
"""

import importlib
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.exceptions import CogneeApiError
from cognee.modules.data.methods.get_unique_dataset_id import get_unique_dataset_id
from cognee.modules.data.models import Dataset
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models.ACL import ACL
from cognee.modules.users.models.Permission import Permission
from cognee.modules.users.models.Principal import Principal
from cognee.modules.users.models.User import User
from cognee.modules.users.permissions import PERMISSION_TYPES

provision_module = importlib.import_module(
    "cognee.modules.data.methods.provision_session_companion"
)
unique_id_module = importlib.import_module("cognee.modules.data.methods.get_unique_dataset_id")

TABLES = [
    Principal.__table__,
    User.__table__,
    Permission.__table__,
    Dataset.__table__,
    ACL.__table__,
]


class _SqliteEngine:
    def __init__(self, engine):
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    def get_async_session(self):
        return self._sessionmaker()


@pytest_asyncio.fixture
async def relational_engine(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Dataset.metadata.create_all, tables=TABLES)
    fake = _SqliteEngine(engine)
    monkeypatch.setattr(provision_module, "get_relational_engine", lambda: fake)
    monkeypatch.setattr(unique_id_module, "get_relational_engine", lambda: fake)
    yield fake
    await engine.dispose()


def _user(parent_user_id=None, tenant_id=None) -> User:
    return User(
        id=uuid4(),
        email=f"{uuid4().hex}@example.com",
        hashed_password="x",
        tenant_id=tenant_id,
        parent_user_id=parent_user_id,
    )


@pytest_asyncio.fixture
async def world(relational_engine):
    """An owner with a fully permissioned primary dataset, an agent of the owner, a stranger."""
    owner = _user()
    agent = _user(parent_user_id=owner.id)
    stranger = _user()
    permissions = [Permission(id=uuid4(), name=name) for name in PERMISSION_TYPES]
    primary = Dataset(id=uuid4(), name="docs", owner_id=owner.id, tenant_id=None)
    async with relational_engine.get_async_session() as session:
        session.add_all([owner, agent, stranger, *permissions, primary])
        await session.flush()
        session.add_all(
            ACL(dataset_id=primary.id, principal_id=owner.id, permission_id=permission.id)
            for permission in permissions
        )
        session.add(
            ACL(dataset_id=primary.id, principal_id=agent.id, permission_id=permissions[0].id)
        )
        await session.commit()
    return {"owner": owner, "agent": agent, "stranger": stranger, "primary": primary}


async def _acls(engine, dataset_id):
    async with engine.get_async_session() as session:
        rows = (await session.scalars(select(ACL).where(ACL.dataset_id == dataset_id))).all()
    return {(row.principal_id, row.permission_id) for row in rows}


@pytest.mark.asyncio
async def test_owner_gets_a_companion_with_the_primary_permission_snapshot(
    relational_engine, world
):
    primary, owner = world["primary"], world["owner"]

    result = await provision_module.provision_session_companion(primary.id, owner)

    expected_id = await get_unique_dataset_id("docs-agent_sessions", owner)
    assert result == {
        "primary_dataset_id": str(primary.id),
        "dataset_id": str(expected_id),  # the id POST /datasets would derive for the owner
        "dataset_name": "docs-agent_sessions",
        "permissions_verified": True,
    }
    assert await _acls(relational_engine, expected_id) == await _acls(relational_engine, primary.id)
    async with relational_engine.get_async_session() as session:
        companion = await session.get(Dataset, expected_id)
    assert (companion.owner_id, companion.tenant_id) == (owner.id, None)


@pytest.mark.asyncio
async def test_a_second_call_verifies_and_returns_the_same_identity(relational_engine, world):
    first = await provision_module.provision_session_companion(world["primary"].id, world["owner"])

    second = await provision_module.provision_session_companion(world["primary"].id, world["owner"])

    assert second == first
    async with relational_engine.get_async_session() as session:
        count = len((await session.scalars(select(Dataset))).all())
    assert count == 2  # primary + one companion


@pytest.mark.asyncio
async def test_an_agent_of_the_owner_may_provision(relational_engine, world):
    result = await provision_module.provision_session_companion(world["primary"].id, world["agent"])

    assert result["permissions_verified"] is True
    assert result["dataset_name"] == "docs-agent_sessions"


@pytest.mark.asyncio
async def test_anyone_else_is_refused_with_403(relational_engine, world):
    with pytest.raises(PermissionDeniedError) as raised:
        await provision_module.provision_session_companion(world["primary"].id, world["stranger"])
    assert raised.value.status_code == 403

    with pytest.raises(PermissionDeniedError):
        await provision_module.provision_session_companion(uuid4(), world["owner"])


@pytest.mark.asyncio
async def test_permission_drift_is_a_409_not_a_silent_change(relational_engine, world):
    result = await provision_module.provision_session_companion(world["primary"].id, world["owner"])
    async with relational_engine.get_async_session() as session:
        rows = (
            await session.scalars(select(ACL).where(ACL.dataset_id == UUID(result["dataset_id"])))
        ).all()
        await session.delete(rows[0])
        await session.commit()

    with pytest.raises(provision_module.CompanionConflictError) as raised:
        await provision_module.provision_session_companion(world["primary"].id, world["owner"])

    assert raised.value.status_code == 409
    assert isinstance(raised.value, CogneeApiError)


@pytest.mark.asyncio
async def test_a_companion_has_no_companion(relational_engine, world):
    result = await provision_module.provision_session_companion(world["primary"].id, world["owner"])

    with pytest.raises(provision_module.CompanionConflictError):
        await provision_module.provision_session_companion(
            UUID(result["dataset_id"]), world["owner"]
        )


def test_route_and_response_keys_are_the_ones_the_plugins_read():
    from fastapi import FastAPI

    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router

    app = FastAPI()
    app.include_router(get_datasets_router(), prefix="/api/v1/datasets")
    spec = app.openapi()

    operation = spec["paths"]["/api/v1/datasets/{dataset_id}/session-companion"]["post"]
    schema_name = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    properties = spec["components"]["schemas"][schema_name.rsplit("/", 1)[1]]["properties"]
    assert set(properties) == {
        "primary_dataset_id",
        "dataset_id",
        "dataset_name",
        "permissions_verified",
    }
