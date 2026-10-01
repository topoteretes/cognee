"""A stored configuration is readable by its owner and the owner's parent user (SDK-803).

``get_principal_configuration`` used to select by config id alone, so any
authenticated user holding another principal's config id could read it. The
lookup is now scoped to the caller plus the agent users they are the parent of
(``get_visible_user_ids``): anyone else's config reads exactly like a missing
one, and an agent cannot read upward to its parent.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.users.models.PrincipalConfiguration import PrincipalConfiguration

# The ``methods`` package re-exports each function under its module's own name,
# so hold the modules themselves for patching.
get_module = importlib.import_module("cognee.modules.users.methods.get_principal_configuration")
store_module = importlib.import_module("cognee.modules.users.methods.store_principal_configuration")
router_module = importlib.import_module("cognee.api.v1.users.routers.get_configuration_router")


class _SqliteEngine:
    """The one relational-engine method these functions use, on in-memory SQLite."""

    def __init__(self, engine):
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    def get_async_session(self):
        return self._sessionmaker()


@pytest.fixture
def children(monkeypatch):
    """Parent -> child agent ids, standing in for ``User.parent_user_id`` rows."""
    tree: dict = {}

    async def visible_user_ids(user_id):
        return [user_id, *tree.get(user_id, [])]

    monkeypatch.setattr(get_module, "get_visible_user_ids", visible_user_ids)
    monkeypatch.setattr(router_module, "get_visible_user_ids", visible_user_ids)
    return tree


@pytest.fixture
async def relational_engine(monkeypatch, children):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(
            PrincipalConfiguration.metadata.create_all, tables=[PrincipalConfiguration.__table__]
        )

    fake_engine = _SqliteEngine(engine)
    monkeypatch.setattr(get_module, "get_relational_engine", lambda: fake_engine)
    monkeypatch.setattr(store_module, "get_relational_engine", lambda: fake_engine)
    yield fake_engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_owner_reads_own_configuration(relational_engine):
    owner_id = uuid4()
    record = await store_module.store_principal_configuration(
        principal_id=owner_id, name="llm", configuration={"model": "m"}
    )

    result = await get_module.get_principal_configuration(
        config_id=record.id, principal_id=owner_id
    )

    assert result == {"model": "m"}


@pytest.mark.asyncio
async def test_other_principal_gets_same_result_as_missing_id(relational_engine):
    owner_id = uuid4()
    record = await store_module.store_principal_configuration(
        principal_id=owner_id, name="llm", configuration={"model": "m"}
    )

    other_principal_read = await get_module.get_principal_configuration(
        config_id=record.id, principal_id=uuid4()
    )
    missing_id_read = await get_module.get_principal_configuration(
        config_id=uuid4(), principal_id=owner_id
    )

    assert other_principal_read == {}
    assert other_principal_read == missing_id_read


@pytest.mark.asyncio
async def test_parent_reads_its_agents_configuration(relational_engine, children):
    parent_id, agent_id = uuid4(), uuid4()
    children[parent_id] = [agent_id]
    record = await store_module.store_principal_configuration(
        principal_id=agent_id, name="llm", configuration={"model": "agent"}
    )

    result = await get_module.get_principal_configuration(
        config_id=record.id, principal_id=parent_id
    )

    assert result == {"model": "agent"}


@pytest.mark.asyncio
async def test_agent_cannot_read_its_parents_configuration(relational_engine, children):
    parent_id, agent_id = uuid4(), uuid4()
    children[parent_id] = [agent_id]
    record = await store_module.store_principal_configuration(
        principal_id=parent_id, name="llm", configuration={"model": "parent"}
    )

    result = await get_module.get_principal_configuration(
        config_id=record.id, principal_id=agent_id
    )

    assert result == {}


@pytest.mark.asyncio
async def test_list_endpoint_includes_agent_records_tagged_by_owner(relational_engine, children):
    parent_id, agent_id, stranger_id = uuid4(), uuid4(), uuid4()
    children[parent_id] = [agent_id]
    await store_module.store_principal_configuration(
        principal_id=parent_id, name="own", configuration={"model": "parent"}
    )
    await store_module.store_principal_configuration(
        principal_id=agent_id, name="delegated", configuration={"model": "agent"}
    )
    await store_module.store_principal_configuration(
        principal_id=stranger_id, name="private", configuration={"model": "stranger"}
    )
    router = router_module.get_configuration_router()
    list_endpoint = next(
        route.endpoint for route in router.routes if route.path == "/get_user_configuration/"
    )

    records = await list_endpoint(user=SimpleNamespace(id=parent_id))

    assert {(r["ownerId"], r["name"]) for r in records} == {
        (str(parent_id), "own"),
        (str(agent_id), "delegated"),
    }
