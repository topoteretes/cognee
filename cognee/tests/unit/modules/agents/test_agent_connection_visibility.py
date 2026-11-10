"""Whose agent connections a caller may see, and which datasets they may register (SDK-927).

A registered connection used to be shown to anyone who could read a dataset it
listed, and the detail endpoint then returned that user's recent questions,
answers and traces from the session cache. A connection now belongs to its user
and the users above them in ``get_visible_user_ids``, and registering one
requires read access to every dataset it lists. A persisted connection is only
trusted for the user whose configuration holds it, since that configuration is
writable by its owner. Runs without a database: the registry, the visibility
lookup, the configuration store and the session cache are replaced at the names
the code imports.
"""

import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.infrastructure.session.get_session_manager  # registers the submodule
import cognee.modules.users.methods.get_principal_configuration  # registers the submodule
from cognee.modules.agents import operations, registry
from cognee.modules.agents.models import AgentConnection, AgentDatasetRef, RegisterAgentRequest
from cognee.modules.users.exceptions import PermissionDeniedError


class _SessionManager:
    def __init__(self):
        self.reads = []

    async def get_session(self, *, user_id, session_id, formatted):
        self.reads.append(("qa", user_id))
        return [{"question": "q", "answer": "a"}]

    async def get_agent_trace_session(self, *, user_id, session_id, last_n):
        self.reads.append(("trace", user_id))
        return [{"method": "m"}]


# The package __init__ rebinds `get_session_manager` to the function of the same
# name, so the module that operations imports it from is only reachable here.
_session_manager_module = sys.modules["cognee.infrastructure.session.get_session_manager"]
_configuration_module = sys.modules["cognee.modules.users.methods.get_principal_configuration"]


@pytest.fixture
def session_manager(monkeypatch):
    manager = _SessionManager()
    monkeypatch.setattr(_session_manager_module, "get_session_manager", lambda: manager)
    return manager


def _setup(monkeypatch, *, visible_user_ids, connections, readable_dataset_ids=()):
    async def readable_datasets(_user_id):
        return [
            SimpleNamespace(id=dataset_id, name="shared") for dataset_id in readable_dataset_ids
        ]

    async def visible(_user_id):
        return list(visible_user_ids)

    async def persisted(_user_ids, active_only=True):
        return []

    monkeypatch.setattr(operations, "get_readable_datasets", readable_datasets)
    monkeypatch.setattr(operations, "get_visible_user_ids", visible)
    monkeypatch.setattr(operations, "list_registered_agent_connections", lambda: connections)
    monkeypatch.setattr(operations, "list_persisted_agent_connections", persisted)


def _connection(user_id, dataset_id=None, name="agent"):
    return AgentConnection(
        id=f"{name}-{user_id}",
        agent_session_name=name,
        user_id=user_id,
        session_id="session-1",
        datasets=[AgentDatasetRef(id=str(dataset_id))] if dataset_id else [],
    )


@pytest.mark.asyncio
async def test_a_connection_is_not_shown_to_someone_who_only_shares_its_dataset(
    monkeypatch, session_manager
):
    me, other, dataset_id = uuid4(), uuid4(), uuid4()
    _setup(
        monkeypatch,
        visible_user_ids=[me],
        connections=[_connection(other, dataset_id)],
        readable_dataset_ids=[str(dataset_id)],
    )

    listing = await operations.list_agent_connections(user=SimpleNamespace(id=me))

    assert listing.agents == []
    assert [source.connected_agent_ids for source in listing.memory_sources] == [[]]


@pytest.mark.asyncio
async def test_the_detail_of_a_connection_that_is_not_theirs_reads_no_session_data(
    monkeypatch, session_manager
):
    me, other, dataset_id = uuid4(), uuid4(), uuid4()
    _setup(
        monkeypatch,
        visible_user_ids=[me],
        connections=[_connection(other, dataset_id)],
        readable_dataset_ids=[str(dataset_id)],
    )

    detail = await operations.get_agent_connection_detail(
        user=SimpleNamespace(id=me), agent_id=other
    )

    assert detail is None
    assert session_manager.reads == []


@pytest.mark.asyncio
async def test_a_parent_still_sees_the_connection_questions_and_traces_of_its_agent(
    monkeypatch, session_manager
):
    me, agent = uuid4(), uuid4()
    _setup(monkeypatch, visible_user_ids=[me, agent], connections=[_connection(agent)])

    detail = await operations.get_agent_connection_detail(
        user=SimpleNamespace(id=me), agent_id=agent
    )

    assert detail.agent.user_id == agent
    assert detail.recent_qas == [{"question": "q", "answer": "a"}]
    assert detail.recent_traces == [{"method": "m"}]


@pytest.mark.asyncio
async def test_only_the_callers_connections_are_listed_and_counted(monkeypatch, session_manager):
    me, other, dataset_id = uuid4(), uuid4(), uuid4()
    _setup(
        monkeypatch,
        visible_user_ids=[me],
        connections=[
            _connection(me, name="mine"),
            _connection(other, dataset_id, name="theirs"),
            _connection(None, name="nobodys"),
        ],
        readable_dataset_ids=[str(dataset_id)],
    )

    listing = await operations.list_agent_connections(user=SimpleNamespace(id=me))

    assert [agent.agent_session_name for agent in listing.agents] == ["mine"]
    assert (listing.total, listing.has_more) == (1, False)


@pytest.mark.asyncio
async def test_a_connection_stored_in_ones_own_configuration_cannot_name_another_user(
    monkeypatch, session_manager
):
    """The configuration endpoint lets anyone write their own agent blob, so a
    stored entry naming someone else would otherwise open that user's sessions."""
    me, other = uuid4(), uuid4()
    _setup(monkeypatch, visible_user_ids=[me], connections=[])
    monkeypatch.setattr(
        operations, "list_persisted_agent_connections", registry.list_persisted_agent_connections
    )

    async def configuration(user_id):
        assert user_id == me
        entries = {
            "mine": {
                "id": "mine",
                "agent_session_name": "mine",
                "user_id": str(me),
                "status": "active",
            },
            "forged": {
                "id": "forged",
                "agent_session_name": "forged",
                "user_id": str(other),
                "session_id": "default_session",
                "status": "active",
            },
            "malformed": {"agent_session_name": ["not", "a", "string"]},
        }
        return [{"name": registry.AGENT_CONFIG_NAME, "configuration": {"agents": entries}}]

    monkeypatch.setattr(_configuration_module, "get_principal_all_configuration", configuration)
    user = SimpleNamespace(id=me)

    listing = await operations.list_agent_connections(user=user)
    detail = await operations.get_agent_connection_detail(user=user, agent_id=other)

    assert [agent.agent_session_name for agent in listing.agents] == ["mine"]
    assert detail is None
    assert session_manager.reads == []


@pytest.fixture
def world(monkeypatch):
    """A caller who owns one readable dataset and one they have lost read on."""
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    readable, unreadable = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())
    checks, registered = [], []

    async def get_authorized_dataset(caller, dataset_id, permission):
        checks.append((caller, permission))
        return readable if permission == "read" and dataset_id == readable.id else None

    async def get_datasets_by_name(name, owner_id):
        assert owner_id == user.id
        return {"mine": [readable], "mine_without_read": [unreadable]}.get(name, [])

    async def register_agent_connection(**kwargs):
        registered.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(operations, "get_authorized_dataset", get_authorized_dataset)
    monkeypatch.setattr(operations, "get_datasets_by_name", get_datasets_by_name)
    monkeypatch.setattr(operations, "register_agent_connection", register_agent_connection)
    return SimpleNamespace(user=user, readable=readable, checks=checks, registered=registered)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_fields",
    [
        # Each list starts with a readable entry, so only checking the first one fails.
        lambda world: {"dataset_ids": [str(world.readable.id), str(uuid4())]},
        lambda world: {"dataset_ids": [str(world.readable.id), "not-a-uuid"]},
        lambda world: {"dataset_names": ["mine", "theirs"]},
        lambda world: {"dataset_names": ["mine", "mine_without_read"]},
    ],
    ids=["unreadable id", "malformed id", "name the caller does not own", "owned without read"],
)
async def test_register_rejects_a_dataset_the_caller_cannot_read(world, request_fields):
    with pytest.raises(PermissionDeniedError):
        await operations.register_agent_from_request(
            world.user, RegisterAgentRequest(agent_session_name="agent", **request_fields(world))
        )

    assert world.registered == []


@pytest.mark.asyncio
async def test_register_accepts_readable_datasets_and_stores_their_canonical_ids(world):
    await operations.register_agent_from_request(
        world.user,
        RegisterAgentRequest(
            agent_session_name="agent",
            dataset_ids=[str(world.readable.id).upper()],
            dataset_names=["mine"],
        ),
    )

    [registration] = world.registered
    assert registration["datasets"][0].id == str(world.readable.id)
    assert world.checks == [(world.user, "read"), (world.user, "read")]
