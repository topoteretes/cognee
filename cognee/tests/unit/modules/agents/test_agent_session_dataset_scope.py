from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.agents.models import (
    AgentConnection,
    AgentDatasetRef,
    AgentsListResponse,
    RegisterAgentRequest,
)
from cognee.modules.agents.operations import (
    _visible_registered_agent,
    get_agent_connection_detail,
    register_agent_from_request,
)


def test_dataset_grant_exposes_only_matching_agent_dataset_refs():
    owner_id = uuid4()
    allowed_id = str(uuid4())
    secret_id = str(uuid4())
    agent = AgentConnection(
        id="agent-1",
        agent_session_name="shared-agent",
        user_id=owner_id,
        datasets=[
            AgentDatasetRef(id=allowed_id, name="allowed"),
            AgentDatasetRef(id=secret_id, name="secret"),
        ],
    )

    visible = _visible_registered_agent(
        agent,
        visible_user_ids=set(),
        permitted_dataset_ids={allowed_id},
    )

    assert visible is not None
    assert [dataset.id for dataset in visible.datasets] == [allowed_id]
    # Filtering returns a copy and does not corrupt the owner's registry entry.
    assert [dataset.id for dataset in agent.datasets] == [allowed_id, secret_id]


@pytest.mark.asyncio
async def test_registration_rejects_unauthorized_dataset_before_registry_write(monkeypatch):
    import cognee.modules.agents.operations as operations

    register = AsyncMock()
    monkeypatch.setattr(operations, "register_agent_connection", register)
    monkeypatch.setattr(operations, "get_authorized_dataset", AsyncMock(return_value=None))

    user = type("User", (), {"id": uuid4(), "tenant_id": None})()
    request = RegisterAgentRequest(
        agent_session_name="untrusted",
        dataset_ids=[str(uuid4())],
    )

    with pytest.raises(CogneeValidationError, match="not found or is not writable"):
        await register_agent_from_request(user, request)

    register.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_detail_does_not_hydrate_after_dataset_access_is_revoked(monkeypatch):
    import cognee.modules.agents.operations as operations

    owner_id = uuid4()
    dataset_id = str(uuid4())
    agent = AgentConnection(
        id="agent-1",
        agent_session_name="revoked",
        user_id=owner_id,
        session_id="session-1",
        datasets=[AgentDatasetRef(id=dataset_id, name="revoked-dataset")],
    )
    monkeypatch.setattr(
        operations,
        "list_agent_connections",
        AsyncMock(
            return_value=AgentsListResponse(
                agents=[agent], total=1, limit=10000, offset=0, has_more=False
            )
        ),
    )
    monkeypatch.setattr(operations, "_readable_datasets_for", AsyncMock(return_value=[]))

    manager = type(
        "Manager",
        (),
        {
            "get_session": AsyncMock(),
            "get_agent_trace_session": AsyncMock(),
        },
    )()
    manager_module = __import__(
        "cognee.infrastructure.session.get_session_manager", fromlist=["get_session_manager"]
    )
    get_manager = MagicMock(return_value=manager)
    monkeypatch.setattr(manager_module, "get_session_manager", get_manager)

    detail = await get_agent_connection_detail(
        user=type("User", (), {"id": owner_id})(), agent_id=owner_id
    )

    assert detail is not None
    assert detail.recent_qas == []
    assert detail.recent_traces == []
    manager.get_session.assert_not_awaited()
    manager.get_agent_trace_session.assert_not_awaited()
    get_manager.assert_not_called()
