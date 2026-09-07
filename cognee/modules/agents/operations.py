from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional
from uuid import UUID as UUIDType

from cognee.modules.agents.models import (
    AgentConnection,
    AgentDatasetRef,
    AgentDetailResponse,
    AgentsListResponse,
    MemorySourceConnection,
    RegisterAgentRequest,
)
from cognee.modules.agents.registry import (
    classify_memory_source_type,
    list_persisted_agent_connections,
    list_registered_agent_connections,
    register_agent_connection,
)

# Import from the defining modules, not the package __init__: these packages
# are mid-initialization on some import orders (their __init__ transitively
# reaches back into agents.operations), and a from-import against a partially
# initialized package silently binds the same-named SUBMODULE instead of the
# function ("'module' object is not callable" at call time).
from cognee.modules.users.methods.get_visible_user_ids import get_visible_user_ids
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods.get_readable_datasets import (
    get_readable_datasets,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("agents")

RangeLiteral = Literal["24h", "7d", "30d", "all"]


def _range_since(range_key: RangeLiteral) -> Optional[datetime]:
    now = datetime.now(timezone.utc)
    if range_key == "24h":
        return now - timedelta(hours=24)
    if range_key == "7d":
        return now - timedelta(days=7)
    if range_key == "30d":
        return now - timedelta(days=30)
    return None


def _entry_to_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return entry
    if hasattr(entry, "model_dump"):
        return entry.model_dump(mode="json")
    if hasattr(entry, "dict"):
        return entry.dict()
    return {"value": str(entry)}


def _memory_sources_from_datasets(datasets: list[Any]) -> list[MemorySourceConnection]:
    sources = []
    for dataset in datasets:
        dataset_id = getattr(dataset, "id", None)
        dataset_name = getattr(dataset, "name", None)
        if dataset_id is None or dataset_name is None:
            continue
        sources.append(
            MemorySourceConnection(
                id=str(dataset_id),
                name=str(dataset_name),
                type=classify_memory_source_type(str(dataset_name)),
                owner_id=getattr(dataset, "owner_id", None),
                tenant_id=getattr(dataset, "tenant_id", None),
                status="active",
            )
        )
    return sources


def _is_visible_registered_agent(
    agent: AgentConnection,
    *,
    visible_user_ids: set[UUIDType],
    permitted_dataset_ids: set[str],
) -> bool:
    if agent.user_id and agent.user_id in visible_user_ids:
        return True
    if any(dataset.id in permitted_dataset_ids for dataset in agent.datasets if dataset.id):
        return True
    return agent.user_id is None and not agent.datasets


def _merge_agents(agents: list[AgentConnection]) -> list[AgentConnection]:
    merged: dict[str, AgentConnection] = {}
    for agent in agents:
        existing = merged.get(agent.id)
        if existing is None:
            merged[agent.id] = agent
            continue
        existing_ts = existing.last_active_at or datetime.min.replace(tzinfo=timezone.utc)
        agent_ts = agent.last_active_at or datetime.min.replace(tzinfo=timezone.utc)
        if agent_ts >= existing_ts:
            metadata = {**existing.metadata, **agent.metadata}
            merged[agent.id] = agent.model_copy(update={"metadata": metadata})
        else:
            metadata = {**agent.metadata, **existing.metadata}
            merged[agent.id] = existing.model_copy(update={"metadata": metadata})
    return sorted(
        merged.values(),
        key=lambda item: item.last_active_at or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def _attach_connected_agent_ids(
    memory_sources: list[MemorySourceConnection],
    agents: list[AgentConnection],
) -> list[MemorySourceConnection]:
    agents_by_dataset: dict[str, list[str]] = {}
    for agent in agents:
        for dataset in agent.datasets:
            if dataset.id:
                agents_by_dataset.setdefault(dataset.id, []).append(agent.id)

    return [
        source.model_copy(
            update={"connected_agent_ids": sorted(set(agents_by_dataset.get(source.id, [])))}
        )
        for source in memory_sources
    ]


async def list_agent_connections(
    *,
    user: User,
    agent_id: Optional[UUIDType] = None,
    range_key: RangeLiteral = "30d",
    status_filter: Optional[str] = None,
    include_sources: bool = True,
    active_only: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> AgentsListResponse:
    readable_datasets = await get_readable_datasets(user.id)
    memory_sources = _memory_sources_from_datasets(readable_datasets)
    visible_user_ids = await get_visible_user_ids(user.id)
    visible_user_id_set = set(visible_user_ids)
    permitted_dataset_id_strings = {source.id for source in memory_sources}

    registered_agents = [
        agent
        for agent in list_registered_agent_connections()
        if _is_visible_registered_agent(
            agent,
            visible_user_ids=visible_user_id_set,
            permitted_dataset_ids=permitted_dataset_id_strings,
        )
    ]
    persisted_agents = await list_persisted_agent_connections(
        visible_user_ids, active_only=active_only
    )
    agents = _merge_agents([*registered_agents, *persisted_agents])
    if agent_id:
        agent_id_str = str(agent_id)
        agents = [
            agent
            for agent in agents
            if agent.user_id is not None and str(agent.user_id) == agent_id_str
        ]
    if status_filter:
        agents = [agent for agent in agents if agent.status == status_filter]

    memory_sources = _attach_connected_agent_ids(memory_sources, agents) if include_sources else []
    total = len(agents)
    sliced_agents = agents[offset : offset + limit]
    return AgentsListResponse(
        agents=sliced_agents,
        memory_sources=memory_sources,
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(sliced_agents) < total,
    )


async def get_agent_connection_detail(
    *,
    user: User,
    agent_id: UUIDType,
    agent_session_name: Optional[str] = None,
) -> Optional[AgentDetailResponse]:
    # A lookup by name must still find a connection that unregister has
    # deactivated; only the unnamed lookup keeps the active-only default.
    response = await list_agent_connections(
        user=user,
        agent_id=agent_id,
        include_sources=True,
        active_only=not agent_session_name,
        limit=10000,
        offset=0,
    )
    if agent_session_name:
        matching = [
            item for item in response.agents if item.agent_session_name == agent_session_name
        ]
    else:
        matching = list(response.agents)

    agent = matching[0] if matching else None
    if agent is None:
        return None

    recent_sessions = []
    recent_traces = []
    recent_qas = []
    if agent.session_id and agent.user_id:
        try:
            from cognee.infrastructure.session.get_session_manager import get_session_manager

            session_manager = get_session_manager()
            qas = await session_manager.get_session(
                user_id=agent.user_id,
                session_id=agent.session_id,
                formatted=False,
            )
            traces = await session_manager.get_agent_trace_session(
                user_id=agent.user_id,
                session_id=agent.session_id,
                last_n=20,
            )
            recent_qas = (
                [_entry_to_dict(entry) for entry in qas[-20:]] if isinstance(qas, list) else []
            )
            recent_traces = [_entry_to_dict(entry) for entry in traces[-20:]]
            recent_sessions = [{"session_id": agent.session_id, "user_id": agent.user_id}]
        except Exception as error:
            logger.debug("Failed to hydrate agent detail from session cache: %s", error)

    return AgentDetailResponse(
        agent=agent,
        memory_sources=response.memory_sources,
        recent_sessions=recent_sessions,
        recent_traces=recent_traces,
        recent_qas=recent_qas,
    )


async def register_agent_from_request(user: User, request: RegisterAgentRequest) -> AgentConnection:
    datasets = [
        AgentDatasetRef(id=dataset_id, role="read_write", type="dataset")
        for dataset_id in request.dataset_ids
    ]
    datasets.extend(
        AgentDatasetRef(
            name=dataset_name,
            role="read_write",
            type=classify_memory_source_type(dataset_name),
        )
        for dataset_name in request.dataset_names
    )
    return await register_agent_connection(
        agent_session_name=request.agent_session_name,
        connection_type=request.type,
        memory_mode=request.memory_mode,
        source=request.source,
        origin_function=request.origin_function,
        user_id=user.id,
        tenant_id=getattr(user, "tenant_id", None),
        session_id=request.session_id,
        datasets=datasets,
        metadata=request.metadata,
    )
