from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional
from uuid import UUID as UUIDType

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.agents.models import (
    AgentConnection,
    AgentDatasetRef,
    AgentDetailResponse,
    AgentsListResponse,
    MemorySourceConnection,
    RegisterAgentRequest,
)
from cognee.modules.agents.registry import (
    build_agent_connection_id,
    classify_memory_source_type,
    derive_connection_type,
    list_registered_agent_connections,
    register_agent_connection,
)
from cognee.modules.session_lifecycle.metrics import list_session_rows
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
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


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _entry_to_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return entry
    if hasattr(entry, "model_dump"):
        return entry.model_dump(mode="json")
    if hasattr(entry, "dict"):
        return entry.dict()
    return {"value": str(entry)}


async def _readable_datasets_for(user: User) -> list[Any]:
    try:
        return await get_specific_user_permission_datasets(user.id, "read", None)
    except PermissionDeniedError:
        return []
    except Exception as error:
        logger.warning("Failed to resolve readable datasets for agents API: %s", error)
        return []


async def _child_agent_user_ids(user_id: UUIDType) -> list[UUIDType]:
    try:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            from cognee.modules.users.models import User as UserModel

            rows = (
                await session.execute(
                    select(UserModel.id).where(UserModel.parent_user_id == user_id)
                )
            ).all()
            return [row.id for row in rows]
    except Exception as error:
        logger.debug("Failed to resolve child agent users for agents API: %s", error)
        return []


async def _visible_user_ids(user: User) -> list[UUIDType]:
    ids = [user.id]
    ids.extend(await _child_agent_user_ids(user.id))
    return ids


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
                owner_id=str(getattr(dataset, "owner_id", "")) or None,
                tenant_id=str(getattr(dataset, "tenant_id", "")) or None,
                status="active",
            )
        )
    return sources


def _dataset_ref_for_id(
    dataset_id: Any,
    memory_sources_by_id: dict[str, MemorySourceConnection],
) -> Optional[AgentDatasetRef]:
    if dataset_id is None:
        return None
    source = memory_sources_by_id.get(str(dataset_id))
    if source is None:
        return AgentDatasetRef(id=str(dataset_id), role="read", type="dataset")
    return AgentDatasetRef(id=source.id, name=source.name, role="read", type=source.type)


async def _trace_agents_for_user(
    *,
    user: User,
    visible_user_ids: list[UUIDType],
    permitted_dataset_ids: list[UUIDType],
    memory_sources_by_id: dict[str, MemorySourceConnection],
    since: Optional[datetime],
) -> list[AgentConnection]:
    try:
        page = await list_session_rows(
            user_ids=visible_user_ids,
            permitted_dataset_ids=permitted_dataset_ids,
            since=since,
            status_filter=None,
            limit=500,
            offset=0,
            order_by="last_activity_at",
            descending=True,
        )
    except Exception as error:
        logger.warning("Failed to list session rows for agents API: %s", error)
        return []

    try:
        from cognee.infrastructure.session.get_session_manager import get_session_manager

        session_manager = get_session_manager()
    except Exception:
        session_manager = None

    agents = []
    for row in page.sessions:
        record = row.record
        owner_user_id = str(getattr(record, "user_id", ""))
        session_id = getattr(record, "session_id", None)
        if not owner_user_id or not session_id or session_manager is None:
            continue

        try:
            traces = await session_manager.get_agent_trace_session(
                user_id=owner_user_id,
                session_id=session_id,
                last_n=20,
            )
        except Exception:
            traces = []
        if not traces:
            continue

        first_trace = traces[0]
        origin_function = _entry_value(first_trace, "origin_function", None)
        name = str(origin_function or session_id)
        dataset_ref = _dataset_ref_for_id(getattr(record, "dataset_id", None), memory_sources_by_id)
        datasets = [dataset_ref] if dataset_ref else []
        connection_type = derive_connection_type(
            origin_function=origin_function,
            session_id=session_id,
            source="session_trace",
        )
        status = "active" if row.effective_status == "running" else "inactive"
        agent_id = build_agent_connection_id(
            name=name,
            origin_function=origin_function,
            user_id=owner_user_id,
            session_id=session_id,
            dataset_id=datasets[0].id if datasets else None,
            connection_type=connection_type,
        )
        agents.append(
            AgentConnection(
                id=agent_id,
                name=name,
                type=connection_type,
                memory_mode="hybrid" if datasets else "session",
                session_id=session_id,
                user_id=owner_user_id,
                tenant_id=str(getattr(user, "tenant_id", "")) or None,
                datasets=datasets,
                last_active_at=getattr(record, "last_activity_at", None),
                status=status,
                source="session_trace",
                origin_function=str(origin_function) if origin_function else None,
                metadata={
                    "session_status": getattr(record, "status", None),
                    "effective_status": row.effective_status,
                    "trace_count": len(traces),
                    "error_count": getattr(record, "error_count", 0),
                },
            )
        )

    return agents


def _is_visible_registered_agent(
    agent: AgentConnection,
    *,
    visible_user_ids: set[str],
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
    range_key: RangeLiteral = "30d",
    status_filter: Optional[str] = None,
    include_sources: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> AgentsListResponse:
    readable_datasets = await _readable_datasets_for(user)
    memory_sources = _memory_sources_from_datasets(readable_datasets)
    memory_sources_by_id = {source.id: source for source in memory_sources}
    permitted_dataset_ids = [UUIDType(source.id) for source in memory_sources]
    visible_user_ids = await _visible_user_ids(user)
    visible_user_id_strings = {str(user_id) for user_id in visible_user_ids}
    permitted_dataset_id_strings = {source.id for source in memory_sources}
    since = _range_since(range_key)

    registered_agents = [
        agent
        for agent in list_registered_agent_connections()
        if _is_visible_registered_agent(
            agent,
            visible_user_ids=visible_user_id_strings,
            permitted_dataset_ids=permitted_dataset_id_strings,
        )
    ]
    trace_agents = await _trace_agents_for_user(
        user=user,
        visible_user_ids=visible_user_ids,
        permitted_dataset_ids=permitted_dataset_ids,
        memory_sources_by_id=memory_sources_by_id,
        since=since,
    )
    agents = _merge_agents([*registered_agents, *trace_agents])
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
    agent_id: str,
    range_key: RangeLiteral = "all",
) -> Optional[AgentDetailResponse]:
    response = await list_agent_connections(
        user=user,
        range_key=range_key,
        include_sources=True,
        limit=10000,
        offset=0,
    )
    agent = next((item for item in response.agents if item.id == agent_id), None)
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


def register_agent_from_request(user: User, request: RegisterAgentRequest) -> AgentConnection:
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
    return register_agent_connection(
        name=request.name,
        connection_type=request.type,
        memory_mode=request.memory_mode,
        source=request.source,
        origin_function=request.origin_function,
        user_id=str(user.id),
        tenant_id=str(getattr(user, "tenant_id", "")) or None,
        session_id=request.session_id,
        datasets=datasets,
        metadata=request.metadata,
    )
