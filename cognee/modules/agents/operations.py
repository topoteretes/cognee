from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional
from uuid import UUID as UUIDType

from sqlalchemy import select

from cognee.exceptions import CogneeValidationError
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
    classify_memory_source_type,
    list_persisted_agent_connections,
    list_registered_agent_connections,
    register_agent_connection,
)
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)
from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset
from cognee.modules.data.methods.get_authorized_dataset_by_name import (
    get_authorized_dataset_by_name,
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
                owner_id=getattr(dataset, "owner_id", None),
                tenant_id=getattr(dataset, "tenant_id", None),
                status="active",
            )
        )
    return sources


def _visible_registered_agent(
    agent: AgentConnection,
    *,
    visible_user_ids: set[UUIDType],
    permitted_dataset_ids: set[str],
) -> AgentConnection | None:
    if agent.user_id and agent.user_id in visible_user_ids:
        return agent

    visible_datasets = [
        dataset
        for dataset in agent.datasets
        if dataset.id is not None and dataset.id in permitted_dataset_ids
    ]
    if visible_datasets:
        # A dataset grant exposes only that dataset's view of the connection. Do not
        # disclose or later hydrate session data from the connection's other datasets.
        return agent.model_copy(update={"datasets": visible_datasets})
    if agent.user_id is None and not agent.datasets:
        return agent
    return None


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
    readable_datasets = await _readable_datasets_for(user)
    memory_sources = _memory_sources_from_datasets(readable_datasets)
    visible_user_ids = await _visible_user_ids(user)
    visible_user_id_set = set(visible_user_ids)
    permitted_dataset_id_strings = {source.id for source in memory_sources}

    registered_agents = []
    for registered_agent in list_registered_agent_connections():
        visible_agent = _visible_registered_agent(
            registered_agent,
            visible_user_ids=visible_user_id_set,
            permitted_dataset_ids=permitted_dataset_id_strings,
        )
        if visible_agent is not None:
            registered_agents.append(visible_agent)
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
    response = await list_agent_connections(
        user=user,
        agent_id=agent_id,
        include_sources=True,
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
            from cognee.modules.session_lifecycle.metrics import get_session_row

            readable_dataset_ids = {
                str(dataset.id) for dataset in await _readable_datasets_for(user)
            }
            referenced_dataset_ids = {
                dataset.id for dataset in agent.datasets if dataset.id is not None
            }
            dataset_ids = referenced_dataset_ids.intersection(readable_dataset_ids)
            # A session read must identify one dataset partition. An unscoped connection
            # can still read its legacy owner-only session. Dataset-backed connections
            # require one currently-readable scope; stored refs alone are never authority.
            if referenced_dataset_ids and len(dataset_ids) != 1:
                raise CogneeValidationError(
                    "Agent session has no single currently-readable dataset scope.",
                    log=False,
                )
            dataset_id = next(iter(dataset_ids), None)
            session_owner_id = str(agent.user_id)
            if dataset_id is not None:
                dataset_uuid = UUIDType(dataset_id)
                row = await get_session_row(
                    session_id=agent.session_id,
                    user_id=user.id,
                    permitted_dataset_ids=[dataset_uuid],
                    dataset_id=dataset_uuid,
                    owner_user_id=agent.user_id,
                )
                if row is None:
                    raise CogneeValidationError(
                        "Agent session is not visible in the authorized dataset scope.",
                        log=False,
                    )
            session_manager = get_session_manager(dataset_id=dataset_id)
            qas = await session_manager.get_session(
                user_id=session_owner_id,
                session_id=agent.session_id,
                formatted=False,
            )
            traces = await session_manager.get_agent_trace_session(
                user_id=session_owner_id,
                session_id=agent.session_id,
                last_n=20,
            )
            recent_qas = (
                [_entry_to_dict(entry) for entry in qas[-20:]] if isinstance(qas, list) else []
            )
            recent_traces = [_entry_to_dict(entry) for entry in traces[-20:]]
            recent_sessions = [
                {
                    "session_id": agent.session_id,
                    "user_id": session_owner_id,
                    "dataset_id": dataset_id,
                }
            ]
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
    datasets: list[AgentDatasetRef] = []
    seen_dataset_ids: set[str] = set()
    for requested_id in request.dataset_ids:
        try:
            parsed_id = UUIDType(requested_id)
        except (TypeError, ValueError) as error:
            raise CogneeValidationError(
                f"Invalid dataset id: {requested_id!r}.", log=False
            ) from error
        dataset = await get_authorized_dataset(user, parsed_id, "write")
        if dataset is None:
            raise CogneeValidationError(
                f"Dataset {requested_id!r} was not found or is not writable.", log=False
            )
        canonical_id = str(dataset.id)
        if canonical_id not in seen_dataset_ids:
            seen_dataset_ids.add(canonical_id)
            datasets.append(
                AgentDatasetRef(
                    id=canonical_id,
                    name=dataset.name,
                    role="read_write",
                    type=classify_memory_source_type(dataset.name),
                )
            )

    for requested_name in request.dataset_names:
        dataset = await get_authorized_dataset_by_name(requested_name, user, "write")
        if dataset is None:
            raise CogneeValidationError(
                f"Dataset {requested_name!r} was not found or is not writable.", log=False
            )
        canonical_id = str(dataset.id)
        if canonical_id not in seen_dataset_ids:
            seen_dataset_ids.add(canonical_id)
            datasets.append(
                AgentDatasetRef(
                    id=canonical_id,
                    name=dataset.name,
                    role="read_write",
                    type=classify_memory_source_type(dataset.name),
                )
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
