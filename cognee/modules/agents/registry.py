from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from threading import RLock
from typing import Iterable, Optional
from uuid import UUID

from cognee.modules.agents.models import (
    AgentConnection,
    AgentConnectionType,
    AgentDatasetRef,
    AgentMemoryMode,
    AgentSource,
    MemorySourceType,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("agents.registry")

AGENT_CONFIG_NAME = "agent_configuration"

_registered_agent_connections: dict[str, AgentConnection] = {}
_registry_lock = RLock()


def classify_memory_source_type(name: str | None) -> MemorySourceType:
    normalized = (name or "").lower()
    if "brain" in normalized:
        return "company_brain"
    if "wiki" in normalized:
        return "knowledge_wiki"
    if "project" in normalized:
        return "project_dataset"
    return "dataset"


def derive_memory_mode(
    *,
    with_memory: bool = False,
    with_session_memory: bool = False,
    save_session_traces: bool = False,
) -> AgentMemoryMode:
    if with_memory and (with_session_memory or save_session_traces):
        return "hybrid"
    if with_memory:
        return "cognee"
    if with_session_memory or save_session_traces:
        return "session"
    return "none"


def derive_connection_type(
    *,
    origin_function: str | None = None,
    session_id: str | None = None,
    source: str | None = None,
) -> AgentConnectionType:
    source_lower = (source or "").lower()
    if source_lower in {"mcp", "api", "api_key", "serve", "workflow", "sdk"}:
        if source_lower == "serve":
            return "api"
        if source_lower == "api_key":
            return "api"
        return source_lower  # type: ignore[return-value]

    text = f"{origin_function or ''} {session_id or ''}".lower()
    if "claude" in text or "claude_code" in text or text.startswith("cc_"):
        return "claude_code"
    if "mcp" in text:
        return "mcp"
    return "sdk" if origin_function else "unknown"


def build_agent_connection_id(
    *,
    agent_session_name: str,
    user_id: str | None = None,
) -> str:
    identity = f"{agent_session_name}|{user_id or ''}"
    digest = hashlib.blake2s(identity.encode("utf-8"), digest_size=8).hexdigest()
    base = re.sub(r"[^a-zA-Z0-9_-]+", "-", agent_session_name).strip("-")
    base = base[-48:] if len(base) > 48 else base
    return f"{base or 'agent'}-{digest}"


def _normalize_datasets(datasets: Iterable[AgentDatasetRef | dict] | None) -> list[AgentDatasetRef]:
    normalized = []
    for dataset in datasets or []:
        if isinstance(dataset, AgentDatasetRef):
            ref = dataset
        else:
            ref = AgentDatasetRef(**dataset)
        if ref.type == "dataset":
            ref = ref.model_copy(update={"type": classify_memory_source_type(ref.name)})
        normalized.append(ref)
    return normalized


async def _persist_agent_connection(user_id: UUID, connection: AgentConnection) -> None:
    from cognee.modules.users.methods.get_principal_configuration import (
        get_principal_all_configuration,
    )
    from cognee.modules.users.methods.store_principal_configuration import (
        store_principal_configuration,
    )

    all_configs = await get_principal_all_configuration(user_id)
    existing_config = {}
    for config in all_configs:
        if config.get("name") == AGENT_CONFIG_NAME:
            existing_config = config.get("configuration", {})
            break

    agents = existing_config.get("agents", {})
    agents[connection.id] = connection.model_dump(mode="json")

    await store_principal_configuration(
        principal_id=user_id,
        name=AGENT_CONFIG_NAME,
        configuration={**existing_config, "agents": agents},
    )


async def register_agent_connection(
    *,
    agent_session_name: str,
    connection_type: AgentConnectionType = "unknown",
    memory_mode: AgentMemoryMode = "unknown",
    source: AgentSource = "api",
    agent_id: Optional[str] = None,
    origin_function: Optional[str] = None,
    user_id: Optional[UUID] = None,
    tenant_id: Optional[UUID] = None,
    session_id: Optional[str] = None,
    datasets: Iterable[AgentDatasetRef | dict] | None = None,
    status: str = "active",
    last_active_at: Optional[datetime] = None,
    metadata: Optional[dict] = None,
) -> AgentConnection:
    dataset_refs = _normalize_datasets(datasets)
    resolved_agent_id = agent_id or build_agent_connection_id(
        agent_session_name=agent_session_name,
        user_id=str(user_id) if user_id is not None else None,
    )
    connection = AgentConnection(
        id=resolved_agent_id,
        agent_session_name=agent_session_name,
        type=connection_type,
        memory_mode=memory_mode,
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
        datasets=dataset_refs,
        last_active_at=last_active_at or datetime.now(timezone.utc),
        status=status if status in {"active", "inactive", "unknown"} else "unknown",
        source=source,
        origin_function=origin_function,
        metadata=metadata or {},
    )

    with _registry_lock:
        existing = _registered_agent_connections.get(connection.id)
        if existing:
            merged_metadata = {**existing.metadata, **connection.metadata}
            connection = connection.model_copy(update={"metadata": merged_metadata})
        _registered_agent_connections[connection.id] = connection

    if user_id:
        await _persist_agent_connection(user_id, connection)

    return connection


def list_registered_agent_connections() -> list[AgentConnection]:
    with _registry_lock:
        return list(_registered_agent_connections.values())


async def list_persisted_agent_connections(
    user_ids: list[UUID], active_only: bool = True
) -> list[AgentConnection]:
    from cognee.modules.users.methods.get_principal_configuration import (
        get_principal_all_configuration,
    )

    agents: list[AgentConnection] = []
    for user_id in user_ids:
        all_configs = await get_principal_all_configuration(user_id)
        for config in all_configs:
            if config.get("name") == AGENT_CONFIG_NAME:
                agents_dict = config.get("configuration", {}).get("agents", {})
                for data in agents_dict.values():
                    connection = AgentConnection(**data)
                    if not active_only or connection.status == "active":
                        agents.append(connection)
    return agents


def _remove_from_registry(user_id: UUID) -> None:
    with _registry_lock:
        to_remove = [
            key
            for key, conn in _registered_agent_connections.items()
            if conn.user_id is not None and conn.user_id == user_id
        ]
        for key in to_remove:
            del _registered_agent_connections[key]


def _remove_connection_from_registry(connection_id: str) -> None:
    with _registry_lock:
        _registered_agent_connections.pop(connection_id, None)


async def deactivate_agent_connection(user_id: UUID, connection_id: str) -> None:
    _remove_connection_from_registry(connection_id)
    await _deactivate_persisted_connection(user_id, connection_id)


async def delete_user_agent_connections(user_id: UUID) -> None:
    _remove_from_registry(user_id)
    await _delete_persisted_agent_connections(user_id)


async def _deactivate_persisted_connection(user_id: UUID, connection_id: str) -> None:
    from cognee.modules.users.methods.get_principal_configuration import (
        get_principal_all_configuration,
    )
    from cognee.modules.users.methods.store_principal_configuration import (
        store_principal_configuration,
    )

    all_configs = await get_principal_all_configuration(user_id)
    for config in all_configs:
        if config.get("name") == AGENT_CONFIG_NAME:
            existing_config = config.get("configuration", {})
            agents = existing_config.get("agents", {})
            if connection_id in agents:
                agents[connection_id]["status"] = "inactive"
                await store_principal_configuration(
                    principal_id=user_id,
                    name=AGENT_CONFIG_NAME,
                    configuration={**existing_config, "agents": agents},
                )
            return


async def _delete_persisted_agent_connections(user_id: UUID) -> None:
    from cognee.modules.users.methods.get_principal_configuration import (
        get_principal_all_configuration,
    )
    from cognee.modules.users.methods.store_principal_configuration import (
        store_principal_configuration,
    )

    all_configs = await get_principal_all_configuration(user_id)
    for config in all_configs:
        if config.get("name") == AGENT_CONFIG_NAME:
            existing_config = config.get("configuration", {})
            existing_config.pop("agents", None)
            await store_principal_configuration(
                principal_id=user_id,
                name=AGENT_CONFIG_NAME,
                configuration=existing_config,
            )
            return


def clear_registered_agent_connections() -> None:
    with _registry_lock:
        _registered_agent_connections.clear()
