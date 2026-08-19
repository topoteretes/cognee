from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Free-form on purpose: the set of client types keeps growing (Claude Code,
# Codex, Slack, OpenCode, Cursor, Windsurf, ...) and gating it behind a
# closed Literal meant every new integration needed a backend code change
# just to be recognized. Callers should self-declare their type via
# ``RegisterAgentRequest.type`` at registration; ``derive_connection_type()``
# is only a best-effort fallback for sessions that never registered. This
# list is documentation, not an enum member set — pick a value from it when
# your client matches one, otherwise just use your own lowercase name.
# "unknown" is deliberately NOT in this list: it is the fallback
# derive_connection_type() returns when nothing matched, not a type a
# client should ever self-declare.
KNOWN_AGENT_CONNECTION_TYPES = (
    "sdk",
    "api",
    "mcp",
    "claude_code",
    "codex",
    "slack",
    "opencode",
    "workflow",
)
AgentConnectionType = str
AgentMemoryMode = Literal["session", "cognee", "hybrid", "none", "unknown"]
AgentStatus = Literal["active", "inactive", "unknown"]
AgentSource = Literal["agent_memory", "session_trace", "serve", "api_key", "mcp", "api"]
MemorySourceType = Literal["dataset", "company_brain", "knowledge_wiki", "project_dataset"]


class AgentDatasetRef(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    role: str = "read"
    type: MemorySourceType = "dataset"


class AgentConnection(BaseModel):
    id: str
    agent_session_name: str
    type: AgentConnectionType = "unknown"
    memory_mode: AgentMemoryMode = "unknown"
    session_id: Optional[str] = None
    user_id: Optional[UUID] = None
    tenant_id: Optional[UUID] = None
    datasets: list[AgentDatasetRef] = Field(default_factory=list)
    last_active_at: Optional[datetime] = None
    status: AgentStatus = "unknown"
    source: AgentSource = "session_trace"
    origin_function: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySourceConnection(BaseModel):
    id: str
    name: str
    type: MemorySourceType = "dataset"
    owner_id: Optional[UUID] = None
    tenant_id: Optional[UUID] = None
    status: AgentStatus = "active"
    connected_agent_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentsListResponse(BaseModel):
    agents: list[AgentConnection]
    memory_sources: list[MemorySourceConnection] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    has_more: bool


class AgentDetailResponse(BaseModel):
    agent: AgentConnection
    memory_sources: list[MemorySourceConnection] = Field(default_factory=list)
    recent_sessions: list[dict[str, Any]] = Field(default_factory=list)
    recent_traces: list[dict[str, Any]] = Field(default_factory=list)
    recent_qas: list[dict[str, Any]] = Field(default_factory=list)


class RegisterAgentRequest(BaseModel):
    agent_session_name: str = Field(
        description="A unique name for this agent connection. "
        "Combined with the authenticated user's ID to identify the connection."
    )
    type: AgentConnectionType = "api"
    memory_mode: AgentMemoryMode = "unknown"
    session_id: Optional[str] = None
    dataset_ids: list[str] = Field(default_factory=list)
    dataset_names: list[str] = Field(default_factory=list)
    source: AgentSource = "api"
    origin_function: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnregisterAgentRequest(BaseModel):
    agent_session_name: str = Field(
        description="The name used when registering the connection. "
        "Combined with the authenticated user's ID to identify which connection to deactivate."
    )
