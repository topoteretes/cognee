from typing import Literal, Optional
from uuid import UUID

from cognee.api.DTO import OutDTO
from cognee.modules.agents.agent_mode import register_agent, unregister_agent
from cognee.modules.agents.models import RegisterAgentRequest, UnregisterAgentRequest
from cognee.modules.agents.create_agent import create_agent
from cognee.modules.agents.get_agent import get_agent
from cognee.modules.agents.list_agents import list_agents
from cognee.modules.agents.delete_agent import delete_agent
from cognee.modules.agents.operations import (
    get_agent_connection_detail,
    list_agent_connections,
)
from cognee.modules.users.methods.get_authenticated_user import get_authenticated_user
from cognee.modules.users.models.User import User
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from fastapi_users.exceptions import UserAlreadyExists

RangeLiteral = Literal["24h", "7d", "30d", "all"]

CONNECTIONS_TAG = "agent connections"
MANAGEMENT_TAG = "agent management"


class AgentDTO(OutDTO):
    agent_id: UUID
    agent_email: str
    api_key_label: Optional[str] = None


class AgentWithApiKeyDTO(OutDTO):
    agent_id: UUID
    agent_email: str
    agent_api_key: str


class AgentModeDTO(OutDTO):
    active_agents: int


def _display_email(internal_email: str) -> str:
    if "+" in internal_email:
        return internal_email.rsplit("+", 1)[0] + "@cognee.agent"
    return internal_email


def get_agents_router() -> APIRouter:
    router = APIRouter()

    # ------------------------------------------------------------------ #
    # Agent management (fixed paths first)
    # ------------------------------------------------------------------ #

    @router.get("/list", tags=[MANAGEMENT_TAG])
    async def list_agents_endpoint(
        user: User = Depends(get_authenticated_user),
    ) -> list[AgentDTO]:
        """List agents endpoint — GET /api/v1/agents/list."""
        agents = await list_agents(user.id)
        return [
            AgentDTO(
                agent_id=agent.user.id,
                agent_email=_display_email(agent.user.email),
                api_key_label=agent.api_key_label,
            )
            for agent in agents
        ]

    @router.post("/create", tags=[MANAGEMENT_TAG])
    async def create_agent_endpoint(
        name: str,
        user: User = Depends(get_authenticated_user),
    ) -> AgentWithApiKeyDTO:
        """Create agent endpoint — POST /api/v1/agents/create.

        ## Query Parameters
        - **name** (str): Unique name for the new agent user; a conflict is returned if an
          agent with this name exists.
        """
        try:
            agent_user, api_key = await create_agent(name, user)
        except UserAlreadyExists:
            raise HTTPException(
                status_code=409,
                detail=f"Agent with name '{name}' already exists",
            )

        return AgentWithApiKeyDTO(
            agent_id=agent_user.id,
            agent_email=_display_email(agent_user.email),
            agent_api_key=api_key,
        )

    # ------------------------------------------------------------------ #
    # Agent connections (fixed paths)
    # ------------------------------------------------------------------ #

    @router.get("/connections", tags=[CONNECTIONS_TAG])
    async def list_agents_connections(
        agent_id: Optional[UUID] = Query(
            None,
            description="Filter connections by agent user ID. "
            "Only returns connections belonging to this specific agent.",
        ),
        range: RangeLiteral = Query("30d"),
        status_filter: Optional[Literal["active", "inactive", "unknown"]] = Query(
            None,
            alias="status",
        ),
        include_sources: bool = Query(True),
        active_only: bool = Query(True),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        user: User = Depends(get_authenticated_user),
    ):
        """List agents connections — GET /api/v1/agents/connections.

        ## Query Parameters
        - **active_only** (bool): When true, restricts results to connections currently
          considered active. Defaults to True.
        - **agent_id** (Optional[UUID]): Filter connections by agent user ID. Only returns
          connections belonging to this specific agent.
        - **include_sources** (bool): When true, includes the source breakdown for each
          returned connection. Defaults to True.
        - **limit** (int): Maximum number of rows to return. Defaults to 50.
        - **offset** (int): Number of rows to skip for pagination. Defaults to 0.
        - **range** (Literal['24h', '7d', '30d', 'all']): One of: '24h', '7d', '30d', 'all'.
          Defaults to '30d'.
        - **status** (Optional[Literal['active', 'inactive', 'unknown']]): One of: 'active',
          'inactive', 'unknown'.
        """
        response = await list_agent_connections(
            user=user,
            agent_id=agent_id,
            range_key=range,
            status_filter=status_filter,
            include_sources=include_sources,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )
        return jsonable_encoder(response)

    @router.get("/connections/me", tags=[CONNECTIONS_TAG])
    async def get_my_connection_detail(
        agent_session_name: Optional[str] = Query(
            None,
            description="Filter by connection name. "
            "Uses the authenticated user's ID as the agent ID.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Get my connection detail — GET /api/v1/agents/connections/me.

        ## Query Parameters
        - **agent_session_name** (Optional[str]): Filter by connection name. Uses the authenticated
          user's ID as the agent ID.
        """
        response = await get_agent_connection_detail(
            user=user,
            agent_id=user.id,
            agent_session_name=agent_session_name,
        )
        if response is None:
            raise HTTPException(status_code=404, detail="connection not found")
        return jsonable_encoder(response)

    @router.get("/connections/{agent_id}", tags=[CONNECTIONS_TAG])
    async def get_connection_detail(
        agent_id: UUID,
        agent_session_name: Optional[str] = Query(
            None,
            description="Filter by connection name within the agent's connections.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Get connection detail — GET /api/v1/agents/connections/{agent_id}.

        ## Path Parameters
        - **agent_id** (UUID): The agent's user ID (from GET /api/v1/agents/list).

        ## Query Parameters
        - **agent_session_name** (Optional[str]): Filter by connection name within the agent's
          connections.
        """
        response = await get_agent_connection_detail(
            user=user,
            agent_id=agent_id,
            agent_session_name=agent_session_name,
        )
        if response is None:
            raise HTTPException(status_code=404, detail="connection not found")
        return jsonable_encoder(response)

    @router.post("/register", status_code=status.HTTP_201_CREATED, tags=[CONNECTIONS_TAG])
    async def register_agent_endpoint(
        request: RegisterAgentRequest,
        user: User = Depends(get_authenticated_user),
    ):
        """Register agent endpoint — POST /api/v1/agents/register.

        ## Request Parameters
        - **agent_session_name** (str): A unique name for this agent connection. Combined with the
          authenticated user's ID to identify the connection.
        - **dataset_ids** (List[str]): UUIDs of the datasets (from GET /api/v1/datasets).
        - **dataset_names** (List[str]): Names of the datasets this agent connection reads
          from and writes to.
        - **memory_mode** (Literal['session', 'cognee', 'hybrid', 'none', 'unknown']): One of:
          'session', 'cognee', 'hybrid', 'none', 'unknown'. Defaults to 'unknown'.
        - **metadata** (Dict[str, Any]): Free-form metadata object.
        - **origin_function** (Optional[str]): Name of the calling function or tool that
          triggered the registration, stored on the connection.
        - **session_id** (Optional[str]): Client-supplied session identifier — the same value passed
          as session_id to POST /api/v1/remember.
        - **source** (Literal['agent_memory', 'session_trace', 'serve', 'api_key', 'mcp', 'api']):
          One of: 'agent_memory', 'session_trace', 'serve', 'api_key', 'mcp', 'api'. Defaults to
          'api'.
        - **type** (str): Connection type label recorded for the registered agent. Defaults
          to 'api'.
        """
        connection = await register_agent(user, request)
        return jsonable_encoder(connection)

    @router.post("/unregister", tags=[CONNECTIONS_TAG])
    async def unregister_agent_endpoint(
        request: UnregisterAgentRequest,
        user: User = Depends(get_authenticated_user),
    ) -> AgentModeDTO:
        """Unregister agent endpoint — POST /api/v1/agents/unregister.

        ## Request Parameters
        - **agent_session_name** (str): The name used when registering the connection. Combined with
          the authenticated user's ID to identify which connection to deactivate.
        """
        count = await unregister_agent(user, request)
        return AgentModeDTO(active_agents=count)

    # ------------------------------------------------------------------ #
    # Path-parameter routes MUST come last
    # ------------------------------------------------------------------ #

    @router.get("/{agent_id}", tags=[MANAGEMENT_TAG])
    async def get_agent_endpoint(
        agent_id: UUID,
        user: User = Depends(get_authenticated_user),
    ) -> AgentDTO:
        """Get agent endpoint — GET /api/v1/agents/{agent_id}.

        ## Path Parameters
        - **agent_id** (UUID): The agent's user ID (from GET /api/v1/agents/list).
        """
        try:
            agent_info = await get_agent(agent_id, user.id)
        except LookupError:
            raise HTTPException(status_code=404, detail="Agent not found")
        except PermissionError:
            raise HTTPException(status_code=403, detail="Not authorized")
        return AgentDTO(
            agent_id=agent_info.user.id,
            agent_email=_display_email(agent_info.user.email),
            api_key_label=agent_info.api_key_label,
        )

    @router.delete("/{agent_id}", tags=[MANAGEMENT_TAG])
    async def delete_agent_endpoint(
        agent_id: UUID, user: User = Depends(get_authenticated_user)
    ) -> None:
        """Delete agent endpoint — DELETE /api/v1/agents/{agent_id}.

        ## Path Parameters
        - **agent_id** (UUID): The agent's user ID (from GET /api/v1/agents/list).
        """
        try:
            await delete_agent(agent_id, user.id)
        except LookupError:
            raise HTTPException(status_code=404, detail="Agent not found")
        except PermissionError:
            raise HTTPException(status_code=403, detail="Not authorized")

    return router
