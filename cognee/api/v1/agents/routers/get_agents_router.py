from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from fastapi_users.exceptions import UserAlreadyExists

from cognee.api.DTO import OutDTO
from cognee.modules.agents.agent_mode import register_agent, unregister_agent
from cognee.modules.agents.create_agent import create_agent
from cognee.modules.agents.delete_agent import delete_agent
from cognee.modules.agents.get_agent import get_agent
from cognee.modules.agents.list_agents import list_agents
from cognee.modules.agents.models import RegisterAgentRequest, UnregisterAgentRequest
from cognee.modules.agents.operations import (
    get_agent_connection_detail,
    list_agent_connections,
)
from cognee.modules.users.methods.get_authenticated_user import get_authenticated_user
from cognee.modules.users.models.User import User

RangeLiteral = Literal["24h", "7d", "30d", "all"]

CONNECTIONS_TAG = "agent connections"
MANAGEMENT_TAG = "agent management"


class AgentDTO(OutDTO):
    agent_id: UUID
    agent_email: str
    api_key_label: str | None = None


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
        """List the agents owned by the authenticated user.

        Each entry carries the agent's user id, its display email and the label of its API
        key, which is null when the agent has no key. Returns an empty list when the user
        owns no agents.
        """
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
        """Create an agent sub-user and issue its API key.

        Returns 409 when an agent with this name already exists for the caller. The agent
        inherits the caller's tenant, and a caller that is itself an agent cannot create
        agents. The response contains the agent id, its display email and the raw API key,
        which is shown only here.

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
        agent_id: UUID | None = Query(
            None,
            description="Filter connections by agent user ID. "
            "Only returns connections belonging to this specific agent.",
        ),
        range: RangeLiteral = Query("30d"),
        status_filter: Literal["active", "inactive", "unknown"] | None = Query(
            None,
            alias="status",
        ),
        include_sources: bool = Query(True),
        active_only: bool = Query(True),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        user: User = Depends(get_authenticated_user),
    ):
        """List agent connections visible to the caller.

        Only connections belonging to the caller or to the caller's own agents are returned;
        sharing a dataset does not expose another user's connections. The body holds the
        page of agents, the memory sources derived from readable datasets (empty when
        include_sources is false), and total, limit, offset and has_more.

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
        agent_session_name: str | None = Query(
            None,
            description="Filter by connection name. A named lookup also returns inactive "
            "(unregistered) connections; check agent.status. "
            "Uses the authenticated user's ID as the agent ID.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Show the caller's own agent connection with its recent activity.

        Returns 404 when no connection matches; without a name only active connections are
        considered, while a named lookup also finds unregistered ones. The body has the
        agent, the memory sources, and the last 20 recent sessions, traces and
        question/answer entries, which stay empty when the session cache cannot be read.

        ## Query Parameters
        - **agent_session_name** (Optional[str]): Filter by connection name. A named lookup also
          returns inactive (unregistered) connections; check agent.status. Uses the authenticated
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
        agent_session_name: str | None = Query(
            None,
            description="Filter by connection name within the agent's connections. "
            "A named lookup also returns inactive (unregistered) connections; "
            "check agent.status.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Show one agent's connection with its recent activity.

        Returns 404 when no connection matches the agent id, including when the agent is not
        visible to the caller; without a name only active connections are considered, while
        a named lookup also finds unregistered ones. The body has the agent, the memory
        sources, and the last 20 recent sessions, traces and question/answer entries.

        ## Path Parameters
        - **agent_id** (UUID): The agent's user ID (from GET /api/v1/agents/list).

        ## Query Parameters
        - **agent_session_name** (Optional[str]): Filter by connection name within the agent's
          connections. A named lookup also returns inactive (unregistered) connections; check
          agent.status.
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
        """Register an agent connection for the authenticated user.

        Responds 201 with the stored connection. The connection is identified by the session
        name together with the caller's user id, so re-registering an already active one
        returns it again without raising the active connection count; the first registration
        in agent mode also starts the background watchdog.

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
        """Deactivate a registered agent connection.

        The connection is identified by the session name together with the caller's user id
        and is marked inactive. The response holds the number of connections still active;
        unregistering one that is unknown or already inactive leaves that count unchanged.

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
        """Fetch one of your agents by id.

        Returns 404 when no user with this id exists and 403 when the user is not an agent
        owned by the caller. The body carries the agent id, its display email and the label
        of its API key, which is null when the agent has no key.

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
        """Delete an agent and its registered connections.

        Returns 404 when no user with this id exists and 403 when the agent is not owned by
        the caller. The agent's connection records are removed first, then the agent user
        itself; on success the response body is JSON null.

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
