from typing import Literal, Optional
from uuid import UUID

from cognee.api.DTO import OutDTO
from cognee.api.v1.agents.agent_mode import register_agent, unregister_agent
from cognee.modules.agents.create_agent import create_agent
from cognee.modules.agents.delete_agent import delete_agent
from cognee.modules.agents.models import RegisterAgentRequest
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

    @router.get("")
    async def list_agents_connections(
        range: RangeLiteral = Query("30d"),
        status_filter: Optional[Literal["active", "inactive", "unknown"]] = Query(
            None,
            alias="status",
        ),
        include_sources: bool = Query(True),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        user: User = Depends(get_authenticated_user),
    ):
        response = await list_agent_connections(
            user=user,
            range_key=range,
            status_filter=status_filter,
            include_sources=include_sources,
            limit=limit,
            offset=offset,
        )
        return jsonable_encoder(response)

    @router.post("/register", status_code=status.HTTP_201_CREATED)
    async def register_agent_endpoint(
        request: RegisterAgentRequest,
        user: User = Depends(get_authenticated_user),
    ):
        connection = await register_agent(user, request)
        return jsonable_encoder(connection)

    @router.post("/create")
    async def create_agent_endpoint(
        name: str,
        user: User = Depends(get_authenticated_user),
    ) -> AgentWithApiKeyDTO:
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

    @router.get("/list")
    async def get_agents_endpoint(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        user: User = Depends(get_authenticated_user),
    ):
        response = await list_agent_connections(
            user=user,
            limit=limit,
            offset=offset,
        )
        return jsonable_encoder(response)

    @router.post("/unregister")
    async def unregister_agent_endpoint(
        user: User = Depends(get_authenticated_user),
    ) -> AgentModeDTO:
        count = unregister_agent()
        return AgentModeDTO(active_agents=count)

    @router.get("/{agent_id}")
    async def get_agent(
        agent_id: str,
        range: RangeLiteral = Query("all"),
        user: User = Depends(get_authenticated_user),
    ):
        response = await get_agent_connection_detail(
            user=user,
            agent_id=agent_id,
            range_key=range,
        )
        if response is None:
            raise HTTPException(status_code=404, detail="agent not found")
        return jsonable_encoder(response)

    @router.delete("/{agent_id}")
    async def delete_agent_endpoint(
        agent_id: UUID, user: User = Depends(get_authenticated_user)
    ) -> None:
        try:
            await delete_agent(agent_id, user.id)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Not authorized")

    return router
