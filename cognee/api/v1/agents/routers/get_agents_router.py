from typing import Optional
from uuid import UUID

from cognee.api.DTO import OutDTO
from cognee.modules.agents.create_agent import create_agent
from cognee.modules.agents.delete_agent import delete_agent
from cognee.modules.agents.list_agents import list_agents
from cognee.modules.users.methods.get_authenticated_user import get_authenticated_user
from cognee.modules.users.models.User import User
from fastapi import APIRouter, Depends, HTTPException
from fastapi_users.exceptions import UserAlreadyExists


class AgentDTO(OutDTO):
    agent_id: UUID
    agent_email: str


class AgentWithApiKeyDTO(OutDTO):
    agent_id: UUID
    agent_email: str
    agent_api_key: str


def _display_email(internal_email: str) -> str:
    if "+" in internal_email:
        return internal_email.rsplit("+", 1)[0] + "@cognee.agent"
    return internal_email


def get_agents_router() -> APIRouter:
    router = APIRouter()

    @router.post("/")
    async def create_agent_endpoint(
        name: str,
        user: User = Depends(get_authenticated_user),
        password: Optional[str] = None,
    ) -> AgentWithApiKeyDTO:
        try:
            agent_user, api_key = await create_agent(name, user, password)
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

    @router.delete("/{agent_id}")
    async def delete_agent_endpoint(
        agent_id: UUID, user: User = Depends(get_authenticated_user)
    ) -> None:
        try:
            await delete_agent(agent_id, user.id)
        except PermissionError:
            raise HTTPException(status_code=403, detail="Not authorized")

    @router.get("/")
    async def get_agents_endpoint(
        user: User = Depends(get_authenticated_user),
    ) -> list[AgentDTO]:
        agents = await list_agents(user.id)
        return [
            AgentDTO(
                agent_id=agent.id,
                agent_email=_display_email(agent.email),
            )
            for agent in agents
        ]

    return router
