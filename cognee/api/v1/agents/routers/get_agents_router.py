from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder

from cognee.modules.agents.models import RegisterAgentRequest
from cognee.modules.agents.operations import (
    get_agent_connection_detail,
    list_agent_connections,
    register_agent_from_request,
)
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User

RangeLiteral = Literal["24h", "7d", "30d", "all"]


def get_agents_router() -> APIRouter:
    router = APIRouter()

    @router.get("")
    async def list_agents(
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
        """List agents and memory sources visible to the current user.

        The endpoint combines live in-process registrations from SDK/API callers with
        session-trace fallback data, then attaches readable datasets as memory sources.
        """
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
    async def register_agent(
        request: RegisterAgentRequest,
        user: User = Depends(get_authenticated_user),
    ):
        """Register a live agent/API/MCP/workflow connection with this API process.

        This is intentionally non-durable until a migration-backed connection table is
        added. Session traces still provide durable fallback after agents execute.
        """
        connection = register_agent_from_request(user, request)
        return jsonable_encoder(connection)

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

    return router
