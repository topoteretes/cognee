"""Authenticated HTTP access to the SDK's promotion operation."""

from dataclasses import asdict
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cognee.api.v1.promote import promote
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User

CurrentUser = Annotated[User, Depends(get_authenticated_user)]


class PromoteRequest(BaseModel):
    data_id: UUID
    source_dataset_id: UUID
    target_dataset_id: UUID
    level: Literal["user", "team"]
    reason: str = Field(min_length=1, max_length=2000)
    dry_run: bool = True
    expected_source_revision: str | None = Field(default=None, pattern="^[a-f0-9]{64}$")


def get_promote_router() -> APIRouter:
    router = APIRouter()

    @router.post("")
    async def promote_document(payload: PromoteRequest, user: CurrentUser):
        if not payload.dry_run and payload.expected_source_revision is None:
            raise HTTPException(400, "Preview the document before confirming promotion")
        try:
            return asdict(await promote(user=user, **payload.model_dump()))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    return router
