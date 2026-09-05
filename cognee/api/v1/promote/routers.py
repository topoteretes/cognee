"""Authenticated HTTP access to the SDK's promotion operation."""

from dataclasses import asdict
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from cognee.api.v1.promote import promote
from cognee.api.v1.promote.promote import _get_data, get_promotion_source, open_data_file
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
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

    async def authorize_source(user, dataset_id):
        await get_promotion_source(user, dataset_id, "read", "team")
        await get_promotion_source(user, dataset_id, "share", "team")

    @router.get("/sources/{dataset_id}/data")
    async def source_documents(dataset_id: UUID, user: CurrentUser):
        await authorize_source(user, dataset_id)
        async with get_relational_engine().get_async_session() as session:
            rows = (
                await session.execute(
                    select(Data.id, Data.name).where(Data.dataset_id == dataset_id)
                )
            ).all()
        return [{"id": row.id, "name": row.name} for row in rows]

    @router.get("/sources/{dataset_id}/data/{data_id}/raw")
    async def source_document(dataset_id: UUID, data_id: UUID, user: CurrentUser):
        await authorize_source(user, dataset_id)
        row = await _get_data(data_id, dataset_id)
        if row is None:
            raise HTTPException(404, "Document not found in this source dataset")
        async with open_data_file(row.raw_data_location, "rb") as stream:
            content = stream.read(64 * 1024 * 1024 + 1)
        if len(content) > 64 * 1024 * 1024:
            raise HTTPException(413, "Document exceeds the 64 MiB promotion limit")
        return Response(content, media_type="text/plain")

    @router.post("")
    async def promote_document(payload: PromoteRequest, user: CurrentUser):
        if not payload.dry_run and payload.expected_source_revision is None:
            raise HTTPException(400, "Preview the document before confirming promotion")
        try:
            return asdict(await promote(user=user, **payload.model_dump()))
        except ValueError as error:
            raise HTTPException(400, str(error)) from error

    return router
