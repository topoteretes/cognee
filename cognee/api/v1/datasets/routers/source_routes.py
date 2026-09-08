"""Generic metadata discovery/routing and document pages, under /datasets."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from cognee.modules.data.source_catalog import (
    route_sources,
    source_catalog,
    source_document,
    source_documents,
)
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class RouteSources(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    source_hint: str | None = Field(default=None, max_length=500)
    dataset_ids: list[UUID] | None = Field(default=None, max_length=1000)
    exclude_source_ids: list[UUID] | None = Field(default=None, max_length=2048)
    max_sources: int = Field(default=6, ge=1, le=8)
    max_catalog_entries: int = Field(default=512, ge=1, le=2048)


def get_source_routes() -> APIRouter:
    router = APIRouter()

    async def guarded(operation):
        try:
            return await operation
        except PermissionError:
            raise HTTPException(403, "Source or dataset unavailable.") from None
        except Exception:  # API boundary: never expose provider exception details.
            logger.exception("Source discovery failed")
            raise HTTPException(409, "Source discovery failed; no content was searched.") from None

    @router.get("/source-catalog")
    async def catalog(
        user: Annotated[User, Depends(get_authenticated_user)],
        dataset_ids: Annotated[list[UUID] | None, Query()] = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ):
        result = await guarded(source_catalog(user, dataset_ids))
        return {
            "items": result["items"][offset : offset + limit],
            "total": len(result["items"]),
            "complete": result["complete"],
            "next_offset": offset + limit if offset + limit < len(result["items"]) else None,
        }

    @router.post("/source-route")
    async def route(payload: RouteSources, user: Annotated[User, Depends(get_authenticated_user)]):
        return await guarded(route_sources(user, **payload.model_dump()))

    @router.get("/source-documents/{source_id}")
    async def documents(
        source_id: UUID,
        user: Annotated[User, Depends(get_authenticated_user)],
        after: UUID | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ):
        return await guarded(source_documents(user, str(source_id), after, limit))

    @router.get("/source-document/{dataset_id}/{document_id}")
    async def document(
        dataset_id: UUID, document_id: UUID, user: Annotated[User, Depends(get_authenticated_user)]
    ):
        return await guarded(source_document(user, dataset_id, document_id))

    return router
