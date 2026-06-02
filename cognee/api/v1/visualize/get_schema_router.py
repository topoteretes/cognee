"""HTTP router for the schema inventory endpoint.

Exposes ``GET /schema/inventory`` so the SaaS frontend (and any HTTP client)
can retrieve the data-derived schema without going through the Python SDK.
"""

from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def get_schema_router() -> APIRouter:
    router = APIRouter()

    @router.get("/inventory", response_model=List[Any])
    async def schema_inventory(
        dataset_id: Optional[UUID] = Query(default=None),
        samples_per_type: int = Query(default=5, ge=0),
        sort: str = Query(default="count"),
        user: User = Depends(get_authenticated_user),
    ):
        """Return the data-derived schema inventory for a dataset.

        Summarizes the knowledge graph by semantic type: per-type instance
        counts, representative sample names, and the per-pair relationship
        distribution. Wraps the ``get_schema_inventory`` SDK function so it
        is accessible over HTTP.

        Query parameters:
            dataset_id: optional dataset UUID to scope the graph databases.
            samples_per_type: max sample instance names per type (default 5).
            sort: ``"count"`` (default) orders types by descending count.
        """
        from cognee.api.v1.visualize.get_schema_inventory import get_schema_inventory

        try:
            result = await get_schema_inventory(
                dataset=dataset_id,
                samples_per_type=samples_per_type,
                sort=sort,
            )
            return result
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        except Exception as exc:
            logger.error(f"Schema inventory failed: {exc}")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return router
