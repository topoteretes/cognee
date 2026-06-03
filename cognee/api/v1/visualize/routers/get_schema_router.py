"""HTTP router for the schema inventory endpoint.

Exposes ``GET /schema/inventory`` so the SaaS frontend (and any HTTP client)
can retrieve the data-derived schema without going through the Python SDK.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class RelationshipDistribution(BaseModel):
    """One ``(relation, target type)`` pair and how often it occurs for a type."""

    to_type: Optional[str] = None
    relation: str
    count: int


class SchemaInventoryItem(BaseModel):
    """Per-type schema inventory record returned by the endpoint."""

    type: str
    count: int
    samples: List[str]
    sample_size: int
    relationships: List[RelationshipDistribution]


def get_schema_router() -> APIRouter:
    router = APIRouter()

    @router.get("/inventory", response_model=List[SchemaInventoryItem])
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
            sort: ``"count"`` (default) orders types by descending count;
                ``"none"`` preserves discovery order.
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
            # Controlled validation messages only (e.g. samples_per_type bounds);
            # never raw internal exception text.
            return JSONResponse(status_code=422, content={"error": str(exc)})
        except Exception:
            # Log the detail server-side; return a generic message so internal
            # exception text / stack info is not exposed to the client (CodeQL).
            logger.exception("Schema inventory failed")
            return JSONResponse(status_code=500, content={"error": "Internal server error"})

    return router
