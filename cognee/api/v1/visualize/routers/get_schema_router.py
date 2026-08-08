"""HTTP router for schema inventory and memory provenance endpoints.

Exposes schema/provenance SDK helpers over HTTP with explicit response schemas
and caller-scoped authorization.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger()


class SchemaInventoryRelationship(BaseModel):
    """Relationship aggregate from one semantic type to another."""

    to_type: str | None = Field(
        default=None,
        description="Target semantic type for this relationship aggregate.",
    )
    relation: str = Field(description="Relationship name.")
    count: int = Field(ge=0, description="Number of matching relationships.")


class SchemaInventoryItem(BaseModel):
    """Per-semantic-type inventory row returned by /schema/inventory."""

    type: str | None = Field(default=None, description="Semantic type name.")
    count: int = Field(ge=0, description="Total number of instances of this type.")
    samples: list[str | None] = Field(
        default_factory=list,
        description="Representative instance names.",
    )
    sample_size: int = Field(ge=0, description="Number of returned samples.")
    relationships: list[SchemaInventoryRelationship] = Field(
        default_factory=list,
        description="Relationship aggregates involving this semantic type.",
    )


class ErrorResponse(BaseModel):
    """Generic API error response."""

    error: str


def get_schema_router() -> APIRouter:
    router = APIRouter()

    @router.get(
        "/inventory",
        response_model=list[SchemaInventoryItem],
        responses={
            403: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
        },
    )
    async def schema_inventory(
        dataset_id: UUID = Query(
            ...,
            description=(
                "Dataset UUID to scope the graph databases. "
                "List your datasets via GET /api/v1/datasets to find it."
            ),
            examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        ),
        samples_per_type: int = Query(default=5, ge=0),
        sort: str = Query(
            default="count",
            description=(
                "Sort order: 'count' (default) orders types by descending instance count; "
                "'none' preserves discovery order. Other values are rejected."
            ),
            examples=["count"],
        ),
        user: User = Depends(get_authenticated_user),
    ) -> list[dict]:
        """Return the data-derived schema inventory for an authorized dataset.

        Summarizes the knowledge graph by semantic type: per-type instance
        counts, representative sample names, and the per-pair relationship
        distribution. Wraps the ``get_schema_inventory`` SDK function so it
        is accessible over HTTP with an OpenAPI response schema.

        Query parameters:
            dataset_id: dataset UUID to scope the graph databases.
            samples_per_type: max sample instance names per type (default 5).
            sort: ``"count"`` (default) orders types by descending count;
                ``"none"`` preserves discovery order.
        """
        send_telemetry(
            "Schema Inventory API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/schema/inventory",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize.get_schema_inventory import get_schema_inventory

        try:
            datasets = await get_authorized_existing_datasets([dataset_id], "read", user)
            if not datasets:
                raise PermissionDeniedError(message="Not authorized to read this dataset")

            return await get_schema_inventory(
                dataset=datasets[0].id,
                samples_per_type=samples_per_type,
                sort=sort,
            )
        except PermissionDeniedError:
            return JSONResponse(
                status_code=403,
                content={"error": "Not authorized to read this dataset"},
            )
        except Exception as exc:
            logger.error("schema inventory failed: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to build schema inventory"},
            )

    @router.get(
        "/provenance",
        response_model=None,
        responses={409: {"model": ErrorResponse}},
    )
    async def schema_provenance(
        include_memory: bool = Query(
            default=False,
            description=(
                "When true, include the extracted memory subgraph "
                "(entities/relationships) in the provenance visualization."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Return a caller-scoped HTML memory-provenance visualization.

        Query parameters:
            include_memory: when true, also folds the extracted memory
                (entities/relationships) into the provenance view alongside
                data lineage (default false).
        """
        send_telemetry(
            "Schema Provenance API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/schema/provenance",
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import visualize_memory_provenance

        tenant_id = getattr(user, "tenant_id", None)
        if tenant_id is not None:
            scope_tenant_ids = [tenant_id]
            scope_user_ids = None
        else:
            scope_tenant_ids = None
            scope_user_ids = [user.id]

        try:
            html = await visualize_memory_provenance(
                include_memory=include_memory,
                scope_tenant_ids=scope_tenant_ids,
                scope_user_ids=scope_user_ids,
            )
            return HTMLResponse(html)
        except Exception as exc:
            logger.error("schema provenance failed: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to build memory provenance"},
            )

    return router
