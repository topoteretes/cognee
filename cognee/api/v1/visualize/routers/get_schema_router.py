"""HTTP router for schema inventory and memory provenance endpoints.

Exposes schema/provenance SDK helpers over HTTP with explicit response schemas
and caller-scoped authorization.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee.exceptions import CogneeApiError
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


async def _provenance_scope(user: User) -> tuple[list | None, list | None, list | None]:
    """Decide what of the provenance graph this caller is entitled to see.

    Returns ``(scope_tenant_ids, scope_user_ids, scope_dataset_ids)``.

    A tenant scope alone answers "what exists in this workspace", which is the
    right answer only for whoever administers it. For every other member it
    named datasets (and their files, the ACL edges around them, and the agents
    and sessions that worked on them) that the member holds no read grant on,
    while `/datasets` and `/visualize/brains` next to it answered the ACL-scoped
    question, and named roles and users the member holds no relation to at
    all, which `/permissions/tenants/{id}/roles` and `.../users` already
    withhold from a non-administrator. Administrators keep the whole-workspace
    view the governance UI is built on; everyone else is narrowed to their
    readable datasets and everything ``get_memory_provenance_graph`` derives
    from them.
    """
    from cognee.modules.users.permissions.methods import get_all_user_permission_datasets

    tenant_id = getattr(user, "tenant_id", None)
    if tenant_id is None:
        # No tenant: the OSS/single-user path, where "the caller" is the whole
        # scope and ownership is already the filter.
        return None, [user.id], None

    if await _administers_tenant(user, tenant_id):
        return [tenant_id], None, None

    readable = await get_all_user_permission_datasets(user, "read")
    return [tenant_id], None, [dataset.id for dataset in readable]


async def _administers_tenant(user: User, tenant_id) -> bool:
    """Whether this caller administers the tenant they belong to.

    Defers to ``has_user_management_permission``, the same check the rest of
    the read-side API (``get_tenant_roles``, ``get_users_in_tenant``, ...)
    already uses to decide who sees a whole tenant versus their own slice of
    it: its owner, plus the role names in ``USER_MANAGEMENT_ALLOWED_ROLE_NAMES``.
    (A handful of tenant-mutation call sites still check ``tenant.owner_id``
    directly rather than this helper; that is a narrower, pre-existing split
    on the write side, not something this endpoint needs to resolve.) That
    helper signals "no" by raising — pass ``log_level="DEBUG"`` since a plain
    member failing this check is the expected common case for a read-only
    view, not a denial worth an ERROR log line — and a tenant row that has
    gone missing is likewise treated as "not an administrator", the narrower
    of the two answers.
    """
    from cognee.modules.users.exceptions import PermissionDeniedError, TenantNotFoundError
    from cognee.modules.users.permissions.methods import has_user_management_permission

    try:
        return await has_user_management_permission(
            requester_id=user.id, tenant_id=tenant_id, log_level="DEBUG"
        )
    except (PermissionDeniedError, TenantNotFoundError):
        return False


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
            user,
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
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("schema inventory failed")
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

        A tenant administrator sees the whole workspace; any other member sees
        only the datasets they hold a read grant on, and the files, grants,
        agents and sessions hanging off them (see ``_provenance_scope``).

        Query parameters:
            include_memory: when true, also folds the extracted memory
                (entities/relationships) into the provenance view alongside
                data lineage (default false).
        """
        send_telemetry(
            "Schema Provenance API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/schema/provenance",
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import visualize_memory_provenance

        try:
            scope_tenant_ids, scope_user_ids, scope_dataset_ids = await _provenance_scope(user)
            html = await visualize_memory_provenance(
                include_memory=include_memory,
                scope_tenant_ids=scope_tenant_ids,
                scope_user_ids=scope_user_ids,
                scope_dataset_ids=scope_dataset_ids,
            )
            return HTMLResponse(html)
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("schema provenance failed")
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to build memory provenance"},
            )

    @router.get(
        "/provenance/json",
        response_model=None,
        responses={409: {"model": ErrorResponse}},
    )
    async def schema_provenance_json(
        include_memory: bool = Query(
            default=False,
            description=(
                "When true, include the extracted memory subgraph "
                "(entities/relationships) in the provenance payload."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Return a caller-scoped memory-provenance graph as a JSON-safe dict.

        Same scoping as `GET /schema/provenance` (the whole tenant for its
        administrators, the caller's readable datasets for any other member,
        just the caller when there is no tenant) and the same underlying graph —
        packaged as a dict instead of an HTML page.

        Query parameters:
            include_memory: when true, also folds the extracted memory
                (entities/relationships) into the payload alongside data
                lineage (default false).
        """
        send_telemetry(
            "Schema Provenance JSON API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/schema/provenance/json",
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import get_memory_provenance_payload

        try:
            scope_tenant_ids, scope_user_ids, scope_dataset_ids = await _provenance_scope(user)
            payload = await get_memory_provenance_payload(
                include_memory=include_memory,
                scope_tenant_ids=scope_tenant_ids,
                scope_user_ids=scope_user_ids,
                scope_dataset_ids=scope_dataset_ids,
            )
            return JSONResponse(status_code=200, content=payload)
        except CogneeApiError:
            raise
        except Exception:
            logger.exception("schema provenance json failed")
            return JSONResponse(
                status_code=409,
                content={"error": "Failed to build memory provenance"},
            )

    return router
