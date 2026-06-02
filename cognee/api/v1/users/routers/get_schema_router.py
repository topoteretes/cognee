from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, JSONResponse

from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

logger = get_logger()


def get_schema_router() -> APIRouter:
    router = APIRouter()

    @router.get("/inventory", response_model=None)
    async def schema_inventory(
        dataset_id: UUID,
        samples_per_type: int = 5,
        sort: str = "count",
        user: User = Depends(get_authenticated_user),
    ):
        """
        Return a data-derived inventory of what the dataset's knowledge graph
        actually contains, summarized by semantic type.

        ## Query Parameters
        - **dataset_id** (UUID): The dataset to inventory.
        - **samples_per_type** (int): Max sample instance names per type (default 5).
        - **sort** (str): ``count`` orders types by descending count then name.

        ## Response
        A JSON array of ``{type, count, samples, sample_size, relationships}``,
        where each ``relationships`` entry is ``{to_type, relation, count}``.

        ## Error Codes
        - **403 Forbidden**: User lacks read permission on the dataset.
        - **409 Conflict**: Error building the inventory.
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

        from cognee.api.v1.visualize import get_schema_inventory

        try:
            # Verify the user has read permission; raises if not authorized.
            dataset = await get_authorized_existing_datasets([dataset_id], "read", user)
            # get_schema_inventory self-scopes the graph databases to the dataset
            # owner via set_database_global_context_variables.
            inventory = await get_schema_inventory(
                dataset[0].id, samples_per_type=samples_per_type, sort=sort
            )
            return JSONResponse(content=inventory)

        except PermissionDeniedError:
            return JSONResponse(
                status_code=403,
                content={"error": "Not authorized to read this dataset"},
            )
        except Exception as error:
            # Log the detail server-side; don't leak raw internals to the client.
            logger.error("schema inventory failed: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409, content={"error": "Failed to build schema inventory"}
            )

    @router.get("/provenance", response_model=None)
    async def schema_provenance(
        include_memory: bool = False,
        user: User = Depends(get_authenticated_user),
    ):
        """
        Return an HTML visualization of the memory-provenance graph
        (Tenant -> User -> Agent -> Brain -> File -> memory), projected from the
        relational database. Works without the graph backend or an LLM.

        ## Query Parameters
        - **include_memory** (bool): Fold in extracted memory linked to source files.

        ## Response
        A self-contained interactive HTML page.
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

        # Scope the provenance graph to the caller so it never leaks other
        # tenants'/users' actors, datasets or files. Prefer the caller's tenant
        # (multi-tenant SaaS); fall back to the caller's own user id when there
        # is no tenant (single-user/OSS). Never call it unscoped from HTTP.
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

        except Exception as error:
            logger.error("schema provenance failed: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409, content={"error": "Failed to build memory provenance"}
            )

    return router
