from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from uuid import UUID
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.models import User

from cognee.context_global_variables import set_database_global_context_variables
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

logger = get_logger()


def get_visualize_router() -> APIRouter:
    router = APIRouter()

    @router.get("", response_model=None)
    async def visualize(dataset_id: UUID, user: User = Depends(get_authenticated_user)):
        """
        Generate an HTML visualization of the dataset's knowledge graph.

        This endpoint creates an interactive HTML visualization of the knowledge graph
        for a specific dataset. The visualization displays nodes and edges representing
        entities and their relationships, allowing users to explore the graph structure
        visually.

        ## Query Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset to visualize

        ## Response
        Returns an HTML page containing the interactive graph visualization.

        ## Error Codes
        - **404 Not Found**: Dataset doesn't exist
        - **403 Forbidden**: User doesn't have permission to read the dataset
        - **500 Internal Server Error**: Error generating visualization

        ## Notes
        - User must have read permissions on the dataset
        - Visualization is interactive and allows graph exploration
        """
        send_telemetry(
            "Visualize API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /v1/visualize",
                "dataset_id": str(dataset_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import visualize_graph

        try:
            # Verify user has permission to read dataset
            dataset = await get_authorized_existing_datasets([dataset_id], "read", user)

            # Will only be used if ENABLE_BACKEND_ACCESS_CONTROL is set to True
            await set_database_global_context_variables(dataset[0].id, dataset[0].owner_id)

            html_visualization = await visualize_graph()
            return HTMLResponse(html_visualization)

        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
