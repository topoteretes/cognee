from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from uuid import UUID
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_authenticated_user, get_user
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.models import User

from cognee.context_global_variables import set_database_global_context_variables
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version

logger = get_logger()


class UserDatasetPair(BaseModel):
    user_id: UUID
    dataset_id: UUID


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

    @router.post("/multi", response_model=None)
    async def visualize_multi(
        pairs: List[UserDatasetPair],
        user: User = Depends(get_authenticated_user),
    ):
        """
        Generate a combined HTML visualization of graph data from multiple users' datasets.

        This endpoint aggregates knowledge graphs from multiple user+dataset pairs
        into a single interactive visualization, with each user's nodes tagged for
        color-by-user rendering.

        ## Request Body
        A JSON array of objects, each with:
        - **user_id** (UUID): The user who owns the dataset
        - **dataset_id** (UUID): The dataset to include

        ## Response
        Returns an HTML page containing the combined interactive graph visualization.

        ## Notes
        - Requires superuser privileges to view other users' data
        - Each user+dataset pair must exist and be accessible
        """
        send_telemetry(
            "Visualize Multi API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/visualize/multi",
                "pair_count": len(pairs),
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.visualize import visualize_multi_user_graph

        try:
            if not user.is_superuser:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Superuser privileges required for multi-user visualization"},
                )

            user_dataset_pairs = []
            for pair in pairs:
                target_user = await get_user(pair.user_id)
                datasets = await get_authorized_existing_datasets(
                    [pair.dataset_id], "read", target_user
                )
                user_dataset_pairs.append((target_user, datasets[0]))

            html_visualization = await visualize_multi_user_graph(user_dataset_pairs)
            return HTMLResponse(html_visualization)

        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
