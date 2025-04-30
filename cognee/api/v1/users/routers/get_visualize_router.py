from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def get_visualize_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_model=None)
    async def visualize():
        """This endpoint is responsible for adding data to the graph."""
        from cognee.api.v1.visualize import visualize_graph

        try:
            html_visualization = await visualize_graph()
            return HTMLResponse(html_visualization)

        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
