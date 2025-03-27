from fastapi import Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


def get_visualize_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def visualize(
        user: User = Depends(get_authenticated_user),
    ):
        """This endpoint is responsible for adding data to the graph."""
        from cognee.api.v1.visualize import visualize_graph

        try:
            html_visualization = await visualize_graph()
            return html_visualization

        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
