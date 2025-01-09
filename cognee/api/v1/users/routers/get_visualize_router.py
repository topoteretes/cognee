from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List
import aiohttp
import subprocess
import logging
import os
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = logging.getLogger(__name__)


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
