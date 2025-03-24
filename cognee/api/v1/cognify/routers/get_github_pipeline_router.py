import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from cognee.api.v1.cognify.github_developer_pipeline import run_github_developer_pipeline
from cognee.modules.storage.utils import JSONEncoder

logger = logging.getLogger(__name__)


class GithubPipelinePayloadDTO(BaseModel):
    username: str
    api_token: Optional[str] = None


def get_github_pipeline_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def run_github_pipeline(payload: GithubPipelinePayloadDTO):
        """Run GitHub developer analysis pipeline for a given username."""
        try:
            # Start the pipeline and return immediately
            result = {"status": "started", "username": payload.username}
            
            # Start the pipeline in the background
            async for _ in run_github_developer_pipeline(payload.username, payload.api_token):
                pass
                
            return JSONResponse(content=result)
        except Exception as e:
            logger.exception(f"Error in GitHub pipeline: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)},
            )

    return router 