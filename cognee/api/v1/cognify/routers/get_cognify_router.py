from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from cognee.modules.users.models import User
from fastapi.responses import JSONResponse
from cognee.modules.users.methods import get_authenticated_user
from fastapi import Depends


class CognifyPayloadDTO(BaseModel):
    datasets: List[str]
    graph_model: BaseModel

def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user), graph_model:BaseModel=None):
        """ This endpoint is responsible for the cognitive processing of the content."""
        from cognee.api.v1.cognify.cognify_v2 import cognify as cognee_cognify
        try:
            await cognee_cognify(payload.datasets, user, payload.graph_model)
        except Exception as error:
            return JSONResponse(
                status_code=409,
                content={"error": str(error)}
            )

    return router
