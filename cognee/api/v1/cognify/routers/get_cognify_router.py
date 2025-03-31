from typing import List, Optional
from pydantic import BaseModel
from fastapi import Depends
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.data_models import KnowledgeGraph


class CognifyPayloadDTO(BaseModel):
    datasets: List[str]
    graph_model: Optional[BaseModel] = KnowledgeGraph


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """This endpoint is responsible for the cognitive processing of the content."""
        from cognee.api.v1.cognify import cognify as cognee_cognify

        try:
            await cognee_cognify(payload.datasets, user, payload.graph_model)
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
