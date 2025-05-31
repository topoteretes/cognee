from uuid import UUID
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
    dataset_ids: Optional[List[UUID]]
    graph_model: Optional[BaseModel] = KnowledgeGraph


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """This endpoint is responsible for the cognitive processing of the content."""
        from cognee.api.v1.cognify import cognify as cognee_cognify

        try:
            # Send dataset UUIDs if they are given, if not send dataset names
            datasets = payload.dataset_ids if payload.dataset_ids else payload.datasets
            await cognee_cognify(datasets, user, payload.graph_model)
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
