from typing import List, Optional
import asyncio
from pydantic import BaseModel
from fastapi import Depends
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.data_models import KnowledgeGraph
from cognee.context_global_variables import set_database_global_context_variables


class CognifyPayloadDTO(BaseModel):
    datasets: List[str]
    graph_model: Optional[BaseModel] = KnowledgeGraph


def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """This endpoint is responsible for the cognitive processing of the content."""

        async def cognify_dataset(dataset, user, graph_model):
            """Process a single dataset in its own async task to allow use of context database values per dataset."""
            # Set DB context for just this dataset
            await set_database_global_context_variables(dataset, user)
            # Run Cognify on this dataset
            from cognee.api.v1.cognify import cognify as cognee_cognify

            await cognee_cognify(dataset, user, graph_model)

        try:
            # Create cognify task for each dataset
            tasks = [
                cognify_dataset(dataset, user, payload.graph_model) for dataset in payload.datasets
            ]

            # Wait for all datasets to finish.
            await asyncio.gather(*tasks)
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
