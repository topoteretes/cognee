from typing import List, Optional
from pydantic import BaseModel
from fastapi import Depends
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.api.v1.infrastructure import get_or_create_dataset_database
from cognee.shared.data_models import KnowledgeGraph
from cognee.context_global_variables import (
    graph_db_config as context_graph_db_config,
    vector_db_config as context_vector_db_config,
)


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
            # TODO: Make async gather per dataset
            for dataset in payload.datasets:
                # TODO: Move context setting handler outside of APIs
                dataset_database = await get_or_create_dataset_database(dataset, user)
                vector_config = {
                    "vector_db_url": dataset_database.vector_database_name,
                    "vector_db_key": "",
                    "vector_db_provider": "lancedb",
                }

                graph_config = {
                    "graph_database_provider": "kuzu",
                    "graph_file_path": dataset_database.graph_database_name,
                }

                context_graph_db_config.set(graph_config)
                context_vector_db_config.set(vector_config)

                await cognee_cognify(payload.datasets, user, payload.graph_model)
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
