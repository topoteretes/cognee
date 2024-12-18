from fastapi import APIRouter
from typing import List, Optional, Any
from pydantic import BaseModel, create_model
from cognee.modules.users.models import User
from fastapi.responses import JSONResponse
from cognee.modules.users.methods import get_authenticated_user
from fastapi import Depends

from cognee.shared.data_models import KnowledgeGraph


class CognifyPayloadDTO(BaseModel):
    datasets: List[str]
    graph_model: Optional[Any] = None


def json_to_pydantic_model(name: str, json_schema: dict) -> BaseModel:
    """
    Create a Pydantic model on the fly from JSON.
    """
    return create_model(name, **{k: (type(v), ...) for k, v in json_schema.items()})

def get_cognify_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def cognify(payload: CognifyPayloadDTO, user: User = Depends(get_authenticated_user)):
        """ This endpoint is responsible for the cognitive processing of the content."""
        from cognee.api.v1.cognify.cognify_v2 import cognify as cognee_cognify
        try:
            # Dynamic conversion of `graph_model` to Pydantic
            if payload.graph_model:
                graph_model_schema = payload.graph_model
                GraphModelDynamic = json_to_pydantic_model("GraphModelDynamic", graph_model_schema)
                graph_model_instance = GraphModelDynamic(**graph_model_schema)
                print(graph_model_instance)
            else:
                graph_model_instance = KnowledgeGraph

            await cognee_cognify(payload.datasets, user, graph_model_instance)
        except Exception as error:
            return JSONResponse(
                status_code=409,
                content={"error": str(error)}
            )

    return router
