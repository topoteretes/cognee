"""
Router for the layered graph cognify endpoint.

This module provides a FastAPI router for the layered graph cognify endpoint,
which can be used to process documents and extract layered knowledge graphs.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user


class LayeredGraphConfigDTO(BaseModel):
    """Configuration for a layer in a layered knowledge graph."""
    name: str
    description: str
    layer_type: str
    prompt: str


class PipelineStepConfigDTO(BaseModel):
    """Configuration for a pipeline step in layered graph processing."""
    type: str
    description: str
    enrichment_type: Optional[str] = None
    content: Optional[str] = None
    parent_layer_ids: Optional[List[str]] = None


class LayeredGraphCognifyPayloadDTO(BaseModel):
    """Payload for the layered graph cognify endpoint."""
    datasets: List[str]
    layer_config: Optional[List[LayeredGraphConfigDTO]] = None
    pipeline_config: Optional[List[PipelineStepConfigDTO]] = None


def get_layered_graph_router() -> APIRouter:
    """
    Create a FastAPI router for the layered graph cognify endpoint.
    
    Returns:
        A FastAPI router
    """
    router = APIRouter()

    @router.post("/layered-graph", response_model=None)
    async def cognify_layered_graph(
        payload: LayeredGraphCognifyPayloadDTO, 
        user: User = Depends(get_authenticated_user)
    ):
        """
        Process documents to extract layered knowledge graphs.
        
        Args:
            payload: The request payload containing datasets and configurations
            user: The authenticated user
            
        Returns:
            The pipeline execution results
        """
        from cognee.api.v1.cognify.cognify_layered_graph import cognify_layered_graph as cognee_cognify_layered_graph

        # Convert DTOs to dictionaries for the pipeline
        layer_config = None
        if payload.layer_config:
            layer_config = [layer.dict() for layer in payload.layer_config]
        
        pipeline_config = None
        if payload.pipeline_config:
            pipeline_config = [step.dict() for step in payload.pipeline_config]
        
        try:
            result = await cognee_cognify_layered_graph(
                datasets=payload.datasets,
                user=user,
                layer_config=layer_config,
                pipeline_config=pipeline_config
            )
            return result
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router 