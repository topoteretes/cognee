from cognee.api.v1.cognify.routers import (
    get_cognify_router,
    get_code_pipeline_router,
    get_layered_graph_router
)

# Add the routers to the API
api_router.include_router(get_cognify_router(), prefix="/cognify", tags=["cognify"])
api_router.include_router(get_code_pipeline_router(), prefix="/cognify", tags=["cognify"])
api_router.include_router(get_layered_graph_router(), prefix="/cognify/layered-graph", tags=["cognify", "layered-graph"]) 