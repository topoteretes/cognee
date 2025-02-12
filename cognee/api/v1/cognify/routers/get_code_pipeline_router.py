from fastapi import APIRouter
from fastapi.responses import JSONResponse
from cognee.api.DTO import InDTO
from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.modules.retrieval import code_graph_retrieval


class CodePipelineIndexPayloadDTO(InDTO):
    repo_path: str
    include_docs: bool = False


class CodePipelineRetrievePayloadDTO(InDTO):
    query: str
    full_input: str


def get_code_pipeline_router() -> APIRouter:
    router = APIRouter()

    @router.post("/index", response_model=None)
    async def code_pipeline_index(payload: CodePipelineIndexPayloadDTO):
        """This endpoint is responsible for running the indexation on code repo."""
        try:
            async for result in run_code_graph_pipeline(payload.repo_path, payload.include_docs):
                print(result)
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.post("/retrieve", response_model=list[dict])
    async def code_pipeline_retrieve(payload: CodePipelineRetrievePayloadDTO):
        """This endpoint is responsible for retrieving the context."""
        try:
            query = (
                payload.full_input.replace("cognee ", "")
                if payload.full_input.startswith("cognee ")
                else payload.full_input
            )

            retrieved_files = await code_graph_retrieval(query)

            return [
                {
                    "name": file_path,
                    "description": file_path,
                    "content": source_code,
                }
                for file_path, source_code in retrieved_files.items()
            ]
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
