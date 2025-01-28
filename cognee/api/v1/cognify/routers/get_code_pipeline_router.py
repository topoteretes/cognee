from fastapi import APIRouter
from pydantic import BaseModel
from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline
from cognee.modules.retrieval.description_to_codepart_search import (
    code_description_to_code_part_search,
)
from fastapi.responses import JSONResponse


class CodePipelineIndexPayloadDTO(BaseModel):
    repo_path: str
    include_docs: bool = False


class CodePipelineRetrievePayloadDTO(BaseModel):
    query: str
    fullInput: str


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
                payload.fullInput.replace("cognee ", "")
                if payload.fullInput.startswith("cognee ")
                else payload.fullInput
            )

            retrieved_codeparts, __ = await code_description_to_code_part_search(
                query, include_docs=False
            )

            return [
                {
                    "name": codepart.attributes["id"],
                    "description": codepart.attributes["id"],
                    "content": codepart.attributes["source_code"],
                }
                for codepart in retrieved_codeparts
            ]
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
