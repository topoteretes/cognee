import json
from cognee.shared.logging_utils import get_logger
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from cognee.api.DTO import InDTO
from cognee.modules.retrieval.code_retriever import CodeRetriever
from cognee.modules.storage.utils import JSONEncoder


logger = get_logger()


class CodePipelineIndexPayloadDTO(InDTO):
    repo_path: str
    include_docs: bool = False


class CodePipelineRetrievePayloadDTO(InDTO):
    query: str
    full_input: str


def get_code_pipeline_router() -> APIRouter:
    try:
        import cognee.api.v1.cognify.code_graph_pipeline
    except ModuleNotFoundError:
        logger.error("codegraph dependencies not found. Skipping codegraph API routes.")
        return None

    router = APIRouter()

    @router.post("/index", response_model=None)
    async def code_pipeline_index(payload: CodePipelineIndexPayloadDTO):
        """
        Run indexation on a code repository.

        This endpoint processes a code repository to create a knowledge graph
        of the codebase structure, dependencies, and relationships.

        ## Request Parameters
        - **repo_path** (str): Path to the code repository
        - **include_docs** (bool): Whether to include documentation files (default: false)

        ## Response
        No content returned. Processing results are logged.

        ## Error Codes
        - **409 Conflict**: Error during indexation process
        """
        from cognee.api.v1.cognify.code_graph_pipeline import run_code_graph_pipeline

        try:
            async for result in run_code_graph_pipeline(payload.repo_path, payload.include_docs):
                logger.info(result)
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.post("/retrieve", response_model=list[dict])
    async def code_pipeline_retrieve(payload: CodePipelineRetrievePayloadDTO):
        """
        Retrieve context from the code knowledge graph.

        This endpoint searches the indexed code repository to find relevant
        context based on the provided query.

        ## Request Parameters
        - **query** (str): Search query for code context
        - **full_input** (str): Full input text for processing

        ## Response
        Returns a list of relevant code files and context as JSON.

        ## Error Codes
        - **409 Conflict**: Error during retrieval process
        """
        try:
            query = (
                payload.full_input.replace("cognee ", "")
                if payload.full_input.startswith("cognee ")
                else payload.full_input
            )

            retriever = CodeRetriever()
            retrieved_files = await retriever.get_context(query)

            return json.dumps(retrieved_files, cls=JSONEncoder)
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
