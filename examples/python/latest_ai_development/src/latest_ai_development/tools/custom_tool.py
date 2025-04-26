from crewai.tools import BaseTool
from typing import Type, Dict, Any, List
from pydantic import BaseModel, Field
from cognee.api.v1.search import SearchType
from cognee.modules.engine.models.Entity import Entity


class CogneeAddInput(BaseModel):
    """Input schema for CogneeAdd tool."""

    context: str = Field(..., description="The text content to add to Cognee memory.")
    node_set: List[str] = Field(
        default=["default"], description="The list of node sets to store the data in."
    )


class CogneeAdd(BaseTool):
    name: str = "Cognee Memory ADD"
    description: str = "Add data to cognee memory to store data in memory for AI memory"
    args_schema: Type[BaseModel] = CogneeAddInput

    def _run(self, context: str, **kwargs) -> str:
        import cognee
        import asyncio

        node_set = kwargs.get("node_set", ["default"])

        async def main(text_content, ns):
            try:
                await cognee.add(text_content, node_set=ns)
                run = await cognee.cognify()
                return run
            except Exception as e:
                return f"Error: {str(e)}"

        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(main(context, node_set))
            return result.__name__ if hasattr(result, "__name__") else str(result)
        except Exception as e:
            return f"Tool execution error: {str(e)}"


class CogneeSearchInput(BaseModel):
    """Input schema for CogneeSearch tool."""

    query_text: str = Field(
        ..., description="The search query to find relevant information in Cognee memory."
    )
    node_set: List[str] = Field(
        default=["default"], description="The list of node sets to search in."
    )


class CogneeSearch(BaseTool):
    name: str = "Cognee Memory SEARCH"
    description: str = "Search data from cognee memory to retrieve relevant information"
    args_schema: Type[BaseModel] = CogneeSearchInput

    def _run(self, query_text: str, **kwargs) -> str:
        import cognee
        import asyncio

        node_set = kwargs.get("node_set", ["default"])

        async def main(query, ns):
            try:
                result = await cognee.search(
                    query_type=SearchType.GRAPH_COMPLETION,
                    query_text=query + " Only return results from context",
                )
                return result
            except Exception as e:
                return f"Error: {str(e)}"

        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(main(query_text, node_set))
            return str(result)
        except Exception as e:
            return f"Tool execution error: {str(e)}"
