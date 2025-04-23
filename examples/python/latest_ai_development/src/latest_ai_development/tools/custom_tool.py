from crewai.tools import BaseTool
from typing import Type, List, Optional
from pydantic import BaseModel, Field, root_validator
from cognee.api.v1.search import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines import run_tasks, Task
from cognee.tasks.experimental_tasks.node_set_edge_association import node_set_edge_association


class CogneeAddInput(BaseModel):
    """Input schema for CogneeAdd tool."""

    context: Optional[str] = Field(None, description="The text content to add to Cognee memory.")
    file_paths: Optional[List[str]] = Field(
        None, description="List of file paths to add to Cognee memory."
    )
    files: Optional[List[str]] = Field(
        None, description="Alias for file_paths; list of file URLs or paths to add to memory."
    )
    text: Optional[str] = Field(
        None, description="Alternative field for text content (maps to context)."
    )
    reasoning: Optional[str] = Field(
        None, description="Alternative field for reasoning text (maps to context)."
    )
    node_set: List[str] = Field(
        default=["default"], description="The list of node sets to store the data in."
    )

    @root_validator(pre=True)
    def normalize_inputs(cls, values):
        """Normalize different input formats to standard fields."""
        # Map alias 'files' to 'file_paths' if provided
        if values.get("files") and not values.get("file_paths"):
            values["file_paths"] = values.get("files")
        # Map text or reasoning to context if provided
        if values.get("text") and not values.get("context"):
            values["context"] = values.get("text")

        if values.get("reasoning") and not values.get("context"):
            values["context"] = values.get("reasoning")
        # Map report_section to context if provided
        if values.get("report_section") and not values.get("context"):
            values["context"] = values.get("report_section")

        # Validate that at least one input field is provided
        if not values.get("context") and not values.get("file_paths"):
            raise ValueError(
                "Either 'context', 'text', 'reasoning', or 'file_paths' must be provided"
            )

        return values


class CogneeAdd(BaseTool):
    name: str = "Cognee Memory ADD"
    description: str = "Add data to cognee memory to store data in memory for AI memory"
    args_schema: Type[BaseModel] = CogneeAddInput
    default_nodeset: List[str] = ["default"]  # Can be overridden per instance

    def _run(self, **kwargs) -> str:
        import cognee
        import asyncio

        # Use the provided node_set if given, otherwise use default_nodeset
        node_set = kwargs.get("node_set", self.default_nodeset)
        context = kwargs.get("context")
        file_paths = kwargs.get("file_paths")

        # Handle alternative input fields
        text = kwargs.get("text")
        reasoning = kwargs.get("reasoning")

        if text and not context:
            context = text

        if reasoning and not context:
            context = reasoning

        async def main(ns):
            try:
                if context:
                    # Handle text content
                    await cognee.add(context, node_set=ns)
                elif file_paths:
                    # Handle file paths
                    await cognee.add(file_paths, node_set=ns)

                run = await cognee.cognify()
                tasks = [Task(node_set_edge_association)]

                user = await get_default_user()
                pipeline = run_tasks(tasks=tasks, user=user)

                async for pipeline_status in pipeline:
                    print(
                        f"Pipeline run status: {pipeline_status.pipeline_name} - {pipeline_status.status}"
                    )

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
            result = loop.run_until_complete(main(node_set))
            return result.__name__ if hasattr(result, "__name__") else str(result)
        except Exception as e:
            return f"Tool execution error: {str(e)}"


class CogneeSearchInput(BaseModel):
    """Input schema for CogneeSearch tool."""

    query_text: Optional[str] = Field(
        None, description="The search query to find relevant information in Cognee memory."
    )
    query: Optional[str] = Field(
        None, description="Alternative field for search query (maps to query_text)."
    )
    search_term: Optional[str] = Field(
        None, description="Alternative field for search term (maps to query_text)."
    )
    node_set: List[str] = Field(
        default=["default"], description="The list of node sets to search in."
    )

    @root_validator(pre=True)
    def normalize_inputs(cls, values):
        """Normalize different input formats to standard fields."""
        # If the dictionary is empty, use a default query
        if not values:
            values["query_text"] = "Latest AI developments"
            return values

        # Map alternative search fields to query_text
        if values.get("query") and not values.get("query_text"):
            values["query_text"] = values.get("query")

        if values.get("search_term") and not values.get("query_text"):
            values["query_text"] = values.get("search_term")

        # If security_context is provided but no query, use a default
        if "security_context" in values and not values.get("query_text"):
            values["query_text"] = "Latest AI developments"

        # Ensure query_text is present
        if not values.get("query_text"):
            values["query_text"] = "Latest AI developments"

        return values


class CogneeSearch(BaseTool):
    name: str = "Cognee Memory SEARCH"
    description: str = "Search data from cognee memory to retrieve relevant information"
    args_schema: Type[BaseModel] = CogneeSearchInput
    default_nodeset: List[str] = ["default"]  # Can be overridden per instance

    def _run(self, **kwargs) -> str:
        import cognee
        import asyncio

        # Use the provided node_set if given, otherwise use default_nodeset
        node_set = kwargs.get("node_set", self.default_nodeset)

        # Get query_text from kwargs or use a default
        query_text = kwargs.get("query_text", "Latest AI developments")

        # Handle alternative input fields
        query = kwargs.get("query")
        search_term = kwargs.get("search_term")

        if query and not query_text:
            query_text = query

        if search_term and not query_text:
            query_text = search_term

        async def main(query, ns):
            try:
                # Use 'datasets' to specify which node sets (datasets) to search
                result = await cognee.search(
                    query_text=query + " Only return results from context",
                    query_type=SearchType.GRAPH_COMPLETION,
                    datasets=ns,
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
