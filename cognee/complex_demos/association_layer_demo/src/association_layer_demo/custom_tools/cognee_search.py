from crewai.tools import BaseTool
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from cognee.api.v1.search import SearchType
from cognee.modules.search.methods import search


class CogneeSearchInput(BaseModel):
    query: Optional[str] = Field(
        None, description="The query/question provided to the search engine"
    )


class CogneeSearch(BaseTool):
    name: str = "Cognee Memory SEARCH"
    description: str = (
        "Search inside the cognee memory engine, providing different questions/queries to answer."
    )
    args_schema: Type[BaseModel] = CogneeSearchInput
    pruned: bool = False

    def _run(self, **kwargs) -> str:
        import cognee
        import asyncio

        async def main():
            try:
                search_results = await cognee.search(
                    query_type=SearchType.GRAPH_COMPLETION, query_text=kwargs.get("query")
                )
                return search_results
            except Exception as e:
                return f"Error: {str(e)}"

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            search_results = loop.run_until_complete(main())
            return search_results
        except Exception as e:
            return f"Tool execution error: {str(e)}"
