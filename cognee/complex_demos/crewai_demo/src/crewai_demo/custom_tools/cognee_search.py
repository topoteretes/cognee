from crewai.tools import BaseTool
from typing import Type, List, Optional
from pydantic import BaseModel, Field, PrivateAttr
from cognee.api.v1.search import SearchType
from cognee.modules.engine.models import NodeSet
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.search.methods import search
from cognee.modules.users.methods import get_default_user


class CogneeSearchInput(BaseModel):
    query: str = Field(None, description="Query to ask from the search engine.")


class CogneeSearch(BaseTool):
    name: str = "Cognee Memory SEARCH"
    description: str = "Search inside the cognee memory engine by providing the query"
    args_schema: Type[BaseModel] = CogneeSearchInput
    _nodeset_name: List[str] = PrivateAttr()

    def __init__(self, nodeset_name: List[str], **kwargs):
        super().__init__(**kwargs)
        self._nodeset_name = nodeset_name

    def _run(self, **kwargs) -> str:
        import cognee
        import asyncio

        async def main():
            try:
                print(kwargs.get("query"))

                search_results = await GraphCompletionRetriever(
                    top_k=5,
                    node_type=NodeSet,
                    node_name=self._nodeset_name,
                ).get_context(query=kwargs.get("query"))

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
