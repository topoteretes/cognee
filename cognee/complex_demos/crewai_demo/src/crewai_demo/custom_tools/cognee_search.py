from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr

from cognee.modules.engine.models import NodeSet


class CogneeSearchInput(BaseModel):
    query: str = Field(
        "",
        description="The natural language question to ask the memory engine."
        "The format you should follow is {'query': 'your query'}",
    )


class CogneeSearch(BaseTool):
    name: str = "search_from_cognee"
    description: str = (
        "Use this tool to search the Cognee memory graph. "
        "Provide a natural language query that describes the information you want to retrieve, "
        "such as comments authored or files changes by a specific person."
    )
    args_schema: Type[BaseModel] = CogneeSearchInput
    _nodeset_name: str = PrivateAttr()

    def __init__(self, nodeset_name: str, **kwargs):
        super().__init__(**kwargs)
        self._nodeset_name = nodeset_name

    def _run(self, query: str) -> str:
        import asyncio
        from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever

        async def main():
            try:
                print(query)

                search_results = await GraphCompletionRetriever(
                    top_k=5,
                    node_type=NodeSet,
                    node_name=[self._nodeset_name],
                ).get_context(query=query)

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
