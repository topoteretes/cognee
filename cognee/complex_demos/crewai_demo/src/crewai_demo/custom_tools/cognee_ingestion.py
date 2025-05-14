from crewai.tools import BaseTool
from typing import Type, List, Optional
from pydantic import BaseModel, Field, PrivateAttr
from cognee.api.v1.search import SearchType
from cognee.modules.engine.models import NodeSet
from cognee.modules.search.methods import search
from cognee.modules.users.methods import get_default_user


class CogneeIngestionInput(BaseModel):
    text: str = Field(None, description="Report to ingest into cognee AI memory")


class CogneeIngestion(BaseTool):
    name: str = "Cognee report ingestion"
    description: str = "Ingest report into cognee AI memory"
    args_schema: Type[BaseModel] = CogneeIngestionInput
    _nodeset_name: List[str] = PrivateAttr()

    def __init__(self, nodeset_name: List[str], **kwargs):
        super().__init__(**kwargs)
        self._nodeset_name = nodeset_name

    def _run(self, **kwargs) -> str:
        import cognee
        import asyncio

        async def main():
            try:
                print(kwargs.get("text"))
                text = kwargs.get("text")
                await cognee.add(text, node_set="reports")
                #:TODO: finish
                return "Report ingested into Cognee"
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
