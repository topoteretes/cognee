from crewai.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field, PrivateAttr
from cognee.modules.engine.models import NodeSet
import asyncio


class CogneeIngestionInput(BaseModel):
    text: str = Field(
        "",
        description="The text of the report The format you should follow is {'text': 'your report'}",
    )


class CogneeIngestion(BaseTool):
    name: str = "ingest_report_to_cognee"
    description: str = "This tool can be used to ingest the final hiring report into cognee"
    args_schema: Type[BaseModel] = CogneeIngestionInput
    _nodeset_name: str

    def __init__(self, nodeset_name: str, **kwargs):
        super().__init__(**kwargs)
        self._nodeset_name = nodeset_name

    def _run(self, text: str) -> str:
        import cognee
        from secrets import choice
        from string import ascii_letters, digits

        async def main():
            try:
                hash6 = "".join(choice(ascii_letters + digits) for _ in range(6))
                await cognee.add(text, node_set=[self._nodeset_name], dataset_name=hash6)
                await cognee.cognify(datasets=hash6)

                return "Report ingested successfully into Cognee memory."
            except Exception as e:
                return f"Error during ingestion: {str(e)}"

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(main())
        except Exception as e:
            return f"Tool execution error: {str(e)}"
