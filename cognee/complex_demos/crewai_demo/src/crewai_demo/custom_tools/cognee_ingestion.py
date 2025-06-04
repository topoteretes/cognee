import asyncio
import nest_asyncio
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr

from cognee.modules.users.models import User


class CogneeIngestionInput(BaseModel):
    text: str = Field(
        "",
        description="The text of the report The format you should follow is {'text': 'your report'}",
    )


class CogneeIngestion(BaseTool):
    name: str = "ingest_report_to_cognee"
    description: str = "This tool can be used to ingest the final hiring report into cognee"
    args_schema: Type[BaseModel] = CogneeIngestionInput
    _user: User = PrivateAttr()
    _nodeset_name: str = PrivateAttr()

    def __init__(self, user: User, nodeset_name: str, **kwargs):
        super().__init__(**kwargs)
        self._user = user
        self._nodeset_name = nodeset_name

    def _run(self, text: str) -> str:
        import cognee
        # from secrets import choice
        # from string import ascii_letters, digits

        async def main():
            try:
                # hash6 = "".join(choice(ascii_letters + digits) for _ in range(6))
                dataset_name = "Github"
                data = await cognee.add(
                    text,
                    node_set=[self._nodeset_name],
                    dataset_name=dataset_name,
                    user=self._user,
                )
                await cognee.cognify(
                    datasets=dataset_name,
                    is_stream_info_enabled=True,
                    datapoints=data.packets,
                    user=self._user,
                    pipeline_name="github_pipeline",
                )

                return "Report ingested successfully into Cognee memory."
            except Exception as e:
                return f"Error during ingestion: {str(e)}"

        try:
            try:
                loop = asyncio.get_event_loop()

                if not loop.is_running():
                    loop = asyncio.new_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

            if not loop.is_running():
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(main())

            return result
        except Exception as e:
            return f"Tool execution error: {str(e)}"
