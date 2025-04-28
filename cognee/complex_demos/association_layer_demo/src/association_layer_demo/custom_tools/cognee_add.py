from crewai.tools import BaseTool
from typing import Type, List, Optional
from pydantic import BaseModel, Field


class CogneeAddInput(BaseModel):
    file_content: Optional[str] = Field(
        None, description="The file content to add to Cognee memory."
    )


class CogneeAdd(BaseTool):
    name: str = "Cognee Memory ADD"
    description: str = "Add data to cognee memory to store data in memory for AI memory"
    args_schema: Type[BaseModel] = CogneeAddInput
    pruned: bool = False

    def _run(self, **kwargs) -> str:
        import cognee
        import asyncio

        async def main():
            try:
                if not self.pruned:
                    print("Pruning data…")
                    await cognee.prune.prune_data()
                    await cognee.prune.prune_system(metadata=True)
                    self.pruned = True
                await cognee.add(kwargs.get("file_content"))
                return True
            except Exception as e:
                return f"Error: {str(e)}"

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
            return "File added"
        except Exception as e:
            return f"Tool execution error: {str(e)}"
