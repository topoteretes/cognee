from crewai.tools import BaseTool


class CogneeCognify(BaseTool):
    name: str = "Cognee Memory COGNIFY"
    description: str = "This tool can be used to cognify the ingested dataset with Cognee"

    def _run(self, **kwargs) -> str:
        import cognee
        import asyncio

        async def main():
            try:
                print("Cognifying dataset…")
                await cognee.cognify()
                return True
            except Exception as e:
                return f"Error: {str(e)}"

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
            return "Dataset cognified"
        except Exception as e:
            return f"Tool execution error: {str(e)}"
