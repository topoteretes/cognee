from crewai.tools import BaseTool


class CogneeBuild(BaseTool):
    name: str = "Cognee Build"
    description: str = "Creates a memory and builds a knowledge graph using cognee."
    pruned: bool = False

    def _run(self, **kwargs) -> str:
        import cognee
        import asyncio

        async def main():
            try:
                text_1 = "Cognee is an AI memory engine"
                text_2 = "Germany is a country"

                await cognee.prune.prune_data()
                await cognee.prune.prune_system(metadata=True)

                await cognee.add(text_1, node_set=["first_text"])
                await cognee.add(text_2, node_set=["second_text"])
                await cognee.cognify()

                return "Knowledge Graph is done."
            except Exception as e:
                return f"Error: {str(e)}"

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            results = loop.run_until_complete(main())
            return results
        except Exception as e:
            return f"Tool execution error: {str(e)}"
