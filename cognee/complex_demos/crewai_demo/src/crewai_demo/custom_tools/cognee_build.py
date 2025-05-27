from crewai.tools import BaseTool


class CogneeBuild(BaseTool):
    name: str = "Cognee Build"
    description: str = "Creates a memory and builds a knowledge graph using cognee."

    def _run(self, inputs) -> str:
        import cognee
        import asyncio

        async def main():
            try:
                await cognee.prune.prune_data()
                await cognee.prune.prune_system(metadata=True)

                for meta in inputs.values():
                    text = meta["file_content"]
                    node_set = meta["nodeset"]
                    await cognee.add(text, node_set=node_set)

                await cognee.cognify(is_stream_info_enabled=True)

                return "Knowledge Graph is done."
            except Exception as e:
                return f"Error: {str(e)}"

        try:
            loop = asyncio.get_event_loop()

            if not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.create_task(main())
        except Exception as e:
            return f"Tool execution error: {str(e)}"
