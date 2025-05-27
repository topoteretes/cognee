from crewai.tools import BaseTool

from ..github_ingest_datapoints import cognify_github_data_from_username


class GithubIngestion(BaseTool):
    name: str = "Github graph builder"
    description: str = "Ingests the github graph of a person into Cognee"

    def _run(self, applicant_1, applicant_2) -> str:
        import asyncio

        # import cognee
        import os
        # from cognee.low_level import setup as cognee_setup

        async def main():
            try:
                # await cognee.prune.prune_data()
                # await cognee.prune.prune_system(metadata=True)
                # await cognee_setup()
                token = os.getenv("GITHUB_TOKEN")

                await cognify_github_data_from_username(applicant_1, token)
                await cognify_github_data_from_username(applicant_2, token)

                return True
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
