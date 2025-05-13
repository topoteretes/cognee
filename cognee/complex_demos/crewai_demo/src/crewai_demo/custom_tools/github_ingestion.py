from crewai.tools import BaseTool
from ..github_ingest import cognify_github_profile


class GithubIngestion(BaseTool):
    name: str = "Github graph builder"
    description: str = "Ingests the github graph of a person into Cognee"

    def _run(self, applicant_1, applicant_2) -> str:
        import asyncio
        import cognee
        import os

        async def main():
            try:
                await cognee.prune.prune_data()
                await cognee.prune.prune_system(metadata=True)

                token = os.getenv("GITHUB_TOKEN")

                await cognify_github_profile(applicant_1, token)
                await cognify_github_profile(applicant_2, token)
                return "Github ingestion finished"
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
