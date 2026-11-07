import os
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, List
import httpx

class GitHubActionsConnector:
    """
    Data source connector to ingest GitHub Actions workflow runs, jobs, 
    and logs into Cognee AI graph memory.
    """
    def __init__(
        self, 
        owner: str, 
        repo: str, 
        token: Optional[str] = None,
        base_url: str = "https://api.github.com"
    ):
        self.owner = owner
        self.repo = repo
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url.rstrip("/")
        
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    async def _fetch_json(self, client: httpx.AsyncClient, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = await client.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    async def fetch_workflow_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        endpoint = f"/repos/{self.owner}/{self.repo}/actions/runs"
        async with httpx.AsyncClient() as client:
            data = await self._fetch_json(client, endpoint, params={"per_page": limit})
            return data.get("workflow_runs", [])

    async def fetch_run_jobs(self, run_id: int) -> List[Dict[str, Any]]:
        endpoint = f"/repos/{self.owner}/{self.repo}/actions/runs/{run_id}/jobs"
        async with httpx.AsyncClient() as client:
            data = await self._fetch_json(client, endpoint)
            return data.get("jobs", [])

    def format_run_as_markdown(self, run: Dict[str, Any], jobs: List[Dict[str, Any]]) -> str:
        md = [
            f"# GitHub Actions Workflow Run: {run.get('name', 'Unnamed Workflow')}",
            f"- **Repository:** {self.owner}/{self.repo}",
            f"- **Run ID:** {run.get('id')}",
            f"- **Status:** {run.get('status')}",
            f"- **Conclusion:** {run.get('conclusion')}",
            f"- **Event:** {run.get('event')}",
            f"- **Branch:** {run.get('head_branch')}",
            f"- **Commit SHA:** {run.get('head_sha')}",
            f"- **URL:** {run.get('html_url')}",
            "\n## Job Details"
        ]

        for job in jobs:
            md.append(f"### Job: {job.get('name')} (Status: {job.get('conclusion') or job.get('status')})")
            md.append(f"- **Started At:** {job.get('started_at')}")
            md.append(f"- **Completed At:** {job.get('completed_at')}")
            md.append("#### Steps:")
            
            for step in job.get("steps", []):
                icon = "✅" if step.get("conclusion") == "success" else "❌"
                md.append(f"  - {icon} **{step.get('name')}** (Status: {step.get('conclusion')})")
        
        return "\n".join(md)

    async def fetch_data(self, limit: int = 10) -> AsyncGenerator[Dict[str, Any], None]:
        runs = await self.fetch_workflow_runs(limit=limit)
        
        for run in runs:
            run_id = run.get("id")
            jobs = await self.fetch_run_jobs(run_id)
            content = self.format_run_as_markdown(run, jobs)
            
            yield {
                "id": f"gh_action_run_{run_id}",
                "name": f"GH Action Run #{run_id} - {run.get('name')}",
                "content": content,
                "metadata": {
                    "source": "github_actions",
                    "owner": self.owner,
                    "repo": self.repo,
                    "run_id": run_id,
                    "conclusion": run.get("conclusion"),
                }
            }

async def fetch_github_actions(owner: str, repo: str, token: Optional[str] = None, limit: int = 10):
    connector = GitHubActionsConnector(owner=owner, repo=repo, token=token)
    async for item in connector.fetch_data(limit=limit):
        yield item
