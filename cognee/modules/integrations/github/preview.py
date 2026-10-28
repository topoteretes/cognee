"""Preview GitHub file selection without downloading blobs or changing memory."""

from urllib.parse import quote

import aiohttp

from cognee.modules.integrations.credentials import require_active_credential
from cognee.modules.integrations.github.app_auth import (
    API_BASE_URL,
    _api_headers,
    mint_installation_token,
)
from cognee.modules.integrations.github.sync import list_installation_repository_records
from cognee.modules.integrations.github.sync_settings import GithubRepositorySettings
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.tasks.code_graph.repo_selection import preview_tree


def tree_listing(payload: dict) -> str:
    """Translate a complete GitHub tree to the selector's Git ls-tree format."""
    if payload.get("truncated") is not False or not isinstance(payload.get("tree"), list):
        raise ValueError("GitHub returned an incomplete repository tree; preview is unavailable")
    records = []
    for entry in payload["tree"]:
        if not isinstance(entry, dict):
            raise ValueError("GitHub returned an invalid repository tree")
        if entry.get("type") == "tree":
            continue
        fields = [entry.get(key) for key in ("mode", "type", "sha", "path")]
        if not all(isinstance(value, str) and "\0" not in value for value in fields):
            raise ValueError("GitHub returned an invalid repository tree entry")
        mode, kind, sha, path = fields
        records.append(f"{mode} {kind} {sha}\t{path}\0")
    return "".join(records)


async def preview_repository(
    credential: IntegrationCredential,
    repository_id: str,
    settings: GithubRepositorySettings,
) -> dict:
    """Resolve a repository through this installation before reading its tree.

    Truncated trees are errors, never partial previews. The returned tree SHA lets
    callers distinguish this preview from a later sync of a changed branch.
    """
    credential = await require_active_credential(credential)
    if credential.provider != "github":
        raise ValueError("A GitHub connection is required")
    settings = GithubRepositorySettings.model_validate(settings)
    token, _ = await mint_installation_token(int(credential.provider_account_id))
    repositories = await list_installation_repository_records(token)
    repository = next((item for item in repositories if str(item["id"]) == repository_id), None)
    if repository is None:
        raise ValueError("Repository is not available to this GitHub installation")
    branch = settings.branch or repository.get("default_branch")
    if not branch:
        raise ValueError("Repository has no branch to preview")
    # Encode the provider's name too; only the installation listing supplies it.
    full_name = quote(repository["full_name"], safe="/")
    url = f"{API_BASE_URL}/repos/{full_name}/git/trees/{quote(branch, safe='')}"
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session,
        session.get(url, params={"recursive": "1"}, headers=_api_headers(token)) as response,
    ):
        if response.status != 200:
            raise RuntimeError(f"GitHub tree lookup failed: HTTP {response.status}")
        payload = await response.json()
    await require_active_credential(credential)
    result = preview_tree(
        tree_listing(payload),
        include_documents=settings.include_documents,
        include_paths=settings.include_paths,
        exclude_paths=settings.exclude_paths,
    )
    return {
        "repository_id": repository_id,
        "branch": branch,
        "tree_sha": payload.get("sha"),
        **result,
    }
