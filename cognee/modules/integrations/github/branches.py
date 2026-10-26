"""Branch discovery and selection within the connected installation's scope."""

from urllib.parse import quote

from cognee.modules.integrations.credentials import update_provider_metadata
from cognee.modules.integrations.github.app_auth import API_BASE_URL, mint_installation_token
from cognee.modules.integrations.github.sync import (
    REPOSITORY_BRANCHES_KEY,
    get_pages,
    list_installation_repositories,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.tasks.code_graph.resolve_repo import check_branch_name


async def _installation_repositories(credential: IntegrationCredential, repository: str):
    """The token and the installation's repositories; ``repository`` must be one of them."""
    token, _ = await mint_installation_token(int(credential.provider_account_id))
    repositories = await list_installation_repositories(token)
    if repository not in repositories:
        raise PermissionError("Repository is not accessible to this installation")
    return token, repositories


async def _list_branches(token: str, repository: str) -> list[str]:
    url = f"{API_BASE_URL}/repos/{quote(repository, safe='/')}/branches?per_page=100"
    return [
        item["name"]
        async for page in get_pages(url, token, "GitHub branch listing failed")
        for item in page
    ]


async def list_repository_branches(credential: IntegrationCredential, repository: str) -> list[str]:
    token, _ = await _installation_repositories(credential, repository)
    return await _list_branches(token, repository)


async def select_repository_branch(
    credential: IntegrationCredential,
    repository: str,
    branch: str | None,
) -> None:
    """Store ``branch`` for ``repository``; None returns it to the default branch.

    Selections of repositories the installation no longer covers are dropped, so a
    repository added back later does not silently resume an old choice.
    """
    if branch is not None:
        check_branch_name(branch)
    token, repositories = await _installation_repositories(credential, repository)
    if branch is not None and branch not in await _list_branches(token, repository):
        raise ValueError("Selected branch is not available in this repository")

    def changed(metadata: dict) -> dict:
        stored = metadata.get(REPOSITORY_BRANCHES_KEY) or {}
        if not isinstance(stored, dict):
            raise ValueError("GitHub branch selection must be a repository-to-branch mapping")
        branches = {name: chosen for name, chosen in stored.items() if name in repositories}
        if branch is None:
            branches.pop(repository, None)
        else:
            branches[repository] = branch
        return {REPOSITORY_BRANCHES_KEY: branches}

    # Computed from the metadata as stored when it is written: a selection made
    # meanwhile for another repository is kept.
    if await update_provider_metadata("github", credential.provider_account_id, changed) is None:
        raise PermissionError("The GitHub connection is no longer active")
