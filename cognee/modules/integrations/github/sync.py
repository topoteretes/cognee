"""Sync a GitHub installation's repositories into the code graph.

Thin orchestration over the existing ``remember(content_type="code")`` path:
mint a fresh installation token, resolve which repositories to index, and
hand authenticated clone URLs to the code-graph pipeline. All the heavy
lifting — clone reuse, snapshot-identity skip on unchanged repos, per-repo
failure isolation — already lives in ``resolve_repo_source`` and the
pipeline, which is what makes re-running this on every webhook cheap and
idempotent.

The indexed graph is searchable via ``SearchType.CODE`` (the code route
produces no chunks or embeddings by design); ``index_vectors`` stays off.

One dataset per installation (``github_<org>``), not per repository —
``remember`` already accepts a list of repo specs targeting one dataset, and
per-repo datasets would mean one isolated database per repo under backend
access control.
"""

import logging
import re
from typing import Any, Optional

import aiohttp

from cognee.modules.integrations.github.app_auth import (
    API_BASE_URL,
    _api_headers,
    mint_installation_token,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=30)

GITHUB_DATASET_PREFIX = "github"


def dataset_name_for_account(account_login: str) -> str:
    """The one dataset an installation's repositories land in."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", account_login).strip("_").lower()
    return f"{GITHUB_DATASET_PREFIX}_{slug or 'account'}"


def clone_url(full_name: str) -> str:
    """The credential-free https clone URL for a repository.

    Deliberately carries no token: auth travels out-of-band as
    ``repo_credentials`` (injected into git via environment config by
    ``resolve_repo_source``), so no URL-derived string — clone slugs, result
    items, logs, git error output — can ever leak a secret.
    """
    return f"https://github.com/{full_name}.git"


async def list_installation_repositories(token: str) -> list[str]:
    """Full names (``org/repo``) of every repository the installation covers."""
    full_names: list[str] = []
    url: Optional[str] = f"{API_BASE_URL}/installation/repositories?per_page=100"
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        while url:
            async with session.get(url, headers=_api_headers(token)) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"GitHub installation repository listing failed: HTTP {response.status}"
                    )
                payload: dict[str, Any] = await response.json()
                full_names.extend(
                    repo["full_name"] for repo in payload.get("repositories", []) if repo
                )
                next_link = response.links.get("next", {}).get("url")
                url = str(next_link) if next_link else None
    return full_names


async def sync_repositories(
    credential: IntegrationCredential,
    repo_full_names: Optional[list[str]] = None,
) -> None:
    """Index ``repo_full_names`` (default: every repo the installation covers).

    Runs the code-graph pipeline to completion — callers are already off the
    request path (the post-install hook and webhook handling both run
    detached), so there is nothing to hand off to.

    The minted token lives ~1 hour and repos are cloned sequentially as the
    pipeline reaches them, so a very large installation can outlive the
    token mid-batch; the affected repos surface as per-repo errors and the
    next webhook (or manual re-sync) picks them up with a fresh token.
    """
    # Imported here, not at module top: this module is imported at API
    # startup (via the adapter registration), and cognee's package root is
    # heavyweight.
    from cognee.api.v1.remember.remember import remember as cognee_remember
    from cognee.modules.users.methods import get_user

    token, _expires_at = await mint_installation_token(int(credential.provider_account_id))

    if repo_full_names is None:
        repo_full_names = await list_installation_repositories(token)
    if not repo_full_names:
        logger.info(
            "GitHub installation %s has no repositories to sync", credential.provider_account_id
        )
        return

    account_login = (credential.provider_metadata or {}).get("account_login") or str(
        credential.provider_account_id
    )
    owner = await get_user(credential.user_id)

    logger.info(
        "Syncing %d GitHub repositories for %s into dataset %s",
        len(repo_full_names),
        account_login,
        dataset_name_for_account(account_login),
    )
    result = await cognee_remember(
        [clone_url(full_name) for full_name in repo_full_names],
        dataset_name=dataset_name_for_account(account_login),
        user=owner,
        content_type="code",
        repo_credentials=token,
    )
    if getattr(result, "status", None) == "errored":
        logger.warning(
            "GitHub sync for %s finished with errors: %s",
            account_login,
            getattr(result, "error", None),
        )
