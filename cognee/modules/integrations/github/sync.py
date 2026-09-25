"""Sync a GitHub installation's repositories into the code graph.

Thin orchestration: mint a fresh installation token, resolve which
repositories to index, clone each one with the token, and ``remember`` the
local clone. ``remember`` stores the clone as one ``code_repo`` row through
``add()`` and cognify builds it on the CODE_REPO route. Clone reuse lives in
``resolve_repo_source`` and incremental loading skips an unchanged repo, which
is what makes re-running this on every webhook cheap and idempotent.

The token reaches git only through ``resolve_repo_source(credentials=...)``;
``remember`` sees a local path, so no secret rides the data it stores.

The indexed graph is searchable via ``SearchType.CODE``; code facts are not
embedded (``index_vectors`` stays off).

One dataset per installation (``github_<org>``), not per repository —
per-repo datasets would mean one isolated database per repo under backend
access control.
"""

import logging
import re
from typing import Any

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
    ``resolve_repo_source(credentials=...)`` (injected into git via
    environment config), so no URL-derived string — clone slugs, stored rows,
    logs, git error output — can ever leak a secret.
    """
    return f"https://github.com/{full_name}.git"


async def list_installation_repositories(token: str) -> list[str]:
    """Full names (``org/repo``) of every repository the installation covers."""
    full_names: list[str] = []
    url: str | None = f"{API_BASE_URL}/installation/repositories?per_page=100"
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
    repo_full_names: list[str] | None = None,
) -> None:
    """Index ``repo_full_names`` (default: every repo the installation covers).

    Runs each repository to completion — callers are already off the request
    path (the post-install hook and webhook handling both run detached), so
    there is nothing to hand off to. A repository that fails (clone, auth,
    pipeline) is logged and the sync continues with the next one.

    The minted token lives ~1 hour and repos are cloned sequentially, so a
    very large installation can outlive the token mid-batch; the affected
    repos fail individually and the next webhook (or manual re-sync) picks
    them up with a fresh token.
    """
    # Imported here, not at module top: this module is imported at API
    # startup (via the adapter registration), and cognee's package root is
    # heavyweight.
    from cognee.api.v1.remember.remember import remember as cognee_remember
    from cognee.modules.users.methods import get_user
    from cognee.tasks.code_graph.resolve_repo import resolve_repo_source

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

    dataset_name = dataset_name_for_account(account_login)
    logger.info(
        "Syncing %d GitHub repositories for %s into dataset %s",
        len(repo_full_names),
        account_login,
        dataset_name,
    )
    failed: list[str] = []
    for full_name in repo_full_names:
        try:
            repo_path = await resolve_repo_source(clone_url(full_name), credentials=token)
            result = await cognee_remember(
                str(repo_path),
                dataset_name=dataset_name,
                user=owner,
                # The code graph is the point of the sync; no session to bridge.
                self_improvement=False,
                raise_on_error=False,
            )
        except Exception:
            logger.exception("GitHub sync failed for repository %s", full_name)
            failed.append(full_name)
            continue
        if getattr(result, "status", None) == "errored":
            logger.warning(
                "GitHub sync for %s errored: %s", full_name, getattr(result, "error", None)
            )
            failed.append(full_name)
    if failed:
        logger.warning(
            "GitHub sync for %s finished with %d failed repositories: %s",
            account_login,
            len(failed),
            ", ".join(failed),
        )
