"""Sync a GitHub installation's repositories into the code graph.

Thin orchestration: mint a fresh installation token, resolve which
repositories to index, and ``remember`` their clone URLs in one call with the
token as ``codegraph_config["repo_credentials"]``. ``add()`` clones each and
stores it as one ``code_repo`` row (with its ``repo_url``), and cognify builds
it on the CODE_REPO route -- each repository in a pipeline run of its own, so
one failing repository never rolls back or blocks the others. Clone reuse lives in ``resolve_repo_source`` and
incremental loading skips an unchanged repo, which is what makes re-running
this on every webhook cheap and idempotent.

The token reaches git only through environment config; the URL and
everything stored from it stay credential-free.

The code graph is searchable via ``SearchType.CODE``; code facts are not
embedded (``index_vectors`` stays off) and the repositories' documents are
not ingested (``include_documents`` off), so a sync makes no LLM or
embedding calls however many repositories it covers.

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
    ``codegraph_config["repo_credentials"]`` (injected into git via
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

    One blocking ``remember`` over every clone URL — callers are already off
    the request path (the post-install hook and webhook handling both run
    detached), so there is nothing to hand off to. ``treat_as_repository``
    with ``raise_on_error=False`` makes remember add and build each repository
    on its own: a repository that fails (clone, auth, pipeline) comes back as
    an errored item and is logged, and the others are built and kept.

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
    full_name_by_url = {clone_url(full_name): full_name for full_name in repo_full_names}
    try:
        result = await cognee_remember(
            list(full_name_by_url),
            dataset_name=dataset_name,
            user=owner,
            codegraph_config={
                "repo_credentials": token,
                # Every URL is a repository: clone it, never sniff it.
                "treat_as_repository": True,
                # The code graph is the point of the sync: indexing every
                # repository's README and docs would mean LLM extraction
                # and embeddings for an installation's whole prose, on
                # every webhook.
                "include_documents": False,
            },
            # The code graph is the point of the sync; no session to bridge.
            self_improvement=False,
            # Per-repository failures come back as errored items instead.
            raise_on_error=False,
        )
    except Exception:
        logger.exception("GitHub sync failed for %s", account_login)
        return

    failed: list[str] = []
    for item in getattr(result, "items", None) or []:
        if item.get("status") != "errored":
            continue
        source = item.get("source")
        full_name = full_name_by_url.get(source, source)
        logger.warning("GitHub sync for %s errored: %s", full_name, item.get("error"))
        failed.append(str(full_name))
    if not failed and getattr(result, "status", None) == "errored":
        logger.warning("GitHub sync for %s errored: %s", account_login, result.error)
    if failed:
        logger.warning(
            "GitHub sync for %s finished with %d failed repositories: %s",
            account_login,
            len(failed),
            ", ".join(failed),
        )
