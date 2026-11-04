"""Sync a GitHub installation's repositories into the code graph.

Copied from cognee.modules.integrations.github.sync (cognee dev, 2026-10-26).
``clone_url`` now lives in :mod:`cognee_connector_github.app_auth`, where the
host-free connector also builds clone URLs from; it is re-exported here.

Thin orchestration: mint a fresh installation token, resolve which
repositories to index, and ``remember`` the clone URLs in one call with the
token as ``codegraph_config["repo_credentials"]``. ``add()`` clones each one
and stores it as one ``code_repo`` row (with its ``repo_url``), cognify builds
them on the CODE_REPO route, and ``remember`` skips its automatic improve
for an all-repository call. Clone reuse lives in ``resolve_repo_source`` and
incremental loading skips an unchanged repository, which is what makes
re-running this on every webhook cheap and idempotent.

The token reaches git only through environment config; the URL and
everything stored from it stay credential-free.

The code graph is searchable via ``SearchType.CODE``; code facts are not
embedded (``index_vectors`` stays off) and the repositories' documents are
not ingested (``include_documents`` off), so a sync makes no LLM or
embedding calls however many repositories it covers.

One dataset per installation (``github_<org>``), not per repository —
``remember`` takes the whole list for one dataset, and per-repo datasets
would mean one isolated database per repo under backend access control.
"""

import logging
import re
from typing import Any

import aiohttp
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential

from cognee_connector_github.app_auth import (
    API_BASE_URL,
    _api_headers,
    clone_url,
    mint_installation_token,
)

try:
    from cognee.modules.integrations.sync_progress import report_sync_progress, track_sync
except ImportError:
    # cognee <= 1.6.6 has no sync-progress seam yet; the sync runs the same
    # way with every report dropped (see cognee_host/compat.py).
    from cognee_connector_github.cognee_host.compat import report_sync_progress, track_sync

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=30)

GITHUB_DATASET_PREFIX = "github"


def dataset_name_for_account(account_login: str) -> str:
    """The one dataset an installation's repositories land in."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", account_login).strip("_").lower()
    return f"{GITHUB_DATASET_PREFIX}_{slug or 'account'}"


async def list_installation_repositories(token: str) -> list[str]:
    """Full names (``org/repo``) of every repository the installation covers."""
    full_names: list[str] = []
    url: str | None = f"{API_BASE_URL}/installation/repositories?per_page=100"
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        while url:
            async with session.get(
                url, headers=_api_headers(token), allow_redirects=False
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"GitHub installation repository listing failed: HTTP {response.status}"
                    )
                payload: dict[str, Any] = await response.json()
                full_names.extend(
                    repo["full_name"] for repo in payload.get("repositories", []) if repo
                )
                await report_sync_progress(
                    phase="discovering",
                    completed=len(full_names),
                    unit="repositories",
                    counts={"scanned": len(full_names)},
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
    detached), so there is nothing to hand off to. ``raise_on_error=False``
    reports a repository whose build failed as an errored item of the result
    (``kind="code_repository"``, ``source``, ``error``) and the run as errored;
    cognify rolls that run's writes back, so the repositories it had built
    are rebuilt by the next sync. A clone that fails (auth, a deleted
    repository) raises out of ``add()`` before anything is stored, and the
    sync ends failed.

    The minted token lives ~1 hour and repos are cloned one after the other,
    so a very large installation can outlive the token mid-batch; the next
    webhook (or manual re-sync) picks the rest up with a fresh token.
    """
    # Imported here, not at module top: this module is imported at API
    # startup (via the adapter registration), and cognee's package root is
    # heavyweight.
    from cognee.api.v1.remember.remember import remember as cognee_remember
    from cognee.modules.users.methods import get_user

    async with track_sync("github", credential) as reporter:
        token, _expires_at = await mint_installation_token(int(credential.provider_account_id))

        if repo_full_names is None:
            await report_sync_progress(phase="discovering", unit="repositories")
            repo_full_names = await list_installation_repositories(token)
        await report_sync_progress(
            phase="processing", completed=0, total=len(repo_full_names), unit="repositories"
        )
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
            codegraph_config={
                "repo_credentials": token,
                # Every URL is a repository: clone it, never sniff it.
                "treat_as_repository": True,
                # The code graph is the point of the sync: indexing every
                # repository's README and docs would mean LLM extraction and
                # embeddings for an installation's whole prose, on every webhook.
                "include_documents": False,
            },
            # Report a repository whose build failed as an errored item of the
            # result instead of raising.
            raise_on_error=False,
        )
        if getattr(result, "status", None) == "errored":
            logger.warning(
                "GitHub sync for %s finished with errors: %s",
                account_login,
                getattr(result, "error", None),
            )
        items = getattr(result, "items", None) or []
        failed = sum(item.get("status") == "errored" for item in items)
        processed = sum(item.get("status") != "errored" for item in items)
        if getattr(result, "status", None) == "errored":
            await reporter.finish(
                "partial" if processed else "failed",
                counts={"processed": processed, "failed": failed},
                reason="processing_error",
            )
        else:
            await report_sync_progress(
                phase="finalizing",
                completed=len(items),
                total=len(repo_full_names),
                unit="repositories",
                counts={"processed": processed, "failed": failed},
            )


__all__ = [
    "GITHUB_DATASET_PREFIX",
    "clone_url",
    "dataset_name_for_account",
    "list_installation_repositories",
    "sync_repositories",
]
