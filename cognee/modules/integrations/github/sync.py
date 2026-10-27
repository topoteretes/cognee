"""Sync a GitHub installation's repositories into the code graph.

Thin orchestration over add() and cognify(): mint a fresh installation token,
resolve which repositories to index, and for each repository ``add()`` its clone
URL, then ``cognify(data_ids=...)`` exactly the rows that add returned. A build
scoped to its own repository means one that fails cannot be retried, or rolled
back, together with another. Clone reuse and the snapshot-identity skip on
unchanged repositories live in ``resolve_repo_source`` and the code graph tasks,
which is what makes re-running this on every webhook cheap and idempotent.

The token reaches git through ``codegraph_config["repo_credentials"]``, never
the URL. The indexed graph is searchable via ``SearchType.CODE`` (the code route
produces no chunks or embeddings by design), and the repositories' documents are
not ingested (``include_documents`` off), so a sync makes no LLM or embedding
calls however many repositories it covers.

One dataset per installation (``github_<org>``), not per repository: per-repo
datasets would mean one isolated database per repo under backend access control.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from cognee.modules.integrations.github.app_auth import (
    API_BASE_URL,
    _api_headers,
    mint_installation_token,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.sync_progress import report_sync_progress, track_sync

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=30)

GITHUB_DATASET_PREFIX = "github"

# A token is renewed before a clone when less than this is left. A clone that outlives
# its token fails for that repository alone.
_TOKEN_RENEWAL = timedelta(minutes=10)


def dataset_name_for_account(account_login: str) -> str:
    """The one dataset an installation's repositories land in."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", account_login).strip("_").lower()
    return f"{GITHUB_DATASET_PREFIX}_{slug or 'account'}"


def clone_url(full_name: str) -> str:
    """The credential-free https clone URL for a repository.

    Deliberately carries no token: auth travels out-of-band as
    ``repo_credentials`` (injected into git via environment config by
    ``resolve_repo_source``), so no URL-derived string — result items, logs,
    git error output — can ever leak a secret.
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
                await report_sync_progress(
                    phase="discovering",
                    completed=len(full_names),
                    unit="repositories",
                    counts={"scanned": len(full_names)},
                )
                next_link = response.links.get("next", {}).get("url")
                url = str(next_link) if next_link else None
    return full_names


async def _forget_legacy_repository(full_name: str, dataset_name: str, owner) -> None:
    """Forget this repository's rows written before repository identities changed.

    The code graph is derived from GitHub, so it is rebuilt by the add that follows.
    """
    from cognee.api.v1.forget import forget
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.tasks.code_graph.code_repo import legacy_repository_row_ids

    (dataset,) = await get_authorized_existing_datasets([dataset_name], "write", owner)
    for data_id in await legacy_repository_row_ids(clone_url(full_name), dataset.id):
        await forget(data_id=data_id, dataset_id=dataset.id, user=owner)


async def _add_and_build(full_name: str, dataset_name: str, owner, token: str) -> None:
    """Add and build one repository, rebuilding it once when it holds a legacy graph.

    A dataset an earlier release synced holds the repository under the old identity, and
    the sync cannot move to another dataset, so the legacy data is forgotten and built
    again instead of failing every sync.
    """
    from cognee.tasks.code_graph.code_repo import LegacyRepositoryError

    try:
        await _add_and_build_once(full_name, dataset_name, owner, token)
    except LegacyRepositoryError:
        logger.warning(
            "GitHub repository %s holds a graph from before repository identities changed: "
            "forgetting it and building it again.",
            full_name,
        )
        await _forget_legacy_repository(full_name, dataset_name, owner)
        await _add_and_build_once(full_name, dataset_name, owner, token)


async def _add_and_build_once(full_name: str, dataset_name: str, owner, token: str) -> None:
    """Add one repository, then build exactly the rows that add returned."""
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify.cognify import cognify
    from cognee.modules.pipelines.exceptions import PipelineRunFailedError
    from cognee.modules.pipelines.models import PipelineRunErrored

    added = await add(
        clone_url(full_name),
        dataset_name=dataset_name,
        user=owner,
        # A code-only add needs no provider: the sync makes no LLM or embedding calls.
        skip_connection_test=True,
        codegraph_config={
            "repo_credentials": token,
            # The code graph is the point of the sync: indexing every repository's
            # README and docs would mean LLM extraction and embeddings for an
            # installation's whole prose, on every webhook.
            "include_documents": False,
        },
    )
    if isinstance(added, PipelineRunErrored):
        # Git keeps the token out of every string it builds, so this is a second line of
        # defence: the error logs itself when built, so the token goes before that.
        raise PipelineRunFailedError(
            message=f"GitHub repository add failed: {added.error_class}: {added.error_message}".replace(
                token, "[secret]"
            )
        )
    if not added.data_ids:
        # cognify(data_ids=None) would build the whole dataset, and an empty list is
        # refused there: this repository fails, the others go on.
        raise PipelineRunFailedError(message="GitHub repository add returned no data to build.")
    # cognify() wraps a failure of the pipeline run in a cognee error (raise_on_error
    # is on), so the loop in sync_repositories sees one repository's failure.
    await cognify(datasets=[added.dataset_id], data_ids=added.data_ids, user=owner)


async def sync_repositories(
    credential: IntegrationCredential,
    repo_full_names: list[str] | None = None,
) -> None:
    """Index ``repo_full_names`` (default: every repo the installation covers).

    Builds each repository to completion in turn — callers are already off the
    request path (the post-install hook and webhook handling both run
    detached), so there is nothing to hand off to. A repository that cannot be
    cloned or built (a cognee error, an I/O or network failure) is logged and the sync
    moves on. A failed token mint (installation-wide) aborts the sync, and so does an
    error of another type from add() or from cognify() before its pipeline runs (for
    example a migration or database error while resolving the rows).

    The token is renewed before a clone whenever it is close to expiring, so a very large
    installation never starts a transfer near the end of one token's hour.
    """
    # Imported here, not at module top: this module is imported at API
    # startup (via the adapter registration), and cognee's package root is
    # heavyweight.
    from cognee.exceptions import CogneeApiError
    from cognee.modules.operations import scrub_error_message
    from cognee.modules.users.methods import get_user

    async with track_sync("github", credential) as reporter:
        installation_id = int(credential.provider_account_id)
        token, expires_at = await mint_installation_token(installation_id)

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

        dataset_name = dataset_name_for_account(account_login)
        logger.info(
            "Syncing %d GitHub repositories for %s into dataset %s",
            len(repo_full_names),
            account_login,
            dataset_name,
        )
        processed = failed = 0
        for position, full_name in enumerate(repo_full_names, start=1):
            if expires_at is None or expires_at - datetime.now(timezone.utc) < _TOKEN_RENEWAL:
                token, expires_at = await mint_installation_token(installation_id)
            try:
                await _add_and_build(full_name, dataset_name, owner, token)
            except (CogneeApiError, OSError, aiohttp.ClientError, asyncio.TimeoutError) as error:
                message = scrub_error_message(str(error).replace(token, "[secret]"))
                logger.error(
                    "GitHub sync for %s failed for %s: %s", account_login, full_name, message
                )
                failed += 1
            else:
                processed += 1
            await report_sync_progress(
                phase="processing",
                completed=position,
                total=len(repo_full_names),
                unit="repositories",
                counts={"processed": processed, "failed": failed},
            )
        if failed:
            logger.warning(
                "GitHub sync for %s finished with %d failed repositories", account_login, failed
            )
            await reporter.finish(
                "partial" if processed else "failed",
                counts={"processed": processed, "failed": failed},
                reason="processing_error",
            )
