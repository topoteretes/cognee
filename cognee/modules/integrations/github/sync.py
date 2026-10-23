"""Sync a GitHub installation's repositories into the code graph.

Thin orchestration over the existing ``remember(content_type="code")`` path:
mint a fresh installation token, resolve which repositories to index, and
hand authenticated clone URLs to the code-graph pipeline. All the heavy
lifting — clone reuse, snapshot-identity skip on unchanged repos, per-repo
failure isolation (``raise_on_error=False``) — already lives in
``resolve_repo_source`` and ``remember``, which is what makes re-running this
on every webhook cheap and idempotent.

The indexed graph is searchable via ``SearchType.CODE`` (the code route
produces no chunks or embeddings by design); ``index_vectors`` stays off.

One dataset per installation (``github_<org>``), not per repository —
``remember`` already accepts a list of repo specs targeting one dataset, and
per-repo datasets would mean one isolated database per repo under backend
access control.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from cognee.modules.integrations.github.app_auth import (
    API_BASE_URL,
    _api_headers,
    mint_installation_token,
)
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.integrations.sync_progress import (
    report_sync_progress,
    report_sync_resource,
    track_sync,
)
from cognee.tasks.code_graph.resolve_repo import check_branch_name

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=30)
_TOKEN_RENEWAL = timedelta(minutes=5)

GITHUB_DATASET_PREFIX = "github"
SELECTED_REPOSITORIES_KEY = "selected_repositories"
REPOSITORY_BRANCHES_KEY = "repository_branches"


def dataset_name_for_account(account_login: str) -> str:
    """The one dataset an installation's repositories land in."""
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", account_login).strip("_").lower()
    return f"{GITHUB_DATASET_PREFIX}_{slug or 'account'}"


def dataset_name_for_credential(credential: IntegrationCredential) -> str:
    """Use the same installation dataset for ingestion and connection status."""
    account_login = (credential.provider_metadata or {}).get("account_login") or str(
        credential.provider_account_id
    )
    return dataset_name_for_account(account_login)


def clone_url(full_name: str) -> str:
    """The credential-free https clone URL for a repository.

    Deliberately carries no token: auth travels out-of-band as
    ``repo_credentials`` (injected into git via environment config by
    ``resolve_repo_source``), so no URL-derived string — clone slugs, result
    items, logs, git error output — can ever leak a secret.
    """
    return f"https://github.com/{full_name}.git"


async def get_pages(url: str, token: str, failure: str) -> AsyncIterator[Any]:
    """Each page's JSON of a paginated GitHub list, following ``Link: next``."""
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        while url:
            async with session.get(url, headers=_api_headers(token)) as response:
                if response.status != 200:
                    raise RuntimeError(f"{failure}: HTTP {response.status}")
                payload = await response.json()
                next_link = response.links.get("next", {}).get("url")
            url = str(next_link) if next_link else None
            yield payload


async def list_installation_repositories(token: str) -> list[str]:
    """Full names (``org/repo``) of every repository the installation covers."""
    full_names: list[str] = []
    async for payload in get_pages(
        f"{API_BASE_URL}/installation/repositories?per_page=100",
        token,
        "GitHub installation repository listing failed",
    ):
        full_names.extend(repo["full_name"] for repo in payload.get("repositories", []) if repo)
        await report_sync_progress(
            phase="discovering",
            completed=len(full_names),
            unit="repositories",
            counts={"scanned": len(full_names)},
        )
    return full_names


async def sync_repositories(
    credential: IntegrationCredential,
    repo_full_names: list[str] | None = None,
) -> None:
    """Index accessible repositories within the stored selection.

    An absent/null selection includes all installation repositories. An empty
    selection includes none. Webhook-triggered syncs respect the same selection.

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

    async with track_sync("github", credential) as reporter:
        selected = (credential.provider_metadata or {}).get(SELECTED_REPOSITORIES_KEY)
        if selected is not None and (
            not isinstance(selected, list)
            or any(not isinstance(name, str) or not name.strip() for name in selected)
        ):
            raise ValueError("GitHub repository selection must be a list of repository names")
        token, expires_at = await mint_installation_token(int(credential.provider_account_id))

        if repo_full_names is None:
            await report_sync_progress(phase="discovering", unit="repositories")
            repo_full_names = await list_installation_repositories(token)
        if selected is not None:
            selected_names = set(selected)
            repo_full_names = [name for name in repo_full_names if name in selected_names]
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
            dataset_name_for_credential(credential),
        )

        branches = (credential.provider_metadata or {}).get(REPOSITORY_BRANCHES_KEY) or {}
        if not isinstance(branches, dict):
            raise ValueError("GitHub branch selection must be a repository-to-branch mapping")
        for full_name in repo_full_names:
            await report_sync_resource(
                clone_url(full_name),
                branch=branches.get(full_name),
                state="queued",
                phase="queued",
                completed=0,
            )
        processed = failed = 0
        finished: set[str] = set()
        try:
            for position, full_name in enumerate(repo_full_names, start=1):
                source = clone_url(full_name)
                branch = branches.get(full_name)
                # A token lasts an hour and each repository is a full code graph build.
                if (
                    expires_at is not None
                    and expires_at - datetime.now(timezone.utc) < _TOKEN_RENEWAL
                ):
                    token, expires_at = await mint_installation_token(
                        int(credential.provider_account_id)
                    )
                try:
                    if branch is not None:
                        check_branch_name(branch)
                except ValueError:
                    # A stored branch that is not a valid name: this repository fails,
                    # the others still sync.
                    logger.warning(
                        "GitHub sync skipped %s: its stored branch is invalid", full_name
                    )
                    errored = True
                else:
                    result = await cognee_remember(
                        source,
                        dataset_name=dataset_name_for_credential(credential),
                        user=owner,
                        content_type="code",
                        repo_credentials=token,
                        **({"branch": branch} if branch is not None else {}),
                        raise_on_error=False,
                    )
                    errored = result.status == "errored"
                if errored:
                    failed += 1
                else:
                    processed += 1
                finished.add(full_name)
                await report_sync_resource(
                    source,
                    branch=branch,
                    state="failed" if errored else "succeeded",
                    phase="failed" if errored else "complete",
                    completed=None if errored else 3,
                )
                await report_sync_progress(
                    phase="processing",
                    completed=position,
                    total=len(repo_full_names),
                    unit="repositories",
                    counts={"processed": processed, "failed": failed},
                )
        except BaseException as stop:
            # A stop part-way (a cancel, a failed token renewal) must not leave the
            # repository it hit, or the ones after it, looking queued or running.
            leftover = "cancelled" if isinstance(stop, asyncio.CancelledError) else "failed"
            for full_name in repo_full_names:
                if full_name not in finished:
                    await report_sync_resource(
                        clone_url(full_name),
                        branch=branches.get(full_name),
                        state=leftover,
                        phase=leftover,
                    )
            raise
        if failed:
            await reporter.finish(
                "partial" if processed else "failed",
                counts={"processed": processed, "failed": failed},
                reason="processing_error",
            )
