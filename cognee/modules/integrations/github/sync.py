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
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from functools import partial
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


async def list_installation_repository_records(token: str) -> list[dict[str, Any]]:
    """Repository records, including stable ids, names and default branches."""
    full_names: list[dict[str, Any]] = []
    url: str | None = f"{API_BASE_URL}/installation/repositories?per_page=100"
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        while url:
            async with session.get(url, headers=_api_headers(token)) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"GitHub installation repository listing failed: HTTP {response.status}"
                    )
                payload: dict[str, Any] = await response.json()
                full_names.extend(repo for repo in payload.get("repositories", []) if repo)
                await report_sync_progress(
                    phase="discovering",
                    completed=len(full_names),
                    unit="repositories",
                    counts={"scanned": len(full_names)},
                )
                next_link = response.links.get("next", {}).get("url")
                url = str(next_link) if next_link else None
    return full_names


async def _forget_legacy_repository(full_name: str, dataset_name: str, owner) -> list:
    """Forget this repository's rows written before repository identities changed.

    Returns the forgotten ids, empty when the dataset holds none. The code graph is
    derived from GitHub, so the add that follows rebuilds it. The graph a row owns goes
    with it; with a graph shared between datasets that can include nodes another
    dataset's legacy build shares.
    """
    from cognee.api.v1.forget import forget
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.tasks.code_graph.code_repo import legacy_repository_row_ids

    datasets = await get_authorized_existing_datasets([dataset_name], "write", owner, strict=True)
    data_ids = await legacy_repository_row_ids(clone_url(full_name), datasets[0].id)
    for data_id in data_ids:
        receipt = await forget(data_id=data_id, dataset_id=datasets[0].id, user=owner)
        logger.info(
            "Forgot the legacy row %s of GitHub repository %s: %s", data_id, full_name, receipt
        )
    return data_ids


async def _add_and_build(
    full_name: str, dataset_name: str, owner, token: str, policy=None, on_added=None
) -> None:
    """Add and build one repository, once more when that can heal it.

    A dataset an earlier release synced holds the repository under the old identity, and
    the sync cannot move to another dataset, so its legacy rows are forgotten and the
    repository is built again instead of failing every sync. A repository that moved
    after this add resolved it, whether add() refuses its stale manifest or cognify finds
    the checkout changed (an overlapping sync or an upstream change), is added and built
    again too. Anything else, or a second failure, fails the repository.

    ``on_added`` runs once add() has stored the repository, before it is built: the content
    exists from then on, whether or not the build succeeds.
    """
    from cognee.modules.pipelines.exceptions import CognifyFailedError
    from cognee.tasks.code_graph.code_repo import LegacyRepositoryError, RepositoryChangedError

    try:
        await _add_and_build_once(full_name, dataset_name, owner, token, policy, on_added)
    except LegacyRepositoryError:
        forgotten = await _forget_legacy_repository(full_name, dataset_name, owner)
        if not forgotten:
            raise
        logger.warning(
            "GitHub repository %s: forgot %d row(s) written before repository identities "
            "changed and building it again.",
            full_name,
            len(forgotten),
        )
        await _add_and_build_once(full_name, dataset_name, owner, token, policy, on_added)
    except (RepositoryChangedError, CognifyFailedError) as error:
        # With RAISE_INCREMENTAL_LOADING_ERRORS off the pipeline reports the error as
        # an errored run, which cognify raises as CognifyFailedError.
        if not isinstance(error, RepositoryChangedError) and (
            error.error_class != "RepositoryChangedError"
        ):
            raise
        logger.warning(
            "GitHub repository %s changed while it was being built: adding and building it again.",
            full_name,
        )
        await _add_and_build_once(full_name, dataset_name, owner, token, policy, on_added)


async def _add_and_build_once(
    full_name: str, dataset_name: str, owner, token: str, policy=None, on_added=None
) -> None:
    """Add one repository, then build exactly the rows that add returned."""
    from cognee.api.v1.add import add
    from cognee.api.v1.cognify.cognify import cognify
    from cognee.modules.pipelines.exceptions import PipelineRunFailedError
    from cognee.modules.pipelines.models import PipelineRunErrored
    from cognee.tasks.code_graph.code_repo import RepositoryChangedError

    options = {}
    if policy is not None:
        options = {"include_paths": policy.include_paths, "exclude_paths": policy.exclude_paths}
    added = await add(
        clone_url(full_name),
        dataset_name=dataset_name,
        user=owner,
        **({"branch": policy.branch} if policy is not None and policy.branch else {}),
        # A code-only add needs no provider: the sync makes no LLM or embedding calls.
        skip_connection_test=True,
        codegraph_config={
            "repo_credentials": token,
            # The code graph is the point of the sync: indexing every repository's
            # README and docs would mean LLM extraction and embeddings for an
            # installation's whole prose, on every webhook.
            "include_documents": False,
            **options,
        },
    )
    if isinstance(added, PipelineRunErrored):
        if added.error_class == "RepositoryChangedError":
            # With RAISE_INCREMENTAL_LOADING_ERRORS off the pipeline reports the error as
            # an errored run: raise it as add() otherwise does, for the retry.
            raise RepositoryChangedError()
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
    if on_added is not None:
        await on_added()
    # cognify() wraps a failure of the pipeline run in a cognee error (raise_on_error
    # is on), so the loop in sync_repositories sees one repository's failure.
    await cognify(datasets=[added.dataset_id], data_ids=added.data_ids, user=owner)


async def _sync_repositories(
    credential: IntegrationCredential,
    repo_full_names: list[str],
    *,
    policies: dict | None = None,
    repository_ids: dict[str, str] | None = None,
    retire_selection: set[str] | None = None,
    progress=None,
) -> str:
    """Index ``repo_full_names``, the repositories the saved policy selects for this run.

    Builds each repository to completion in turn — callers are already off the
    request path (the post-install hook and webhook handling both run
    detached), so there is nothing to hand off to. A repository that cannot be
    cloned or built (a cognee error, an I/O or network failure) is logged and the sync
    moves on. A failed token mint (installation-wide) aborts the sync, and so does an
    error of another type from add() or from cognify() before its pipeline runs (for
    example a migration or database error while resolving the rows), and so does an
    error of its own from forgetting a legacy repository.

    The token is renewed before a clone whenever it is close to expiring, so a very large
    installation never starts a transfer near the end of one token's hour.
    """
    # Imported here, not at module top: this module is imported at API
    # startup (via the adapter registration), and cognee's package root is
    # heavyweight.
    from cognee.exceptions import CogneeApiError
    from cognee.modules.operations import scrub_error_message
    from cognee.modules.users.methods import get_user

    context = track_sync("github", credential) if progress is None else nullcontext(progress)
    async with context as reporter:
        from cognee.modules.integrations.credentials import require_active_credential

        if repository_ids is not None:
            credential = await require_active_credential(credential)
        installation_id = int(credential.provider_account_id)
        token, expires_at = await mint_installation_token(installation_id)

        await report_sync_progress(
            phase="processing", completed=0, total=len(repo_full_names), unit="repositories"
        )
        account_login = (credential.provider_metadata or {}).get("account_login") or str(
            credential.provider_account_id
        )
        owner = await get_user(credential.user_id)

        dataset_name = dataset_name_for_account(account_login)
        # Retirement goes first: it follows the saved selection, not how this run's builds go.
        if retire_selection is not None:
            from cognee.modules.integrations.github.retire import retire_unselected_repositories

            await retire_unselected_repositories(credential, retire_selection, dataset_name, owner)
        if not repo_full_names:
            logger.info(
                "GitHub installation %s has no repositories to sync", credential.provider_account_id
            )
            return "nothing_selected"

        logger.info(
            "Syncing %d GitHub repositories for %s into dataset %s",
            len(repo_full_names),
            account_login,
            dataset_name,
        )
        processed = failed = 0
        for position, full_name in enumerate(repo_full_names, start=1):
            if repository_ids is not None:
                credential = await require_active_credential(credential)
            if expires_at is None or expires_at - datetime.now(timezone.utc) < _TOKEN_RENEWAL:
                token, expires_at = await mint_installation_token(installation_id)
            on_added = None
            if repository_ids is not None:
                from cognee.modules.integrations.github.retire import record_indexed_repository

                on_added = partial(
                    record_indexed_repository, credential, repository_ids[full_name], full_name
                )

            try:
                await _add_and_build(
                    full_name,
                    dataset_name,
                    owner,
                    token,
                    policies.get(full_name) if policies else None,
                    on_added,
                )
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
            return "partial" if processed else "failed"
        return "completed"


async def sync_repositories(
    credential: IntegrationCredential,
    repo_full_names: list[str] | None = None,
    *,
    trigger: str = "manual",
    now: datetime | None = None,
    pushed_branch: str | None = None,
) -> str:
    """Serialize queued syncs and recheck policy after waiting, without dropping pushes.

    The SDK lock is process-local. Run a single worker for each installation.
    Manual sync is explicit and remains available while automatic sync is paused.
    """
    from cognee.infrastructure.locks.session_lock import session_lock
    from cognee.modules.integrations.credentials import require_active_credential
    from cognee.modules.integrations.github.sync_settings import (
        GithubRepositorySettings,
        claim_scheduled_sync,
        read_sync_settings,
    )

    if trigger not in {"manual", "push", "scheduled", "initial", "settings"}:
        raise ValueError("Unknown GitHub sync trigger")
    async with session_lock(str(credential.id), "github_sync"):
        credential = await require_active_credential(credential)
        settings = read_sync_settings(credential)
        if trigger != "manual" and not settings.enabled:
            return "paused"
        if trigger == "push" and settings.mode != "push":
            return "mode_skipped"
        if trigger == "initial" and (
            settings.mode == "manual" or settings.first_sync == "scheduled"
        ):
            return "mode_skipped"
        if trigger == "scheduled":
            credential, outcome = await claim_scheduled_sync(
                credential, now or datetime.now(timezone.utc)
            )
            if outcome != "claimed":
                return outcome
            settings = read_sync_settings(credential)
        from cognee.modules.integrations.credentials import record_sync_result

        try:
            async with track_sync("github", credential) as progress:
                covers_all = repo_full_names is None
                token, _ = await mint_installation_token(int(credential.provider_account_id))
                await report_sync_progress(phase="discovering", unit="repositories")
                records = await list_installation_repository_records(token)
                if settings.repositories is not None:
                    missing = set(settings.repositories) - {str(repo["id"]) for repo in records}
                    if missing:
                        # An admin removed them from the installation. The rest still sync;
                        # the saved selection and the ledger keep them, since deleting indexed
                        # data on that signal would be a silent, destructive surprise.
                        logger.warning(
                            "Selected GitHub repositories %s are not available to installation %s; "
                            "syncing the others.",
                            ", ".join(sorted(missing)),
                            credential.provider_account_id,
                        )
                policies = {}
                repository_ids = {}
                for repo in records:
                    name = repo["full_name"]
                    if repo_full_names is not None and name not in repo_full_names:
                        continue
                    if settings.repositories is not None:
                        policy = settings.repositories.get(str(repo["id"]))
                        if policy is None:
                            continue
                    else:
                        policy = GithubRepositorySettings()
                    branch = policy.branch or repo.get("default_branch")
                    if not branch:
                        raise ValueError(f"GitHub repository {name} has no default branch")
                    if pushed_branch is not None and pushed_branch != branch:
                        continue
                    policies[name] = policy
                    repository_ids[name] = str(repo["id"])
                outcome = await _sync_repositories(
                    credential,
                    list(policies),
                    progress=progress,
                    policies=policies,
                    repository_ids=repository_ids,
                    retire_selection=set(settings.repositories)
                    if covers_all and settings.repositories is not None
                    else None,
                )
        except BaseException:
            await record_sync_result(credential, status="degraded")
            raise
        if outcome != "nothing_selected":
            # No attempt was made for an empty selection, so the connection's last attempt stands.
            await record_sync_result(
                credential, status="degraded" if outcome in {"partial", "failed"} else "ok"
            )
        return outcome
