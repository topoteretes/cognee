"""Retire only repository rows recorded by this GitHub connection."""

from sqlalchemy import select

from cognee.api.v1.forget import forget
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.locks.session_lock import session_lock
from cognee.modules.data.methods.get_datasets_by_name import get_datasets_by_name
from cognee.modules.data.models import Data, Dataset
from cognee.modules.integrations.credentials import STATUS_ACTIVE, CredentialInactiveError
from cognee.modules.integrations.github.sync_settings import read_sync_settings
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.tasks.code_graph.resolve_repo import repo_key

INDEXED_REPOSITORIES_KEY = "github_indexed_repositories"


async def _current_credential(db, credential):
    result = await db.execute(
        select(IntegrationCredential)
        .where(
            IntegrationCredential.id == credential.id,
            IntegrationCredential.provider == "github",
            IntegrationCredential.user_id == credential.user_id,
            IntegrationCredential.workspace_id == credential.workspace_id,
            IntegrationCredential.provider_account_id == credential.provider_account_id,
            IntegrationCredential.status == STATUS_ACTIVE,
        )
        .with_for_update()
    )
    current = result.scalar_one_or_none()
    if current is None:
        raise CredentialInactiveError("GitHub connection is no longer active")
    return current


async def assert_tracked_selection(db, current: IntegrationCredential) -> None:
    """Refuse a selection change when existing repository content lacks a ledger.

    The caller holds the settings lock and has checked current ownership. Query
    actual content because older syncs did not consistently stamp last_synced_at.
    """
    from cognee.modules.integrations.github.sync import dataset_name_for_account

    metadata = current.provider_metadata or {}
    account_login = metadata.get("account_login") or str(current.provider_account_id)
    tracked = {
        repo_key(f"https://github.com/{full_name}.git")
        for full_name in metadata.get(INDEXED_REPOSITORIES_KEY, {}).values()
    }
    rows = (
        (
            await db.execute(
                select(Data.system_metadata)
                .join(Dataset, Dataset.id == Data.dataset_id)
                .where(
                    Dataset.name == dataset_name_for_account(account_login),
                    Dataset.owner_id == current.user_id,
                    Data.owner_id == current.user_id,
                    Data.system_metadata["source"].as_string() == "code_repo",
                )
            )
        )
        .scalars()
        .all()
    )
    for system_metadata in rows:
        url = system_metadata.get("repo_url")
        if url is None or repo_key(url) not in tracked:
            raise ValueError(
                "Run a full sync of the current selection before changing repository settings; "
                "older repository content is not tracked."
            )


async def _forget_repository_urls(db, urls: set[str], dataset_name: str, owner) -> None:
    """Delete only exact repository URL matches in this owner's named dataset.

    The graph a row owns goes with it; with a graph shared between datasets that can include
    nodes another dataset's build shares.
    """
    datasets = await get_datasets_by_name(dataset_name, owner.id)
    for dataset in datasets:
        rows = (
            await db.execute(
                select(Data.id, Data.system_metadata).where(
                    Data.dataset_id == dataset.id,
                    Data.owner_id == owner.id,
                    Data.system_metadata["source"].as_string() == "code_repo",
                )
            )
        ).all()
        for data_id, system_metadata in rows:
            url = system_metadata.get("repo_url")
            if url is not None and repo_key(url) in urls:
                await forget(data_id=data_id, dataset_id=dataset.id, user=owner)


async def record_indexed_repository(
    credential: IntegrationCredential, repository_id: str | int, full_name: str
) -> None:
    """Record a stored repository under its immutable GitHub id, built or not."""
    repository_id = str(repository_id)
    if not repository_id.isascii() or not repository_id.isdigit() or int(repository_id) <= 0:
        raise ValueError("repository_id must be a positive numeric GitHub id")
    if len(full_name.split("/")) != 2 or any(
        not part or part in {".", ".."} for part in full_name.split("/")
    ):
        raise ValueError("full_name must contain a GitHub owner and repository")
    async with (
        session_lock(str(credential.id), "github_settings"),
        get_relational_engine().get_async_session() as db,
    ):
        current = await _current_credential(db, credential)
        metadata = dict(current.provider_metadata or {})
        indexed = dict(metadata.get(INDEXED_REPOSITORIES_KEY, {}))
        previous = indexed.get(repository_id)
        if previous is not None and repo_key(f"https://github.com/{previous}.git") != repo_key(
            f"https://github.com/{full_name}.git"
        ):
            from cognee.modules.integrations.github.sync import dataset_name_for_account
            from cognee.modules.users.methods import get_user

            owner = await get_user(current.user_id)
            account_login = metadata.get("account_login") or str(current.provider_account_id)
            await _forget_repository_urls(
                db,
                {repo_key(f"https://github.com/{previous}.git")},
                dataset_name_for_account(account_login),
                owner,
            )
        indexed[repository_id] = full_name
        metadata[INDEXED_REPOSITORIES_KEY] = indexed
        current.provider_metadata = metadata
        await db.commit()


async def retire_unselected_repositories(
    credential: IntegrationCredential, selected_ids: set[str] | None, dataset_name: str, owner
) -> None:
    """Forget tracked repositories omitted from the current explicit selection.

    Legacy connections without a ledger are left untouched. The settings lock and
    fresh selection check prevent an outdated sync from deleting reselected content.
    Failed deletions retain the ledger so a subsequent full sync can retry.
    """
    if selected_ids is None:
        return
    if owner.id != credential.user_id:
        raise CredentialInactiveError("GitHub connection belongs to another owner")
    async with (
        session_lock(str(credential.id), "github_settings"),
        get_relational_engine().get_async_session() as db,
    ):
        current = await _current_credential(db, credential)
        repositories = read_sync_settings(current).repositories
        if repositories is None or set(repositories) != selected_ids:
            return
        metadata = dict(current.provider_metadata or {})
        indexed = dict(metadata.get(INDEXED_REPOSITORIES_KEY, {}))
        retired = set(indexed) - selected_ids
        if not retired:
            return
        retired_urls = {
            repo_key(f"https://github.com/{indexed[repository_id]}.git")
            for repository_id in retired
        }
        await _forget_repository_urls(db, retired_urls, dataset_name, owner)
        for repository_id in retired:
            del indexed[repository_id]
        metadata[INDEXED_REPOSITORIES_KEY] = indexed
        current.provider_metadata = metadata
        await db.commit()
