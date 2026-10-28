"""Repository retirement is scoped to the connection ledger and exact dataset."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.data.models import Data, Dataset
from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.integrations.github import retire
from cognee.modules.integrations.github.sync_settings import SETTINGS_KEY
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential


@pytest_asyncio.fixture
async def setup(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'retire.db'}")
    async with engine.begin() as connection:
        for model in (IntegrationCredential, Dataset, Data):
            await connection.run_sync(model.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        retire, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    owner = SimpleNamespace(id=uuid4())
    credential = IntegrationCredential(
        id=uuid4(),
        user_id=owner.id,
        workspace_id=uuid4(),
        provider="github",
        provider_account_id="1",
        ciphertext=b"x",
        nonce=b"y",
        status="active",
        provider_metadata={SETTINGS_KEY: {"repositories": {}}},
    )
    dataset = Dataset(id=uuid4(), name="github_org", owner_id=owner.id)
    async with sessions() as db:
        db.add_all([credential, dataset])
        await db.commit()
    monkeypatch.setattr(retire, "get_datasets_by_name", AsyncMock(return_value=[dataset]))
    monkeypatch.setattr(retire, "forget", AsyncMock())
    yield sessions, credential, dataset, owner
    await engine.dispose()


async def ledger(sessions, credential):
    async with sessions() as db:
        row = await db.get(IntegrationCredential, credential.id)
        return row.provider_metadata.get(retire.INDEXED_REPOSITORIES_KEY, {})


@pytest.mark.asyncio
async def test_empty_selection_deletes_only_tracked_exact_dataset_code_rows(setup):
    sessions, credential, dataset, owner = setup
    await retire.record_indexed_repository(credential, "1", "Org/Repo")
    target = Data(
        id=uuid4(),
        dataset_id=dataset.id,
        owner_id=owner.id,
        system_metadata={"source": "code_repo", "repo_url": "https://github.com/org/repo.git"},
    )
    async with sessions() as db:
        db.add_all(
            [
                target,
                Data(dataset_id=uuid4(), owner_id=owner.id, system_metadata=target.system_metadata),
                Data(
                    dataset_id=dataset.id, owner_id=uuid4(), system_metadata=target.system_metadata
                ),
                Data(
                    dataset_id=dataset.id,
                    owner_id=owner.id,
                    system_metadata={
                        "source": "code_repo",
                        "repo_url": "https://github.com/org/manual.git",
                    },
                ),
                Data(
                    dataset_id=dataset.id,
                    owner_id=owner.id,
                    system_metadata={
                        "source": "text",
                        "repo_url": "https://github.com/org/repo.git",
                    },
                ),
            ]
        )
        await db.commit()
    await retire.retire_unselected_repositories(credential, set(), dataset.name, owner)
    retire.forget.assert_awaited_once_with(data_id=target.id, dataset_id=dataset.id, user=owner)
    assert await ledger(sessions, credential) == {}


@pytest.mark.asyncio
async def test_failure_keeps_ledger_for_retry(setup):
    sessions, credential, dataset, owner = setup
    await retire.record_indexed_repository(credential, 1, "org/repo")
    async with sessions() as db:
        db.add(
            Data(
                dataset_id=dataset.id,
                owner_id=owner.id,
                system_metadata={
                    "source": "code_repo",
                    "repo_url": "https://github.com/org/repo.git",
                },
            )
        )
        await db.commit()
    retire.forget.side_effect = RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"):
        await retire.retire_unselected_repositories(credential, set(), dataset.name, owner)
    assert await ledger(sessions, credential) == {"1": "org/repo"}


@pytest.mark.asyncio
async def test_fresh_selection_prevents_stale_removal(setup):
    sessions, credential, dataset, owner = setup
    await retire.record_indexed_repository(credential, "1", "org/repo")
    async with sessions() as db:
        row = await db.get(IntegrationCredential, credential.id)
        row.provider_metadata = {**row.provider_metadata, SETTINGS_KEY: {"repositories": {"1": {}}}}
        await db.commit()
    await retire.retire_unselected_repositories(credential, set(), dataset.name, owner)
    retire.forget.assert_not_awaited()
    assert await ledger(sessions, credential) == {"1": "org/repo"}


@pytest.mark.asyncio
async def test_legacy_without_ledger_and_all_selection_do_not_delete(setup):
    sessions, credential, dataset, owner = setup
    await retire.retire_unselected_repositories(credential, set(), dataset.name, owner)
    await retire.record_indexed_repository(credential, "1", "org/repo")
    await retire.retire_unselected_repositories(credential, None, dataset.name, owner)
    retire.get_datasets_by_name.assert_not_awaited()
    retire.forget.assert_not_awaited()
    assert await ledger(sessions, credential) == {"1": "org/repo"}


@pytest.mark.asyncio
async def test_missing_dataset_is_noop(setup):
    sessions, credential, dataset, owner = setup
    await retire.record_indexed_repository(credential, "1", "org/repo")
    retire.get_datasets_by_name.return_value = []
    await retire.retire_unselected_repositories(credential, set(), dataset.name, owner)
    retire.forget.assert_not_awaited()
    assert await ledger(sessions, credential) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["user_id", "workspace_id", "status"])
async def test_transfer_or_revocation_blocks_record_and_retirement(setup, field):
    sessions, credential, dataset, owner = setup
    async with sessions() as db:
        await db.execute(
            update(IntegrationCredential)
            .where(IntegrationCredential.id == credential.id)
            .values(**{field: "revoked" if field == "status" else uuid4()})
        )
        await db.commit()
    with pytest.raises(CredentialInactiveError):
        await retire.record_indexed_repository(credential, "1", "org/repo")
    with pytest.raises(CredentialInactiveError):
        await retire.retire_unselected_repositories(credential, set(), dataset.name, owner)
    retire.forget.assert_not_awaited()


@pytest.mark.asyncio
async def test_deletion_can_commit_in_its_own_session(setup, monkeypatch):
    from sqlalchemy import delete

    sessions, credential, dataset, owner = setup
    await retire.record_indexed_repository(credential, "1", "org/repo")
    async with sessions() as db:
        db.add(
            Data(
                dataset_id=dataset.id,
                owner_id=owner.id,
                system_metadata={
                    "source": "code_repo",
                    "repo_url": "https://github.com/org/repo.git",
                },
            )
        )
        await db.commit()

    async def remove(*, data_id, dataset_id, user):
        async with sessions() as db:
            await db.execute(delete(Data).where(Data.id == data_id))
            await db.commit()

    monkeypatch.setattr(retire, "forget", remove)
    await retire.retire_unselected_repositories(credential, set(), dataset.name, owner)
    assert await ledger(sessions, credential) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("tracked", [False, True])
async def test_selection_change_checks_actual_repository_content(setup, tracked):
    sessions, credential, dataset, owner = setup
    async with sessions() as db:
        current = await db.get(IntegrationCredential, credential.id)
        current.provider_metadata = {**current.provider_metadata, "account_login": "org"}
        db.add(
            Data(
                dataset_id=dataset.id,
                owner_id=owner.id,
                system_metadata={
                    "source": "code_repo",
                    "repo_url": "https://github.com/Org/Repo.git",
                },
            )
        )
        await db.commit()
    if tracked:
        await retire.record_indexed_repository(credential, "1", "org/repo")
    async with sessions() as db:
        current = await db.get(IntegrationCredential, credential.id)
        if tracked:
            await retire.assert_tracked_selection(db, current)
        else:
            with pytest.raises(ValueError, match="Run a full sync"):
                await retire.assert_tracked_selection(db, current)


@pytest.mark.asyncio
async def test_new_selection_allowed_without_existing_repository_rows(setup):
    sessions, credential, dataset, owner = setup
    async with sessions() as db:
        current = await db.get(IntegrationCredential, credential.id)
        current.provider_metadata = {**current.provider_metadata, "account_login": "org"}
        db.add_all(
            [
                Data(dataset_id=dataset.id, owner_id=owner.id, system_metadata={"source": "text"}),
                Data(
                    dataset_id=uuid4(),
                    owner_id=owner.id,
                    system_metadata={
                        "source": "code_repo",
                        "repo_url": "https://github.com/org/other.git",
                    },
                ),
            ]
        )
        await db.commit()
        await retire.assert_tracked_selection(db, current)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_rename_retires_only_previous_url_and_preserves_ledger_on_failure(
    setup, monkeypatch, fails
):
    from sqlalchemy import delete, select

    from cognee.modules.users import methods

    sessions, credential, dataset, owner = setup
    monkeypatch.setattr(methods, "get_user", AsyncMock(return_value=owner))
    await retire.record_indexed_repository(credential, "1", "org/old")
    rows = [
        Data(
            id=uuid4(),
            dataset_id=dataset.id,
            owner_id=owner.id,
            system_metadata={
                "source": "code_repo",
                "repo_url": f"https://github.com/org/{name}.git",
            },
        )
        for name in ("old", "new", "manual")
    ]
    async with sessions() as db:
        current = await db.get(IntegrationCredential, credential.id)
        current.provider_metadata = {**current.provider_metadata, "account_login": "org"}
        db.add_all(rows)
        await db.commit()

    async def remove(*, data_id, dataset_id, user):
        assert data_id == rows[0].id
        assert dataset_id == dataset.id
        assert user is owner
        if fails:
            raise RuntimeError("delete failed")
        async with sessions() as db:
            await db.execute(delete(Data).where(Data.id == data_id))
            await db.commit()

    monkeypatch.setattr(retire, "forget", remove)
    if fails:
        with pytest.raises(RuntimeError, match="delete failed"):
            await retire.record_indexed_repository(credential, "1", "org/new")
        assert await ledger(sessions, credential) == {"1": "org/old"}
    else:
        await retire.record_indexed_repository(credential, "1", "org/new")
        assert await ledger(sessions, credential) == {"1": "org/new"}
        async with sessions() as db:
            remaining = set((await db.execute(select(Data.id))).scalars().all())
        assert remaining == {rows[1].id, rows[2].id}
