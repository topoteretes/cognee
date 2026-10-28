"""History retains every invocation while enforcing current connection ownership."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.integrations import sync_progress as progress
from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.integrations.models.IntegrationCredential import IntegrationCredential
from cognee.modules.pipelines.models.PipelineRun import PipelineRun


@pytest_asyncio.fixture
async def storage(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'history.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(IntegrationCredential.__table__.create)
        await connection.run_sync(PipelineRun.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(
        progress, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )
    yield sessions
    await engine.dispose()


@pytest_asyncio.fixture
async def credential(storage):
    row = IntegrationCredential(
        id=uuid4(),
        user_id=uuid4(),
        workspace_id=uuid4(),
        provider="github",
        provider_account_id="account",
        ciphertext=b"encrypted",
        nonce=b"nonce",
        status="active",
    )
    async with storage() as db:
        db.add(row)
        await db.commit()
    return row


@pytest.mark.asyncio
async def test_history_returns_all_invocations_newest_first(storage, credential):
    assert await progress.get_sync_history(credential) == []
    expected = []
    for index, state in enumerate(("succeeded", "partial", "failed", "cancelled")):
        async with progress.track_sync("github", credential) as reporter:
            await reporter.finish(state, counts={"items": index})
        expected.insert(0, reporter.progress.run_id)
        async with storage() as db:
            await db.execute(
                update(PipelineRun)
                .where(PipelineRun.id == reporter.progress.run_id)
                .values(
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
                )
            )
            await db.commit()
    history = await progress.get_sync_history(credential)
    assert [item.run_id for item in history] == expected
    assert [item.state for item in history] == ["cancelled", "failed", "partial", "succeeded"]
    assert all(item.execution_active is False for item in history)
    assert history[0].counts == {"items": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["user_id", "workspace_id", "status"])
async def test_history_rejects_revoked_or_transferred_connection(storage, credential, field):
    async with progress.track_sync("github", credential):
        pass
    async with storage() as db:
        await db.execute(
            update(IntegrationCredential)
            .where(IntegrationCredential.id == credential.id)
            .values(**{field: "revoked" if field == "status" else uuid4()})
        )
        await db.commit()
        current = await db.get(IntegrationCredential, credential.id)
    with pytest.raises(CredentialInactiveError):
        await progress.get_sync_history(credential)
    if field != "status":
        assert await progress.get_sync_history(current) == []


@pytest.mark.asyncio
async def test_history_isolates_accounts_and_unrelated_runs(storage, credential):
    async with progress.track_sync("github", credential):
        pass
    async with storage() as db:
        other = IntegrationCredential(
            id=uuid4(),
            user_id=credential.user_id,
            workspace_id=credential.workspace_id,
            provider="github",
            provider_account_id="other",
            ciphertext=b"x",
            nonce=b"y",
            status="active",
        )
        db.add(other)
        db.add(PipelineRun(user_id=credential.user_id, operation_name="recall"))
        await db.commit()
    assert await progress.get_sync_history(other) == []
    assert len(await progress.get_sync_history(credential)) == 1


@pytest.mark.asyncio
async def test_history_uses_live_snapshot_without_claiming_remote_activity(
    storage, credential, monkeypatch
):
    async with progress.track_sync("github", credential) as reporter:
        await reporter.report(completed=2, total=5)
        await reporter.flush()
        history = await progress.get_sync_history(credential)
        assert history[0].execution_active is True
        assert history[0].completed == 2
        monkeypatch.setattr(progress, "_active", {})
        history = await progress.get_sync_history(credential)
        assert history[0].execution_active is None
        assert history[0].phase_eta_seconds is None


@pytest.mark.asyncio
async def test_history_can_be_paged_and_is_complete_by_default(storage, credential):
    for index in range(5):
        async with progress.track_sync("github", credential) as reporter:
            await reporter.finish("succeeded", counts={"items": index})
        async with storage() as db:
            await db.execute(
                update(PipelineRun)
                .where(PipelineRun.id == reporter.progress.run_id)
                .values(
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
                )
            )
            await db.commit()
    everything = await progress.get_sync_history(credential)
    assert len(everything) == 5
    page = await progress.get_sync_history(credential, limit=2, offset=1)
    assert [item.run_id for item in page] == [item.run_id for item in everything[1:3]]
    assert await progress.get_sync_history(credential, offset=4) == everything[4:]
