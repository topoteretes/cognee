"""Promotion must enforce the sharing boundary before reading or copying data."""

import hashlib
import importlib
from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

m = importlib.import_module("cognee.api.v1.promote.promote")


@pytest.fixture
def state(monkeypatch):
    actor = SimpleNamespace(id=uuid4(), tenant_id=None, parent_user_id=uuid4())
    parent = SimpleNamespace(id=actor.parent_user_id, tenant_id=None, parent_user_id=None)
    source = SimpleNamespace(id=uuid4(), owner_id=actor.id, tenant_id=None)
    target = SimpleNamespace(id=uuid4(), owner_id=parent.id, tenant_id=None)
    content = b"A selected, reusable lesson."
    row = SimpleNamespace(
        id=uuid4(),
        dataset_id=source.id,
        raw_data_location="file:///memory.txt",
        raw_content_hash=hashlib.md5(content).hexdigest(),
        extension="txt",
        label="lesson",
        external_metadata={"tag": "verified"},
        system_metadata={"private_routing": "do-not-copy"},
    )
    rows = {(row.id, source.id): row}

    async def get_data(ident, dataset):
        return rows.get((ident, dataset))

    async def get_user(ident):
        return actor if ident == actor.id else parent

    async def authorize(user, dataset, permission):
        return source if dataset == source.id else target

    @asynccontextmanager
    async def open_file(*args):
        yield BytesIO(content)

    async def ingest(source_row, content, provenance, actor, target, target_id):
        rows[(target_id, target.id)] = SimpleNamespace(system_metadata={"promotion": provenance})
        return True

    monkeypatch.setattr(m, "_get_data", get_data)
    monkeypatch.setattr(m, "get_user", get_user)
    auth = AsyncMock(side_effect=authorize)
    monkeypatch.setattr(m, "get_authorized_dataset", auth)
    opened = []

    def read(*args):
        opened.append(args)
        return open_file(*args)

    monkeypatch.setattr(m, "open_data_file", read)
    add = AsyncMock(side_effect=ingest)
    monkeypatch.setattr(m, "_persist_copy", add)

    async def run(**overrides):
        args = {
            "source_dataset_id": source.id,
            "target_dataset_id": target.id,
            "level": "user",
            "reason": "Useful beyond this task",
            "user": actor,
        }
        args.update(overrides)
        return await m.promote(row.id, **args)

    return SimpleNamespace(**locals())


@pytest.mark.asyncio
async def test_copy_has_provenance_and_retries_do_not_overwrite(state):
    first = await state.run()
    second = await state.run()
    assert (first.status, second.status) == ("copied", "already_promoted")
    assert first.target_data_id == second.target_data_id != state.row.id
    state.add.assert_awaited_once()
    source, content, provenance, actor, target, copied_id = state.add.await_args.args
    assert source is state.row and target is state.target
    assert copied_id == first.target_data_id
    assert content == state.content
    assert provenance["source_data_id"] == str(state.row.id)
    assert provenance["promoted_by"] == str(state.actor.id)
    assert actor is state.actor
    assert [call.args[2] for call in state.auth.await_args_list[:3]] == ["read", "share", "write"]


@pytest.mark.asyncio
async def test_dry_run_reads_the_selected_snapshot_without_writes(state):
    result = await state.run(dry_run=True)
    assert result.status == "planned"
    assert len(state.opened) == 1
    state.add.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("permission", ["read", "share", "write"])
async def test_denied_permission_precedes_storage_reads(state, permission):
    original = state.auth.side_effect

    async def reject(user, dataset, requested):
        if permission == requested:
            raise PermissionError("denied")
        return await original(user, dataset, requested)

    state.auth.side_effect = reject
    with pytest.raises(PermissionError):
        await state.run()
    assert not state.opened
    state.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_cannot_skip_user_and_promote_to_team(state):
    with pytest.raises(ValueError, match="before"):
        await state.run(level="team")
    state.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_wrong_parent_rejected(state):
    state.target.owner_id = uuid4()
    with pytest.raises(ValueError, match="parent"):
        await state.run()
    state.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_source_aborts_copy(state):
    state.row.raw_content_hash = "stale"
    with pytest.raises(ValueError, match="changed"):
        await state.run()
    state.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_bounded_copy_rejects_oversized_memory(state):
    with pytest.raises(ValueError, match="max_bytes"):
        await state.run(max_bytes=3)
    state.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_ingestion_is_not_reported_as_success(state):
    state.add.side_effect = RuntimeError("persist failed")
    with pytest.raises(RuntimeError, match="persist"):
        await state.run()


@pytest.mark.asyncio
async def test_team_promotion_requires_preexisting_team_read_access(state, monkeypatch):
    state.actor.parent_user_id = None
    state.actor.tenant_id = state.source.tenant_id = state.target.tenant_id = uuid4()
    tenant = SimpleNamespace(id=state.actor.tenant_id)
    monkeypatch.setattr(m, "get_principal", AsyncMock(return_value=tenant))
    readable = AsyncMock(return_value=[])
    monkeypatch.setattr(m, "get_principal_datasets", readable)
    with pytest.raises(ValueError, match="not shared"):
        await state.run(level="team")
    state.add.assert_not_awaited()
    readable.return_value = [state.target]
    result = await state.run(level="team")
    assert result.status == "copied"


@pytest.mark.asyncio
async def test_team_promotion_rejects_other_tenant(state):
    state.actor.parent_user_id = None
    state.actor.tenant_id, state.target.tenant_id = uuid4(), uuid4()
    with pytest.raises(ValueError, match="current tenant"):
        await state.run(level="team")
    state.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_promotion_inserts_once_and_never_overwrites_edits(tmp_path, monkeypatch):
    import asyncio

    from sqlalchemy import select, update

    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )
    from cognee.modules.data.models import Data

    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'promotion.db'}")
    await engine.create_database()
    objects = {}

    class Storage:
        async def store(self, path, stream):
            objects[path] = stream.getvalue()
            await asyncio.sleep(0)
            return path

        async def remove(self, path):
            objects.pop(path, None)

    monkeypatch.setattr(m, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(m, "get_file_storage", lambda root: Storage())
    actor = SimpleNamespace(id=uuid4())
    target = SimpleNamespace(id=uuid4(), tenant_id=None)
    target_id = uuid4()
    content = b"lesson"
    revision = hashlib.md5(content).hexdigest()
    row = SimpleNamespace(
        name="original",
        label="lesson",
        extension="txt",
        mime_type="text/plain",
        loader_engine="text_loader",
        raw_content_hash=revision,
        external_metadata={},
    )
    provenance = {"source_data_id": str(uuid4()), "source_revision": revision}
    try:
        results = await asyncio.gather(
            *[m._persist_copy(row, content, provenance, actor, target, target_id) for _ in range(2)]
        )
        assert sorted(results) == [False, True]
        assert len(objects) == 1
        async with engine.get_async_session() as session:
            await session.execute(
                update(Data).where(Data.id == target_id).values(name="user-edited")
            )
            await session.commit()
        assert await m._persist_copy(row, content, provenance, actor, target, target_id) is False
        async with engine.get_async_session() as session:
            rows = (await session.execute(select(Data))).scalars().all()
            assert len(rows) == 1 and rows[0].name == "user-edited"
        assert len(objects) == 1
    finally:
        await engine.engine.dispose()
