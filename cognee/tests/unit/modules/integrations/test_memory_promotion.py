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


@pytest.mark.asyncio
@pytest.mark.parametrize("personal_workspace", [False, True])
async def test_real_permissions_and_storage_agent_to_user_to_team(
    tmp_path, monkeypatch, personal_workspace
):
    """Exercise public promotion with real ACL joins, SQLite and local files."""
    from sqlalchemy import func, select, update

    from cognee.base_config import get_base_config
    from cognee.infrastructure.databases.relational import (
        get_relational_config,
        get_relational_engine,
    )
    from cognee.infrastructure.files.storage import get_file_storage
    from cognee.modules.data.models import Data, Dataset
    from cognee.modules.users.exceptions import PermissionDeniedError
    from cognee.modules.users.models import ACL, Permission, Role, Tenant, User
    from cognee.modules.users.permissions.methods.authorized_get_principal_datasets import (
        authorized_get_principal_datasets,
    )
    from cognee.tasks.documents.classify_documents import classify_documents

    monkeypatch.setattr(get_relational_config(), "db_path", str(tmp_path))
    monkeypatch.setattr(get_relational_config(), "db_name", "promotion-real.db")
    monkeypatch.setattr(get_relational_config(), "db_provider", "sqlite")
    monkeypatch.setattr(get_base_config(), "data_root_directory", str(tmp_path / "files"))
    engine = get_relational_engine()
    await engine.create_database()
    tenant = Tenant(id=uuid4(), name="test team")
    parent = User(
        id=uuid4(),
        email="parent@example.test",
        hashed_password="unused",
        tenant_id=tenant.id,
        tenants=[tenant],
    )
    agent = User(
        id=uuid4(),
        email="agent@example.test",
        hashed_password="unused",
        parent_user_id=parent.id,
        tenant_id=tenant.id,
        tenants=[tenant],
    )
    role = Role(id=uuid4(), name="publisher", tenant_id=tenant.id, users=[parent])
    datasets = [
        Dataset(id=uuid4(), name=name, owner_id=owner, tenant_id=tenant.id)
        for name, owner in [("agent", agent.id), ("user", parent.id), ("team", parent.id)]
    ]
    source, personal, shared = datasets
    if personal_workspace:
        source.tenant_id = personal.tenant_id = parent.tenant_id = agent.tenant_id = None
    permissions = {name: Permission(id=uuid4(), name=name) for name in ("read", "share", "write")}

    def acl(principal, dataset, name):
        return ACL(
            principal_id=principal.id, dataset_id=dataset.id, permission_id=permissions[name].id
        )

    content = b"One verified lesson selected by the agent."
    storage = get_file_storage(get_base_config().data_root_directory)
    location = await storage.store("source.txt", BytesIO(content))
    original = Data(
        id=uuid4(),
        dataset_id=source.id,
        owner_id=agent.id,
        tenant_id=source.tenant_id,
        name="lesson",
        extension="txt",
        mime_type="text/plain",
        loader_engine="text_loader",
        raw_data_location=location,
        original_data_location=location,
        content_hash=hashlib.md5(content).hexdigest(),
        raw_content_hash=hashlib.md5(content).hexdigest(),
        pipeline_status={},
        external_metadata={"reviewed": True},
    )
    try:
        async with engine.get_async_session() as session:
            session.add_all(
                [tenant, parent, agent, role, *datasets, *permissions.values(), original]
            )
            await session.flush()
            session.add_all(
                [
                    acl(agent, source, "read"),
                    acl(agent, source, "share"),
                    acl(parent, personal, "read"),
                    acl(parent, personal, "share"),
                    acl(role, shared, "write"),
                    acl(tenant, shared, "read"),
                ]
            )
            await session.commit()
        kwargs = {
            "source_dataset_id": source.id,
            "target_dataset_id": personal.id,
            "level": "user",
            "reason": "Reusable lesson",
            "user": agent,
        }
        with pytest.raises(PermissionDeniedError):
            await m.promote(original.id, **kwargs)
        async with engine.get_async_session() as session:
            assert await session.scalar(select(func.count()).select_from(Data)) == 1
            session.add(acl(agent, personal, "write"))
            await session.commit()
        first = await m.promote(original.id, **kwargs)
        assert first.status == "copied"
        assert (await m.promote(original.id, **kwargs)).status == "already_promoted"
        if personal_workspace:
            async with engine.get_async_session() as session:
                await session.execute(
                    update(User).where(User.id == parent.id).values(tenant_id=tenant.id)
                )
                await session.commit()
        effective = await authorized_get_principal_datasets(parent.id, "write", parent.id)
        assert shared.id in {dataset.id for dataset in effective}
        second = await m.promote(
            first.target_data_id,
            source_dataset_id=personal.id,
            target_dataset_id=shared.id,
            level="team",
            reason="Team approved lesson",
            user=parent,
        )
        assert second.status == "copied"
        copied = await m._get_data(second.target_data_id, shared.id)
        assert copied.system_metadata["promotion"]["previous"]["source_data_id"] == str(original.id)
        async with m.open_data_file(copied.raw_data_location, "rb") as stream:
            assert stream.read() == content
        documents = await classify_documents([copied])
        assert documents[0].id == second.target_data_id
        async with engine.get_async_session() as session:
            assert await session.scalar(select(func.count()).select_from(Data)) == 3
    finally:
        await engine.engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("can_verify", [True, False])
async def test_commit_ack_failure_never_removes_published_storage(
    tmp_path, monkeypatch, can_verify
):
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )
    from cognee.modules.data.models import Data

    engine = SQLAlchemyAdapter(f"sqlite+aiosqlite:///{tmp_path / 'ack.db'}")
    await engine.create_database()
    objects = {}

    class Storage:
        async def store(self, path, stream):
            objects[path] = stream.getvalue()
            return path

        async def remove(self, path):
            objects.pop(path, None)

    class InterruptedEngine:
        calls = 0

        @asynccontextmanager
        async def get_async_session(self):
            self.calls += 1
            first = self.calls == 1
            if not first and not can_verify:
                raise ConnectionError("database unavailable")
            async with engine.get_async_session() as session:
                yield session
            if first:
                raise ConnectionError("commit acknowledgement lost")

    monkeypatch.setattr(m, "get_relational_engine", lambda: interrupted_engine)
    interrupted_engine = InterruptedEngine()
    monkeypatch.setattr(m, "get_file_storage", lambda root: Storage())
    actor = SimpleNamespace(id=uuid4())
    target = SimpleNamespace(id=uuid4(), tenant_id=None)
    target_id = uuid4()
    row = SimpleNamespace(
        name="lesson",
        label=None,
        extension="txt",
        mime_type="text/plain",
        loader_engine="text_loader",
        raw_content_hash=hashlib.md5(b"lesson").hexdigest(),
        external_metadata={},
    )
    provenance = {"source_revision": hashlib.sha256(b"lesson").hexdigest()}
    try:
        if can_verify:
            assert (
                await m._persist_copy(row, b"lesson", provenance, actor, target, target_id) is True
            )
        else:
            with pytest.raises(ConnectionError, match="acknowledgement"):
                await m._persist_copy(row, b"lesson", provenance, actor, target, target_id)
        async with engine.get_async_session() as session:
            copied = (await session.execute(select(Data))).scalar_one()
        assert objects[copied.raw_data_location] == b"lesson"
    finally:
        await engine.engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["own_personal", "foreign_personal", "other_team", "user_level"])
async def test_personal_source_exception_is_limited_to_explicit_own_team_promotion(
    state, monkeypatch, kind
):
    from cognee.modules.users.exceptions import PermissionDeniedError

    state.actor.parent_user_id = None
    state.actor.tenant_id = uuid4()
    state.source.tenant_id = uuid4() if kind == "other_team" else None
    state.source.owner_id = uuid4() if kind == "foreign_personal" else state.actor.id
    monkeypatch.setattr(
        m, "get_authorized_dataset", AsyncMock(side_effect=PermissionDeniedError("tenant scope"))
    )
    monkeypatch.setattr(m, "get_principal_datasets", AsyncMock(return_value=[state.source]))
    args = (state.actor, state.source.id, "read", "user" if kind == "user_level" else "team")
    if kind == "own_personal":
        assert await m.get_promotion_source(*args) is state.source
    else:
        with pytest.raises(PermissionDeniedError):
            await m.get_promotion_source(*args)
