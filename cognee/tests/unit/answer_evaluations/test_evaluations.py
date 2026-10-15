import asyncio
import json
import time
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.answer_evaluations.models import EvaluationRun, EvaluationScope, EvaluationTrace
from cognee.modules.answer_evaluations.privacy import Protection, sanitize
from cognee.modules.answer_evaluations.schema import Trace, score
from cognee.modules.answer_evaluations.service import (
    EvaluationError,
    erase,
    ingest,
    lock_scope,
    summary,
)
from cognee.modules.answer_evaluations.warehouse import Warehouse
from cognee.modules.answer_evaluations.worker import process_scope

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def protection(monkeypatch):
    monkeypatch.setenv("COGNEE_EVALUATION_ENCRYPTION_KEYS", Fernet.generate_key().decode())
    monkeypatch.setenv("COGNEE_EVALUATION_ID_KEY", "test-only-identity-key-32-characters")
    return Protection()


def trace(dataset, **updates):
    return Trace.model_validate(
        dict(
            event_id=str(uuid4()),
            dataset_id=dataset,
            captured_at=datetime.now(timezone.utc).isoformat(),
            consent_version="answer-evaluation-v1",
            prompt="What does alice@example.com use?",
            answer="Alice uses the SDK.",
            context="",
            evidence=[
                {"source_id": "source-1", "version": "v1", "text": "alice@example.com uses the SDK"}
            ],
            model="pinned-test",
            plugin="claude",
            **updates,
        )
    )


async def database(path):
    engine = create_async_engine("sqlite+aiosqlite:///" + str(path), connect_args={"timeout": 60})
    async with engine.begin() as connection:
        for model in (EvaluationScope, EvaluationTrace, EvaluationRun):
            await connection.run_sync(lambda c, model=model: model.__table__.create(c))
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.asyncio
async def test_retries_conflicts_cap_and_ciphertext(tmp_path, protection):
    engine, sessions = await database(tmp_path / "state.db")
    owner, dataset = str(uuid4()), str(uuid4())
    first = trace(dataset)

    async def write(item):
        async with sessions() as s:
            result = await ingest(s, protection, owner, item)
            await s.commit()
            return result

    receipt = await write(first)
    assert await write(first) == receipt  # Commit-before-ACK retry.
    changed = first.model_copy(update={"answer": "changed"})
    with pytest.raises(EvaluationError) as exc:
        await write(changed)
    assert exc.value.status == 409
    # Existing scope serializes concurrent uploads across connections.
    await asyncio.gather(*(write(trace(dataset)) for _ in range(110)))
    async with sessions() as s:
        rows = (await s.scalars(select(EvaluationTrace))).all()
        assert len(rows) == 111
        assert sum(bool(r.ciphertext) for r in rows) == 100
        assert all("alice" not in (r.ciphertext or "") for r in rows)
        body = protection.decrypt(next(r.ciphertext for r in rows if r.ciphertext))
        assert "alice@example.com" not in json.dumps(body)
        assert body["prompt"].split()[2] == body["evidence"][0]["text"].split()[0]
    await engine.dispose()


@pytest.mark.asyncio
async def test_selection_history_daily_gate_and_deletion(tmp_path, protection):
    engine, sessions = await database(tmp_path / "state.db")
    owner, dataset = str(uuid4()), str(uuid4())
    identity = protection.identity(owner, dataset)
    async with sessions() as s:
        for _ in range(30):
            await ingest(s, protection, owner, trace(dataset))
        await s.commit()
    warehouse = Warehouse(duckdb.connect(str(tmp_path / "warehouse.db")))
    calls = []

    async def judge(body, swapped):
        # Selection must already be durable before the first paid call.
        async with sessions() as s:
            run = await s.scalar(select(EvaluationRun))
            assert len(run.result["selected"]) == 20
        calls.append(body)
        return {"verdict": "better", "method": "test", "replay_tokens": 12}

    run_id = await process_scope(sessions, warehouse, protection, identity, judge)
    assert run_id and len(calls) == 20
    await process_scope(sessions, warehouse, protection, identity, judge)
    assert len(calls) == 20
    async with sessions() as s:
        result = await summary(s, identity)
        assert result["score"] == 100 and result["status"] == "ready"
        assert (await summary(s, protection.identity(str(uuid4()), dataset)))["status"] == "pending"
        run = await s.get(EvaluationRun, run_id)
        assert len({x["trace_id"] for x in run.result["selected"]}) == 20
        assert "prompt" not in json.dumps(run.result)
        scope = await lock_scope(s, identity, owner, dataset)
        scope.enabled = False
        await erase(s, scope)
        await s.commit()
    await process_scope(sessions, warehouse, protection, identity, judge)
    assert (
        warehouse.connection.execute("SELECT count(*) FROM answer_quality.traces").fetchone()[0]
        == 0
    )
    async with sessions() as s:
        with pytest.raises(EvaluationError):
            await ingest(s, protection, owner, trace(dataset))
    warehouse.connection.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_delete_during_judging_does_not_resurrect_results(tmp_path, protection):
    engine, sessions = await database(tmp_path / "state.db")
    owner, dataset = str(uuid4()), str(uuid4())
    identity = protection.identity(owner, dataset)
    async with sessions() as s:
        for _ in range(20):
            await ingest(s, protection, owner, trace(dataset))
        await s.commit()
    warehouse = Warehouse(duckdb.connect(":memory:"))

    async def judge(body, swapped):
        async with sessions() as s:
            scope = await lock_scope(s, identity, owner, dataset)
            scope.enabled = False
            await erase(s, scope)
            await s.commit()
        return {"verdict": "better"}

    await process_scope(sessions, warehouse, protection, identity, judge)
    async with sessions() as s:
        assert not (await s.scalars(select(EvaluationRun))).all()
    warehouse.connection.close()
    await engine.dispose()


def test_score_and_redaction():
    assert [score(0, 0, 0), score(0, 20, 20), score(12, 2, 20), score(20, 0, 20)] == [
        47,
        47,
        74,
        100,
    ]
    safe = sanitize({"a": "alice@example.com Bearer secretvalue", "b": "alice@example.com"})
    assert "alice" not in str(safe) and "secretvalue" not in str(safe)
    assert safe["a"].split()[0] == safe["b"]


@pytest.mark.asyncio
async def test_authenticated_http_receipt_and_isolation(tmp_path, protection, monkeypatch):
    import importlib
    from types import SimpleNamespace

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    api = importlib.import_module("cognee.api.v1.evaluations.routers.get_evaluations_router")
    engine, sessions = await database(tmp_path / "http.db")
    dataset, other_dataset = uuid4(), uuid4()
    owner = SimpleNamespace(id=uuid4())
    monkeypatch.setattr(
        api, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )

    async def permitted(_):
        return [dataset]

    monkeypatch.setattr(api, "get_permitted_dataset_ids", permitted)
    monkeypatch.setenv("COGNEE_EVALUATIONS_ENABLED", "true")
    app = FastAPI()
    app.dependency_overrides[api.get_authenticated_user] = lambda: owner
    app.include_router(api.get_evaluations_router(), prefix="/api/v1/evaluations")
    body = trace(str(dataset)).model_dump(mode="json")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/evaluations/traces", json=body)
        assert response.status_code == 202
        assert (
            await client.post("/api/v1/evaluations/traces", json=body)
        ).json() == response.json()
        assert (
            await client.post("/api/v1/evaluations/traces", json={**body, "answer": "other"})
        ).status_code == 409
        assert (
            await client.get(f"/api/v1/evaluations/summary?dataset_id={other_dataset}")
        ).status_code == 403
        assert (await client.get(f"/api/v1/evaluations/summary?dataset_id={dataset}")).json()[
            "score"
        ] == 47
        response = await client.put(
            f"/api/v1/evaluations/consent?dataset_id={dataset}", json={"enabled": False}
        )
        assert response.status_code == 200
        assert (await client.post("/api/v1/evaluations/traces", json=body)).status_code == 403
    await engine.dispose()


@pytest.mark.asyncio
async def test_expiry_and_warehouse_failure_preserve_durable_receipt(tmp_path, protection):
    engine, sessions = await database(tmp_path / "state.db")
    owner, dataset = str(uuid4()), str(uuid4())
    identity = protection.identity(owner, dataset)
    item = trace(dataset)
    async with sessions() as s:
        receipt = await ingest(s, protection, owner, item)
        await s.commit()

    class FailedWarehouse:
        def replace(self, *args):
            raise OSError("offline")

    with pytest.raises(OSError):
        await process_scope(sessions, FailedWarehouse(), protection, identity, None)
    async with sessions() as s:
        assert await ingest(s, protection, owner, item) == receipt
        await s.commit()
    warehouse = Warehouse(duckdb.connect(":memory:"))
    await process_scope(
        sessions, warehouse, protection, identity, None, now=time.time() + 31 * 86400
    )
    async with sessions() as s:
        assert not (await s.scalars(select(EvaluationTrace))).all()
    assert (
        warehouse.connection.execute("SELECT count(*) FROM answer_quality.traces").fetchone()[0]
        == 0
    )
    warehouse.connection.close()
    await engine.dispose()


def test_migration_matches_models_and_downgrades():
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect

    path = Path(__file__).parents[3] / "alembic/versions/a9e1b2c3d4f5_add_answer_evaluations.py"
    spec = importlib.util.spec_from_file_location("evaluation_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        for model in (EvaluationScope, EvaluationTrace, EvaluationRun):
            assert {c["name"] for c in inspect(connection).get_columns(model.__tablename__)} == set(
                model.__table__.columns.keys()
            )
        migration.downgrade()
        assert not inspect(connection).get_table_names()


@pytest.mark.asyncio
async def test_agent_maps_to_owner_but_cannot_control_privacy(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from fastapi import HTTPException

    api = importlib.import_module("cognee.api.v1.evaluations.routers.get_evaluations_router")
    parent, dataset = uuid4(), uuid4()
    agent = SimpleNamespace(id=uuid4(), parent_user_id=parent)

    async def permitted(_):
        return [dataset]

    monkeypatch.setattr(api, "get_permitted_dataset_ids", permitted)
    assert await api.authorize(agent, dataset) == str(parent)
    with pytest.raises(HTTPException) as exc:
        api.require_human(agent)
    assert exc.value.status_code == 403
