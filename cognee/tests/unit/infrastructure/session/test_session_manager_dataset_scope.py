import importlib
from collections import defaultdict
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.cache.models import (
    SessionAgentTraceEntry,
    SessionQAEntry,
)
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.infrastructure.session.session_scope import (
    get_storage_session_id,
    parse_storage_session_id,
)
from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord


class _InMemorySessionCache:
    """Small cache double that exposes every physical session key read by the manager."""

    def __init__(self) -> None:
        self.qas = defaultdict(list)
        self.traces = defaultdict(list)
        self.context = defaultdict(list)
        self.qa_reads = []
        self.trace_reads = []
        self.context_reads = []

    async def create_qa_entry(
        self,
        *,
        user_id,
        session_id,
        qa_id,
        question,
        context,
        answer,
        **kwargs,
    ):
        self.qas[(user_id, session_id)].append(
            SessionQAEntry(
                time="2026-01-01T00:00:00+00:00",
                qa_id=qa_id,
                question=question,
                context=context,
                answer=answer,
                feedback_text=kwargs.get("feedback_text"),
                feedback_score=kwargs.get("feedback_score"),
                used_graph_element_ids=kwargs.get("used_graph_element_ids"),
                used_session_context_ids=kwargs.get("used_session_context_ids"),
            )
        )

    async def get_all_qa_entries(self, user_id, session_id):
        self.qa_reads.append((user_id, session_id))
        return list(self.qas[(user_id, session_id)])

    async def append_agent_trace_step(
        self,
        *,
        user_id,
        session_id,
        trace_id,
        origin_function,
        status,
        **kwargs,
    ):
        self.traces[(user_id, session_id)].append(
            SessionAgentTraceEntry(
                trace_id=trace_id,
                origin_function=origin_function,
                status=status,
                memory_query=kwargs.get("memory_query", ""),
                memory_context=kwargs.get("memory_context", ""),
                method_params=kwargs.get("method_params") or {},
                method_return_value=kwargs.get("method_return_value"),
                error_message=kwargs.get("error_message", ""),
                session_feedback=kwargs.get("session_feedback", ""),
            )
        )

    async def get_agent_trace_session(self, user_id, session_id, last_n=None):
        self.trace_reads.append((user_id, session_id))
        entries = list(self.traces[(user_id, session_id)])
        return entries[-last_n:] if last_n is not None else entries

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        self.context[(user_id, session_id)].append(dict(entry_dump))

    async def get_session_context_entries(self, user_id, session_id):
        self.context_reads.append((user_id, session_id))
        return [dict(entry) for entry in self.context[(user_id, session_id)]]

    async def delete_session_context(self, user_id, session_id):
        return bool(self.context.pop((user_id, session_id), []))

    async def delete_session(self, *, user_id, session_id):
        qas = self.qas.pop((user_id, session_id), [])
        traces = self.traces.pop((user_id, session_id), [])
        context = self.context.pop((user_id, session_id), [])
        return bool(qas or traces or context)


@pytest.fixture
def scoped_manager_factory(monkeypatch):
    cache = _InMemorySessionCache()
    dataset_context_token = current_dataset_id.set(None)
    factory_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    session_manager_module = importlib.import_module(
        "cognee.infrastructure.session.session_manager"
    )

    monkeypatch.setattr(factory_module, "get_cache_engine", lambda: cache)
    monkeypatch.setattr(session_manager_module, "record_session_activity", AsyncMock())
    monkeypatch.setattr(session_manager_module, "index_session_qa", AsyncMock())
    monkeypatch.setattr(session_manager_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(SessionManager, "is_auto_feedback_enabled", lambda self: False)

    yield cache, factory_module.get_session_manager
    current_dataset_id.reset(dataset_context_token)


async def _write_all_payload_types(manager, *, label: str) -> None:
    await manager.add_qa(
        user_id="user-1",
        session_id="shared-public-id",
        question=f"question-{label}",
        context=f"qa-context-{label}",
        answer=f"answer-{label}",
    )
    await manager.add_agent_trace_step(
        user_id="user-1",
        session_id="shared-public-id",
        origin_function=f"tool-{label}",
        status="success",
        generate_feedback_with_llm=False,
    )
    assert await manager.create_session_context_entry(
        user_id="user-1",
        session_id="shared-public-id",
        entry_dump={"id": f"context-{label}", "kind": "context", "text": label},
    )


@pytest.mark.asyncio
async def test_dataset_bound_managers_isolate_qa_trace_and_context(scoped_manager_factory):
    _, get_session_manager = scoped_manager_factory
    dataset_a = uuid4()
    dataset_b = uuid4()

    manager_a = get_session_manager(dataset_id=dataset_a)
    manager_b = get_session_manager(dataset_id=str(dataset_b))
    await _write_all_payload_types(manager_a, label="a")
    await _write_all_payload_types(manager_b, label="b")

    # A UUID and its canonical string representation must address the same scope.
    manager_a_reader = get_session_manager(dataset_id=str(dataset_a))
    qas_a = await manager_a_reader.get_session(user_id="user-1", session_id="shared-public-id")
    qas_b = await manager_b.get_session(user_id="user-1", session_id="shared-public-id")
    traces_a = await manager_a_reader.get_agent_trace_session(
        user_id="user-1", session_id="shared-public-id"
    )
    traces_b = await manager_b.get_agent_trace_session(
        user_id="user-1", session_id="shared-public-id"
    )
    context_a = await manager_a_reader.get_session_context_entries(
        user_id="user-1", session_id="shared-public-id"
    )
    context_b = await manager_b.get_session_context_entries(
        user_id="user-1", session_id="shared-public-id"
    )

    assert [entry.question for entry in qas_a] == ["question-a"]
    assert [entry.question for entry in qas_b] == ["question-b"]
    assert [entry.origin_function for entry in traces_a] == ["tool-a"]
    assert [entry.origin_function for entry in traces_b] == ["tool-b"]
    assert [entry["id"] for entry in context_a] == ["context-a"]
    assert [entry["id"] for entry in context_b] == ["context-b"]


@pytest.mark.asyncio
async def test_dataset_scope_never_falls_back_to_legacy_session_key(scoped_manager_factory):
    cache, get_session_manager = scoped_manager_factory
    legacy_manager = get_session_manager()
    dataset_id = uuid4()
    scoped_manager = get_session_manager(dataset_id=dataset_id)

    await _write_all_payload_types(legacy_manager, label="legacy")
    cache.qa_reads.clear()
    cache.trace_reads.clear()
    cache.context_reads.clear()

    assert await scoped_manager.get_session(user_id="user-1", session_id="shared-public-id") == []
    assert (
        await scoped_manager.get_agent_trace_session(
            user_id="user-1", session_id="shared-public-id"
        )
        == []
    )
    assert (
        await scoped_manager.get_session_context_entries(
            user_id="user-1", session_id="shared-public-id"
        )
        == []
    )

    # Each scoped read performs exactly one lookup, and none touches the legacy public key.
    assert len(cache.qa_reads) == len(cache.trace_reads) == len(cache.context_reads) == 1
    physical_session_ids = {
        cache.qa_reads[0][1],
        cache.trace_reads[0][1],
        cache.context_reads[0][1],
    }
    assert physical_session_ids == {get_storage_session_id("shared-public-id", dataset_id)}

    legacy_qas = await legacy_manager.get_session(user_id="user-1", session_id="shared-public-id")
    legacy_traces = await legacy_manager.get_agent_trace_session(
        user_id="user-1", session_id="shared-public-id"
    )
    legacy_context = await legacy_manager.get_session_context_entries(
        user_id="user-1", session_id="shared-public-id"
    )
    assert [entry.question for entry in legacy_qas] == ["question-legacy"]
    assert [entry.origin_function for entry in legacy_traces] == ["tool-legacy"]
    assert [entry["id"] for entry in legacy_context] == ["context-legacy"]


@pytest.mark.asyncio
async def test_dataset_scoped_delete_cleans_exact_lifecycle_scope(
    scoped_manager_factory,
    monkeypatch,
):
    _, get_session_manager = scoped_manager_factory
    dataset_a = uuid4()
    dataset_b = uuid4()
    manager_a = get_session_manager(dataset_id=dataset_a)
    manager_b = get_session_manager(dataset_id=dataset_b)
    await _write_all_payload_types(manager_a, label="a")
    await _write_all_payload_types(manager_b, label="b")

    session_manager_module = importlib.import_module(
        "cognee.infrastructure.session.session_manager"
    )
    lifecycle_delete = AsyncMock(return_value=True)
    vector_delete = AsyncMock()
    monkeypatch.setattr(session_manager_module, "delete_session_lifecycle", lifecycle_delete)
    monkeypatch.setattr(session_manager_module, "delete_session_qa_vectors", vector_delete)

    assert await manager_a.delete_session(
        user_id="user-1",
        session_id="shared-public-id",
    )
    lifecycle_delete.assert_awaited_once_with(
        session_id="shared-public-id",
        user_id="user-1",
        dataset_id=str(dataset_a),
    )
    vector_delete.assert_awaited_once_with(
        user_id="user-1",
        session_id=get_storage_session_id("shared-public-id", dataset_a),
    )

    assert await manager_a.get_session(user_id="user-1", session_id="shared-public-id") == []
    remaining = await manager_b.get_session(user_id="user-1", session_id="shared-public-id")
    assert [entry.question for entry in remaining] == ["question-b"]


@pytest.mark.asyncio
async def test_dataset_scoped_delete_removes_stale_lifecycle_when_cache_is_absent(
    scoped_manager_factory,
    monkeypatch,
):
    _, get_session_manager = scoped_manager_factory
    dataset_id = uuid4()
    manager = get_session_manager(dataset_id=dataset_id)

    session_manager_module = importlib.import_module(
        "cognee.infrastructure.session.session_manager"
    )
    lifecycle_delete = AsyncMock(return_value=True)
    vector_delete = AsyncMock()
    monkeypatch.setattr(session_manager_module, "delete_session_lifecycle", lifecycle_delete)
    monkeypatch.setattr(session_manager_module, "delete_session_qa_vectors", vector_delete)

    assert await manager.delete_session(
        user_id="user-1",
        session_id="stale-public-id",
    )
    lifecycle_delete.assert_awaited_once_with(
        session_id="stale-public-id",
        user_id="user-1",
        dataset_id=str(dataset_id),
    )
    vector_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_dataset_scoped_delete_keeps_lifecycle_when_cache_is_unavailable(monkeypatch):
    dataset_id = uuid4()
    lifecycle_delete = AsyncMock(return_value=True)
    session_manager_module = importlib.import_module(
        "cognee.infrastructure.session.session_manager"
    )
    monkeypatch.setattr(session_manager_module, "delete_session_lifecycle", lifecycle_delete)
    manager = session_manager_module.SessionManager(cache_engine=None, dataset_id=dataset_id)

    assert not await manager.delete_session(
        user_id="user-1",
        session_id="still-persisted",
    )
    lifecycle_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_manager_inherits_active_dataset_context(scoped_manager_factory):
    _, get_session_manager = scoped_manager_factory
    dataset_id = uuid4()
    token = current_dataset_id.set(dataset_id)
    try:
        contextual_manager = get_session_manager()
        await contextual_manager.add_qa(
            user_id="user-1",
            session_id="shared-public-id",
            question="scoped question",
            context="scoped context",
            answer="scoped answer",
        )
    finally:
        current_dataset_id.reset(token)

    unscoped_manager = get_session_manager()
    assert await unscoped_manager.get_session(user_id="user-1", session_id="shared-public-id") == []
    contextual_reader = get_session_manager(dataset_id=dataset_id)
    scoped_entries = await contextual_reader.get_session(
        user_id="user-1", session_id="shared-public-id"
    )
    assert [entry.question for entry in scoped_entries] == ["scoped question"]


def test_dataset_storage_session_id_round_trips_public_identity():
    dataset_id = uuid4()
    public_session_id = "session:with spaces/and-üñicode"

    storage_session_id = get_storage_session_id(public_session_id, dataset_id)
    decoded = parse_storage_session_id(storage_session_id)

    assert storage_session_id != public_session_id
    assert decoded.session_id == public_session_id
    assert decoded.dataset_id == str(dataset_id)


def test_unscoped_and_malformed_reserved_session_ids_stay_legacy():
    legacy_session_id = "legacy-public-id"
    malformed_reserved_id = "__cognee_dataset_session_v1__:not-valid"

    assert get_storage_session_id(legacy_session_id) == legacy_session_id
    assert parse_storage_session_id(legacy_session_id).dataset_id is None
    assert parse_storage_session_id(malformed_reserved_id).session_id == malformed_reserved_id
    assert parse_storage_session_id(malformed_reserved_id).dataset_id is None


@pytest.mark.asyncio
async def test_unscoped_write_cannot_alias_an_existing_scoped_physical_id(
    scoped_manager_factory,
):
    _, get_session_manager = scoped_manager_factory
    dataset_id = uuid4()
    scoped_manager = get_session_manager(dataset_id=dataset_id)
    legacy_manager = get_session_manager(dataset_id=None)

    await scoped_manager.add_qa(
        user_id="user-1",
        session_id="public-id",
        question="scoped",
        context="",
        answer="safe",
    )
    physical_id = get_storage_session_id("public-id", dataset_id)

    with pytest.raises(ValueError, match="reserved prefix"):
        await legacy_manager.add_qa(
            user_id="user-1",
            session_id=physical_id,
            question="legacy collision",
            context="",
            answer="must not mix",
        )

    entries = await scoped_manager.get_session(user_id="user-1", session_id="public-id")
    assert [entry.question for entry in entries] == ["scoped"]


def test_lifecycle_serializers_expose_public_not_storage_session_id():
    dataset_id = uuid4()
    user_id = uuid4()
    public_session_id = "public-session-id"
    storage_session_id = get_storage_session_id(public_session_id, dataset_id)
    now = datetime.now(timezone.utc)
    record = SessionRecord(
        session_id=storage_session_id,
        public_session_id=public_session_id,
        user_id=user_id,
        dataset_id=dataset_id,
        status="running",
        started_at=now,
        last_activity_at=now,
        ended_at=None,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        error_count=0,
        last_model=None,
    )
    model_usage = SessionModelUsage(
        session_id=storage_session_id,
        user_id=user_id,
        model="test-model",
        tokens_in=1,
        tokens_out=2,
        cost_usd=0.01,
        updated_at=now,
    )

    assert record.to_dict()["session_id"] == public_session_id
    assert model_usage.to_dict()["session_id"] == public_session_id
    assert storage_session_id not in {
        record.to_dict()["session_id"],
        model_usage.to_dict()["session_id"],
    }
