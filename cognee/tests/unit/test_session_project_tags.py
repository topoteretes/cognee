from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.session.project_tags import (
    TaggedTrace,
    bind_project_tags,
    get_project_tags,
)
from cognee.infrastructure.session.session_persist_watermark import SessionPersistWindow
from cognee.memory.entries import QAEntry, TraceEntry
from cognee.modules.data.methods.provision_session_companion import (
    CompanionConflict,
    provision_session_companion,
)
from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback
from cognee.tasks.memify.cognify_session import cognify_session


@pytest.mark.asyncio
async def test_tags_are_immutable_and_cache_errors_propagate():
    rows = []

    async def append(user, session, value):
        rows.append(value)

    manager = SimpleNamespace(
        _cache=SimpleNamespace(
            get_session_context_entries=AsyncMock(side_effect=lambda *_: rows),
            create_session_context_entry=AsyncMock(side_effect=append),
        )
    )
    await bind_project_tags(manager, "u", "s", ["project-a"])
    await bind_project_tags(manager, "u", "s", ["project-a"])
    assert await get_project_tags(manager, "u", "s") == ("project-a",)
    with pytest.raises(ValueError):
        await bind_project_tags(manager, "u", "s", ["project-b"])
    assert len(rows) == 1
    manager._cache.get_session_context_entries.side_effect = OSError("unavailable")
    with pytest.raises(OSError):
        await get_project_tags(manager, "u", "s")


@pytest.mark.asyncio
async def test_qa_and_trace_tags_reach_graph_ingestion():
    qa = QAEntry(question="q", answer="a", node_set=["project-a"])
    trace = TraceEntry(origin_function="edit", node_set=["project-a"])
    assert qa.model_dump()["node_set"] == trace.model_dump()["node_set"]
    window = SessionPersistWindow("u", "s", "Question: q\nAnswer: a", 1, tuple(qa.node_set))
    with (
        patch("cognee.add", new_callable=AsyncMock) as add,
        patch("cognee.cognify", new_callable=AsyncMock),
        patch(
            "cognee.tasks.memify.cognify_session.save_persisted_qa_count", new_callable=AsyncMock
        ),
    ):
        await cognify_session(window, dataset_id="dataset")
        assert add.call_args.kwargs["node_set"] == ["user_sessions_from_cache", "project-a"]
        await cognify_agent_trace_feedback(
            TaggedTrace("trace", tuple(trace.node_set)), dataset_id="dataset"
        )
        assert add.call_args.kwargs["node_set"] == ["agent_trace_feedbacks", "project-a"]


@pytest.mark.asyncio
async def test_companion_copies_authoritative_acl_and_rejects_drift():
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.methods.create_authorized_dataset import create_authorized_dataset
    from cognee.modules.users.methods import get_default_user
    from cognee.modules.users.models import ACL

    user = await get_default_user()
    primary = await create_authorized_dataset("companion-test-" + uuid4().hex, user)
    result = await provision_session_companion(primary.id, user)
    assert result["permissions_verified"]
    assert result == await provision_session_companion(primary.id, user)
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        original = (await session.scalars(select(ACL).where(ACL.dataset_id == primary.id))).all()
        copied = (
            await session.scalars(select(ACL).where(ACL.dataset_id == UUID(result["dataset_id"])))
        ).all()
        assert {(a.principal_id, a.permission_id) for a in copied} == {
            (a.principal_id, a.permission_id) for a in original
        }
        await session.delete(copied[0])
        await session.commit()
    with pytest.raises(CompanionConflict):
        await provision_session_companion(primary.id, user)
    with pytest.raises(PermissionError):
        await provision_session_companion(
            primary.id, SimpleNamespace(id=uuid4(), tenant_id=user.tenant_id)
        )


def test_http_schema_advertises_typed_project_tags():
    from fastapi import FastAPI

    from cognee.api.v1.remember.routers.get_remember_router import get_remember_router

    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/api/v1/remember")
    schemas = app.openapi()["components"]["schemas"]
    assert "node_set" in schemas["QAEntry"]["properties"]
    assert "node_set" in schemas["TraceEntry"]["properties"]


def test_model_config_passes_provider_model_id_without_integration_allowlist(monkeypatch):
    import importlib

    config_module = importlib.import_module("cognee.api.v1.config.config")
    config = SimpleNamespace(llm_model="old")
    monkeypatch.setattr(config_module, "get_llm_config", lambda: config)
    config_module.config.set_llm_model("anthropic/provider-model-from-provider-docs")
    assert config.llm_model == "anthropic/provider-model-from-provider-docs"
