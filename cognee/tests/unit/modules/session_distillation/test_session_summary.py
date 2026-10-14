import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.session_distillation import summary as module
from cognee.modules.session_distillation.distill import SessionDistillationScope
from cognee.tests.unit.infrastructure.session.test_agent_context_extraction import (
    FakeSessionManager,
)


@pytest.mark.asyncio
async def test_summary_retries_exact_prose_and_watermarks_only_after_publication(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4())
    scope = SessionDistillationScope(session_id="s", user=user, dataset=dataset)
    manager = FakeSessionManager()
    monkeypatch.setattr(module, "get_session_manager", lambda: manager)
    monkeypatch.setattr(module, "resolve_distillation_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(
        module,
        "load_session_qa_rows",
        AsyncMock(
            return_value=[
                {
                    "question": "Make outbox delivery safe",
                    "answer": "Committed the cursor atomically.",
                }
            ]
        ),
    )
    llm = AsyncMock(
        return_value=module.SessionSummary(
            goal="Safe delivery",
            outcome="Atomic cursor update implemented",
            decisions=["Use a shared transaction to avoid loss"],
            open_items=["Run crash recovery test"],
        )
    )
    monkeypatch.setattr(module.LLMGateway, "acreate_structured_output", llm)
    add = AsyncMock()
    cognify = AsyncMock(side_effect=[RuntimeError("offline"), None])
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.add"), "add", add)
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.cognify"), "cognify", cognify)
    with pytest.raises(RuntimeError, match="offline"):
        await module.publish_session_summary("s", dataset.id, user)
    assert await module.latest_session_summary(manager, str(user.id), "s") is None
    assert await module.publish_session_summary("s", dataset.id, user)
    assert not await module.publish_session_summary("s", dataset.id, user)
    llm.assert_awaited_once()
    assert add.await_args_list[0] == add.await_args_list[1]
    assert "session_summaries" in add.await_args.kwargs["node_set"]
    saved = await module.latest_session_summary(manager, str(user.id), "s")
    assert saved["open_items"] == ["Run crash recovery test"]
