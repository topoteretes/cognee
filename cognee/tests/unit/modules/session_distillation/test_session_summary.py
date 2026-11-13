import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.exceptions import UpdateTargetNotFoundError
from cognee.infrastructure.session.session_node_set import SESSION_NODE_SET_STATE_ID
from cognee.modules.pipelines.exceptions import AfterRunHookError
from cognee.modules.session_distillation import distill
from cognee.modules.session_distillation import summary as module
from cognee.modules.session_distillation.distill import SessionDistillationScope
from cognee.tests.unit.infrastructure.session.test_agent_context_extraction import (
    FakeSessionManager,
)

pytestmark = pytest.mark.asyncio

SUMMARY = module.SessionSummary(
    goal="Safe delivery",
    outcome="Atomic cursor update implemented",
    decisions=["Use a shared transaction to avoid loss"],
    open_items=["Run crash recovery test"],
)


class SummaryCache(FakeSessionManager):
    """The fake context store plus the QA read the real loader makes."""

    def __init__(self, turns):
        super().__init__()
        self.turns = list(turns)
        self.unreadable = False
        self.since_seen = []

    async def get_session_context_entries(self, user_id, session_id, raise_on_error=False):
        if self.unreadable:
            if raise_on_error:
                raise ConnectionError("cache down")
            return []
        return list(self.store)

    async def get_session(self, *, user_id, session_id, formatted, since, lead_in):
        self.since_seen.append(since)
        return list(self.turns)


def _turn(question):
    return {"question": question, "answer": "done"}


@pytest.fixture
def env(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    scope = SessionDistillationScope(session_id="s", user=user, dataset=dataset)
    cache = SummaryCache([_turn("Make outbox delivery safe")])
    monkeypatch.setattr(module, "get_session_manager", lambda: cache)
    # The real loader runs: a signature change between it and its caller fails here.
    monkeypatch.setattr(distill, "get_session_manager", lambda: cache)
    monkeypatch.setattr(module, "resolve_distillation_scope", AsyncMock(return_value=scope))
    llm = AsyncMock(return_value=SUMMARY)
    monkeypatch.setattr(module.LLMGateway, "acreate_structured_output", llm)
    data_id = uuid4()
    add = AsyncMock(return_value=SimpleNamespace(data_ids=[data_id]))
    cognify = AsyncMock()
    update = AsyncMock(return_value={"status": "incremental", "after_run_error": None})
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.add"), "add", add)
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.cognify"), "cognify", cognify)
    monkeypatch.setattr(importlib.import_module("cognee.api.v1.update"), "update", update)
    return SimpleNamespace(
        user=user,
        dataset=dataset,
        cache=cache,
        llm=llm,
        add=add,
        cognify=cognify,
        update=update,
        data_id=data_id,
    )


async def _publish(env):
    return await module.publish_session_summary("s", env.dataset.id, env.user)


async def test_summary_reads_the_whole_session(env):
    await _publish(env)

    assert env.cache.since_seen == [None]


async def test_retry_after_failed_cognify_updates_the_same_document(env):
    env.cognify.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        await _publish(env)
    assert await module.latest_session_summary(env.cache, str(env.user.id), "s") is None

    env.cognify.side_effect = None
    assert await _publish(env)
    assert not await _publish(env)

    env.llm.assert_awaited_once()
    env.add.assert_awaited_once()
    assert env.update.await_args.kwargs["data_id"] == env.data_id
    saved = await module.latest_session_summary(env.cache, str(env.user.id), "s")
    assert saved["open_items"] == ["Run crash recovery test"]


async def test_a_new_turn_replaces_the_summary_in_place(env):
    assert await _publish(env)
    env.cache.turns.append(_turn("Add crash recovery test"))
    assert await _publish(env)

    env.add.assert_awaited_once()
    env.update.assert_awaited_once()
    assert env.update.await_args.kwargs["data_id"] == env.data_id
    assert "node_set" not in env.update.await_args.kwargs


async def test_summary_carries_the_pinned_node_set(env):
    env.cache.store.append({"id": SESSION_NODE_SET_STATE_ID, "node_set": ["payments"]})

    await _publish(env)

    node_set = env.add.await_args.kwargs["node_set"]
    assert node_set[0] == module.SUMMARY_NODE_SET
    assert "payments" in node_set


async def test_unreadable_cache_adds_no_second_document(env):
    assert await _publish(env)
    env.cache.turns.append(_turn("Add crash recovery test"))
    env.cache.unreadable = True

    with pytest.raises(ConnectionError):
        await _publish(env)

    env.add.assert_awaited_once()
    env.update.assert_not_awaited()


async def test_after_run_hook_error_is_reported_once(env):
    env.cognify.side_effect = AfterRunHookError([(env.dataset.id, "ds", "compaction failed")])

    with pytest.raises(AfterRunHookError):
        await _publish(env)
    assert not await _publish(env)

    env.cognify.assert_awaited_once()
    assert await module.latest_session_summary(env.cache, str(env.user.id), "s") is not None


async def test_update_after_run_error_is_recorded_then_raised(env):
    assert await _publish(env)
    env.cache.turns.append(_turn("Add crash recovery test"))
    env.update.return_value = {"status": "incremental", "after_run_error": "compaction failed"}

    with pytest.raises(AfterRunHookError):
        await _publish(env)
    assert not await _publish(env)

    env.update.assert_awaited_once()


async def test_deleted_summary_document_is_written_again(env):
    assert await _publish(env)
    env.cache.turns.append(_turn("Add crash recovery test"))
    env.update.side_effect = UpdateTargetNotFoundError(
        data_id=env.data_id, dataset_id=env.dataset.id
    )

    assert await _publish(env)

    assert env.add.await_count == 2
