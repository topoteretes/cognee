from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.context_global_variables import session_user
from cognee.infrastructure.session.session_maintenance import process_session_maintenance
from cognee.infrastructure.session.session_search_models import (
    SessionMaintenanceResult,
    SessionMaintenanceWorkItem,
    SessionTurnEvidence,
)


class FakeSessionManager:
    def __init__(self, rows, *, fail_read=False, fail_create=False, fail_evidence_update=False):
        self.rows = list(rows)
        self.fail_read = fail_read
        self.fail_create = fail_create
        self.fail_evidence_update = fail_evidence_update
        self.updates = []

    async def get_session_context_entries(self, user_id, session_id, strict=False):
        if self.fail_read:
            raise RuntimeError("read failed")
        return list(self.rows)

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        if self.fail_create:
            return False
        self.rows.append(entry_dump)
        return True

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        self.updates.append((entry_id, merge))
        if self.fail_evidence_update and entry_id == "e1":
            return False
        for row in self.rows:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False


def _evidence(**overrides):
    values = {
        "id": "e1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "current_raw_message": "Please keep answers concise.",
        "current_response": "This assistant response must not become a candidate.",
        "previous_answer": "A prior answer",
        "previous_served_context": (("ctx1", "Be concise."),),
    }
    values.update(overrides)
    return SessionTurnEvidence(**values).model_dump(mode="json")


def _work_item():
    return SessionMaintenanceWorkItem(
        evidence_id="e1",
        user_id=str(uuid4()),
        session_id="s1",
    )


@pytest.mark.asyncio
async def test_valid_empty_maintenance_completes_without_leaking_current_response(monkeypatch):
    manager = FakeSessionManager([_evidence()])
    llm = AsyncMock(return_value=SessionMaintenanceResult())
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.get_session_manager",
        lambda dataset_id: manager,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.LLMGateway.acreate_structured_output",
        llm,
    )
    prior_user = session_user.get()

    await process_session_maintenance(_work_item())

    assert manager.rows[0]["status"] == "completed"
    llm_input = llm.await_args.kwargs["text_input"]
    assert "current_assistant" not in llm_input
    assert "must not become a candidate" not in llm_input
    assert "previous_assistant_answer_untrusted" in llm_input
    assert session_user.get() is prior_user


@pytest.mark.asyncio
async def test_maintenance_applies_only_served_ratings_and_candidates(monkeypatch):
    manager = FakeSessionManager(
        [
            _evidence(),
            {
                "id": "ctx1",
                "kind": "context",
                "section": "rules",
                "content": "Be concise.",
                "normalized_content": "be concise.",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "helpful_count": 0,
                "harmful_count": 0,
            },
        ]
    )
    maintenance = SessionMaintenanceResult(
        served_context_ratings=[
            {"entry_id": "ctx1", "rating": "helpful"},
            {"entry_id": "not-served", "rating": "harmful"},
        ],
        candidate_context_updates=[
            {"section": "preferences", "content": "Prefer short answers.", "confidence": 0.9}
        ],
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.get_session_manager",
        lambda dataset_id: manager,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.LLMGateway.acreate_structured_output",
        AsyncMock(return_value=maintenance),
    )

    await process_session_maintenance(_work_item())

    assert manager.rows[1]["helpful_count"] == 1
    assert not any(entry_id == "not-served" for entry_id, _merge in manager.updates)
    assert any(row.get("content") == "Prefer short answers." for row in manager.rows)
    assert manager.rows[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_missing_evidence_does_no_work(monkeypatch):
    manager = FakeSessionManager([])
    llm = AsyncMock()
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.get_session_manager",
        lambda dataset_id: manager,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.LLMGateway.acreate_structured_output",
        llm,
    )

    await process_session_maintenance(_work_item())

    llm.assert_not_awaited()
    assert manager.updates == []


@pytest.mark.asyncio
async def test_read_failure_attempts_failed_status_without_calling_llm(monkeypatch):
    manager = FakeSessionManager([], fail_read=True)
    llm = AsyncMock()
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.get_session_manager",
        lambda dataset_id: manager,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.LLMGateway.acreate_structured_output",
        llm,
    )

    await process_session_maintenance(_work_item())

    llm.assert_not_awaited()
    assert manager.updates[0][1]["status"] == "failed"
    assert manager.updates[0][1]["error"]


@pytest.mark.asyncio
async def test_apply_failure_marks_evidence_failed(monkeypatch):
    manager = FakeSessionManager([_evidence()], fail_create=True)
    maintenance = SessionMaintenanceResult(
        candidate_context_updates=[
            {"section": "rules", "content": "Always cite sources.", "confidence": 0.9}
        ]
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.get_session_manager",
        lambda dataset_id: manager,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.LLMGateway.acreate_structured_output",
        AsyncMock(return_value=maintenance),
    )

    await process_session_maintenance(_work_item())

    assert manager.rows[0]["status"] == "failed"
    assert manager.updates[-1][1]["error"]


@pytest.mark.asyncio
async def test_failed_completion_status_write_leaves_evidence_pending(monkeypatch):
    manager = FakeSessionManager([_evidence()], fail_evidence_update=True)
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.get_session_manager",
        lambda dataset_id: manager,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.LLMGateway.acreate_structured_output",
        AsyncMock(return_value=SessionMaintenanceResult()),
    )

    await process_session_maintenance(_work_item())

    # The completed write was attempted and rejected, so the record stays recoverable.
    assert manager.updates == [("e1", {"status": "completed", "error": None})]
    assert manager.rows[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_worker_skips_apply_when_evidence_was_distilled_during_llm(monkeypatch):
    manager = FakeSessionManager([_evidence()])

    async def distill_during_call(**kwargs):
        manager.rows[0]["distilled_at"] = "2026-08-06T10:00:00+00:00"
        return SessionMaintenanceResult(
            candidate_context_updates=[
                {"section": "rules", "content": "Always cite sources.", "confidence": 0.9}
            ]
        )

    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.get_session_manager",
        lambda dataset_id: manager,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_maintenance.LLMGateway.acreate_structured_output",
        distill_during_call,
    )

    await process_session_maintenance(_work_item())

    assert len(manager.rows) == 1
    assert manager.updates == []
