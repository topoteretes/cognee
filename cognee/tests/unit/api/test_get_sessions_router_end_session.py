"""Wiring tests for POST /api/v1/sessions/{session_id}/end (SDK-593).

``mark_ended`` itself is a one-statement lifecycle helper; this file checks the
router's contract: owner-only lookup, 404 on unknown, idempotent 200 on an
already-ended session, and the optional ``failed`` status.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.sessions.routers import get_sessions_router as router_module


def _app(user_id):
    app = FastAPI()
    app.include_router(router_module.get_sessions_router(), prefix="/api/v1/sessions")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id, tenant_id=None
    )
    return app


class _FakeRows:
    """A tiny session_records stand-in keyed by (session_id, user_id)."""

    def __init__(self, rows):
        self.rows = {(r.session_id, r.user_id): r for r in rows}
        self.mark_calls = []

    async def get_session_row(self, *, session_id, user_id, **_):
        return self.rows.get((session_id, user_id))

    async def mark_ended(self, *, session_id, user_id, status):
        self.mark_calls.append((session_id, user_id, status))
        row = self.rows.get((session_id, user_id))
        if row is None or row.status != "running":
            return False
        row.status = status.value
        row.ended_at = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
        return True


def _row(session_id, user_id, status="running", ended_at=None):
    return SimpleNamespace(session_id=session_id, user_id=user_id, status=status, ended_at=ended_at)


def _wire(monkeypatch, rows):
    monkeypatch.setattr(router_module, "get_session_row", rows.get_session_row)
    monkeypatch.setattr(router_module, "mark_ended", rows.mark_ended)


def test_end_marks_a_running_session_completed(monkeypatch):
    user_id = uuid4()
    rows = _FakeRows([_row("s1", user_id)])
    _wire(monkeypatch, rows)

    response = TestClient(_app(user_id)).post("/api/v1/sessions/s1/end")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s1"
    assert body["status"] == "completed"
    assert body["already_ended"] is False
    assert body["ended_at"] == "2026-09-09T12:00:00+00:00"
    assert rows.mark_calls == [("s1", user_id, router_module.SessionStatus.COMPLETED)]


def test_end_accepts_failed_status(monkeypatch):
    user_id = uuid4()
    rows = _FakeRows([_row("s1", user_id)])
    _wire(monkeypatch, rows)

    response = TestClient(_app(user_id)).post("/api/v1/sessions/s1/end", json={"status": "failed"})

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_end_is_idempotent_on_an_already_ended_session(monkeypatch):
    user_id = uuid4()
    ended = datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)
    rows = _FakeRows([_row("s1", user_id, status="completed", ended_at=ended)])
    _wire(monkeypatch, rows)

    response = TestClient(_app(user_id)).post("/api/v1/sessions/s1/end", json={"status": "failed"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"  # the first terminal status stands
    assert body["already_ended"] is True
    assert body["ended_at"] == ended.isoformat()
    assert rows.mark_calls == []


def test_end_unknown_session_is_404(monkeypatch):
    user_id = uuid4()
    _wire(monkeypatch, _FakeRows([]))

    response = TestClient(_app(user_id)).post("/api/v1/sessions/missing/end")

    assert response.status_code == 404


def test_end_is_owner_only(monkeypatch):
    """A row that belongs to another user is invisible here, even with dataset grants."""
    owner, caller = uuid4(), uuid4()
    rows = _FakeRows([_row("s1", owner)])
    _wire(monkeypatch, rows)

    response = TestClient(_app(caller)).post("/api/v1/sessions/s1/end")

    assert response.status_code == 404
    assert rows.mark_calls == []


def test_end_rejects_non_terminal_status(monkeypatch):
    user_id = uuid4()
    _wire(monkeypatch, _FakeRows([_row("s1", user_id)]))

    response = TestClient(_app(user_id)).post("/api/v1/sessions/s1/end", json={"status": "running"})

    assert response.status_code == 422
