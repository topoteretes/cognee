"""Auto-improve debounce for ``remember(session_id=...)`` (plan item B6).

The bridge into the permanent graph fires only when at least
``IMPROVE_DEBOUNCE_ENTRIES`` new session entries accumulated since the last
automatic improve, or ``IMPROVE_DEBOUNCE_SECONDS`` elapsed. State is an
internal session-context row in the session cache, the same pattern as
``session_persist_watermark``.
"""

import importlib
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.improve.config import ImproveConfig
from cognee.modules.improve.result import ImproveResult

remember_module = importlib.import_module("cognee.api.v1.remember.remember")
debounce_module = importlib.import_module("cognee.api.v1.remember.auto_improve_debounce")
improve_pkg = importlib.import_module("cognee.api.v1.improve")


class FakeSessionManager:
    """In-memory stand-in for the session cache: QA entries + context rows."""

    is_available = True

    def __init__(self):
        self.qa: dict[tuple, list] = {}
        self.context: dict[tuple, list] = {}
        self.reads = 0

    async def add_qa(self, *, user_id, session_id, question, context, answer, **kwargs):
        self.qa.setdefault((user_id, session_id), []).append(answer)
        return str(uuid4())

    async def get_session(self, *, user_id, session_id=None, **kwargs):
        self.reads += 1
        return [SimpleNamespace(answer=a) for a in self.qa.get((user_id, session_id), [])]

    async def get_session_context_entries(self, *, user_id, session_id=None):
        self.reads += 1
        return list(self.context.get((user_id, session_id), []))

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        for row in self.context.get((user_id, session_id), []):
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        self.context.setdefault((user_id, session_id), []).append(dict(entry_dump))
        return True


@pytest.fixture
def fake_sm(monkeypatch):
    sm = FakeSessionManager()
    # The session package re-exports get_session_manager, shadowing the
    # submodule — resolve the submodule explicitly and patch its attribute.
    sm_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")
    monkeypatch.setattr(sm_module, "get_session_manager", lambda: sm)

    async def _noop_setup():
        return None

    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", _noop_setup)
    return sm


@pytest.fixture
def improve_calls(monkeypatch):
    calls = []

    async def fake_improve(**kwargs):
        calls.append(kwargs)
        return ImproveResult(stages=[], memify_run={})

    monkeypatch.setattr(improve_pkg, "improve", fake_improve)
    return calls


def _config(monkeypatch, **overrides):
    config = ImproveConfig(**overrides)
    monkeypatch.setattr(debounce_module, "get_improve_config", lambda: config)
    return config


async def _remember_n(n: int, user, session_id="s-debounce"):
    results = []
    for i in range(n):
        result = await remember_module.remember(
            f"turn {i}", dataset_id=uuid4(), session_id=session_id, user=user
        )
        await result  # let the bridge task finish before the next call
        results.append(result)
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize("n,entries", [(7, 3), (10, 5), (4, 4), (5, 1)])
async def test_debounce_fires_at_most_ceil_n_over_entries(
    monkeypatch, fake_sm, improve_calls, n, entries
):
    _config(monkeypatch, debounce_entries=entries)
    user = SimpleNamespace(id=uuid4())

    results = await _remember_n(n, user)

    assert len(improve_calls) == math.ceil(n / entries)
    assert all(result.status == "session_stored" for result in results)
    # Every call stored its entry regardless of whether the bridge fired.
    assert len(fake_sm.qa[(str(user.id), "s-debounce")]) == n
    # The first remember always fires (no state yet); later ones every `entries`.
    fired = [result._task is not None for result in results]
    assert fired[0] is True
    assert fired == [i % entries == 0 for i in range(n)]


@pytest.mark.asyncio
async def test_default_config_fires_every_time_and_reads_nothing(
    monkeypatch, fake_sm, improve_calls
):
    _config(monkeypatch)  # entries=1, seconds=0
    user = SimpleNamespace(id=uuid4())

    await _remember_n(4, user)

    assert len(improve_calls) == 4
    assert fake_sm.reads == 0
    assert fake_sm.context == {}


@pytest.mark.asyncio
async def test_debounced_call_leaves_improve_unset(monkeypatch, fake_sm, improve_calls):
    _config(monkeypatch, debounce_entries=3)
    user = SimpleNamespace(id=uuid4())

    first, second = await _remember_n(2, user)

    assert first._task is not None
    assert first.improve is not None
    assert second._task is None
    assert second.improve is None
    assert second.improve_error is None
    assert len(improve_calls) == 1


@pytest.mark.asyncio
async def test_time_trigger_fires_after_debounce_seconds(monkeypatch, fake_sm):
    _config(monkeypatch, debounce_entries=0, debounce_seconds=60)
    user_id, session_id = str(uuid4()), "s-time"
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)

    fake_sm.qa[(user_id, session_id)] = ["a"]
    first = await debounce_module.should_auto_improve(fake_sm, user_id, session_id, now=now)
    assert first.due and first.reason == debounce_module.REASON_FIRST_RUN

    await debounce_module.mark_auto_improve_fired(
        fake_sm, user_id, session_id, qa_count=first.qa_count, now=now
    )
    fake_sm.qa[(user_id, session_id)].extend(["b", "c", "d"])

    soon = await debounce_module.should_auto_improve(
        fake_sm, user_id, session_id, now=now + timedelta(seconds=30)
    )
    assert not soon.due
    assert soon.reason == debounce_module.REASON_DEBOUNCED
    assert soon.new_entries == 3  # entry trigger is off, so entries alone never fire

    later = await debounce_module.should_auto_improve(
        fake_sm, user_id, session_id, now=now + timedelta(seconds=61)
    )
    assert later.due and later.reason == debounce_module.REASON_ELAPSED


@pytest.mark.asyncio
async def test_state_row_uses_internal_context_row_pattern(fake_sm, monkeypatch):
    _config(monkeypatch, debounce_entries=2)
    user_id, session_id = str(uuid4()), "s-row"
    fake_sm.qa[(user_id, session_id)] = ["a", "b"]

    await debounce_module.mark_auto_improve_fired(fake_sm, user_id, session_id)
    rows = fake_sm.context[(user_id, session_id)]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == debounce_module.AUTO_IMPROVE_STATE_ID
    assert row["kind"] == debounce_module.AUTO_IMPROVE_STATE_KIND
    assert row["qa_count"] == 2
    assert datetime.fromisoformat(row["last_improve_at"]).tzinfo is not None

    # A second fire merges into the same row instead of appending another.
    fake_sm.qa[(user_id, session_id)].append("c")
    await debounce_module.mark_auto_improve_fired(fake_sm, user_id, session_id)
    assert len(fake_sm.context[(user_id, session_id)]) == 1
    assert fake_sm.context[(user_id, session_id)][0]["qa_count"] == 3


@pytest.mark.asyncio
async def test_cleared_session_counts_all_entries_as_new(fake_sm, monkeypatch):
    _config(monkeypatch, debounce_entries=3)
    user_id, session_id = str(uuid4()), "s-cleared"
    fake_sm.context[(user_id, session_id)] = [
        {
            "id": debounce_module.AUTO_IMPROVE_STATE_ID,
            "kind": debounce_module.AUTO_IMPROVE_STATE_KIND,
            "qa_count": 10,
            "last_improve_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    fake_sm.qa[(user_id, session_id)] = ["x", "y", "z"]

    decision = await debounce_module.should_auto_improve(fake_sm, user_id, session_id)
    assert decision.due
    assert decision.reason == debounce_module.REASON_ENTRIES
    assert decision.new_entries == 3


@pytest.mark.asyncio
async def test_state_read_failure_fails_open(monkeypatch):
    _config(monkeypatch, debounce_entries=5)

    class BrokenSessionManager:
        is_available = True

        async def get_session_context_entries(self, **kwargs):
            raise ConnectionError("cache down")

        async def get_session(self, **kwargs):
            return []

    decision = await debounce_module.should_auto_improve(BrokenSessionManager(), "u", "s")
    assert decision.due
    assert decision.reason == debounce_module.REASON_STATE_UNAVAILABLE
