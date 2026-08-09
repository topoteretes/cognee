"""Race hardening: counter increments and extraction watermarks serialize per session."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.session import agent_context_extraction
from cognee.infrastructure.session.session_turn import apply_served_context_ratings


class _RacySessionManager:
    """Read-modify-write store with forced await points between read and write."""

    def __init__(self):
        self.entries = {
            "c1": {"id": "c1", "kind": "context", "helpful_count": 0, "harmful_count": 0}
        }

    async def get_session_context_entries(self, *, user_id, session_id):
        await asyncio.sleep(0)  # force interleaving between read and write
        return [dict(entry) for entry in self.entries.values()]

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        await asyncio.sleep(0)
        if entry_id not in self.entries:
            return False
        self.entries[entry_id] = {**self.entries[entry_id], **merge}
        return True


@pytest.mark.asyncio
async def test_concurrent_ratings_never_lose_increments():
    sm = _RacySessionManager()
    rating = SimpleNamespace(entry_id="c1", rating="helpful")

    await asyncio.gather(
        *[
            apply_served_context_ratings(
                sm, user_id="u", session_id="race-session", ratings=[rating]
            )
            for _ in range(10)
        ]
    )

    assert sm.entries["c1"]["helpful_count"] == 10


@pytest.mark.asyncio
async def test_concurrent_pending_extractions_do_not_duplicate_windows(monkeypatch):
    """Two concurrent drains over the same session must not run overlapping LLM windows."""
    processed_windows = []

    state = {"row": None}

    sm = SimpleNamespace()
    sm.get_agent_trace_count = AsyncMock(return_value=6)

    async def get_entries(*, user_id, session_id):
        await asyncio.sleep(0)
        return [state["row"]] if state["row"] else []

    async def update_entry(*, user_id, session_id, entry_id, merge):
        await asyncio.sleep(0)
        if state["row"] is None:
            return False
        state["row"] = {**state["row"], **merge}
        return True

    async def create_entry(*, user_id, session_id, entry_dump):
        state["row"] = dict(entry_dump)

    sm.get_session_context_entries = get_entries
    sm.update_session_context_entry = update_entry
    sm.create_session_context_entry = create_entry

    async def get_traces(*, user_id, session_id, last_n):
        await asyncio.sleep(0)
        return [SimpleNamespace(index=i) for i in range(last_n)]

    sm.get_agent_trace_session = get_traces

    async def fake_batch(*, session_manager, user_id, session_id, traces):
        processed_windows.append(len(traces))
        await asyncio.sleep(0)
        return []

    monkeypatch.setattr(agent_context_extraction, "_extract_batch_from_traces", fake_batch)

    await asyncio.gather(
        agent_context_extraction.extract_pending_agent_context(
            session_manager=sm,
            user_id="u",
            session_id="race-drain",
            min_new_traces=1,
            overlap=0,
            max_window=6,
        ),
        agent_context_extraction.extract_pending_agent_context(
            session_manager=sm,
            user_id="u",
            session_id="race-drain",
            min_new_traces=1,
            overlap=0,
            max_window=6,
        ),
    )

    # First drain processes all 6 pending traces in one window; the second sees
    # an up-to-date watermark and does nothing. Without the lock both would run.
    assert processed_windows == [6]
    assert state["row"]["processed_trace_count"] == 6
