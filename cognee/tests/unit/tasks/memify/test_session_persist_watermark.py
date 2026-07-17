"""Watermark-based incremental session persistence.

Covers the three contract guarantees:
1. Completeness — every session entry is ingested (nothing skipped).
2. Incrementality — already-persisted entries are never re-ingested.
3. Retry safety — a failed cognify leaves the watermark untouched so the
   same window is re-extracted on the next run.
"""

import sys
import uuid
from types import SimpleNamespace

import pytest

import cognee
from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.session_persist_watermark import (
    SessionPersistWindow,
    get_persisted_qa_count,
    save_persisted_qa_count,
)
from cognee.tasks.memify.cognify_session import cognify_session
from cognee.tasks.memify.extract_user_sessions import extract_user_sessions

SESSION = "watermark_test_session"


class FakeSessionManager:
    """In-memory stand-in exposing the SessionManager surface the tasks use."""

    is_available = True

    def __init__(self):
        self.qa: dict[tuple[str, str], list[SessionQAEntry]] = {}
        self.context: dict[tuple[str, str], list[dict]] = {}

    def add_entry(self, user_id: str, session_id: str, question: str, answer: str):
        self.qa.setdefault((user_id, session_id), []).append(
            SessionQAEntry(
                time="2026-07-13T00:00:00+00:00",
                question=question,
                context="",
                answer=answer,
                qa_id=str(uuid.uuid4()),
            )
        )

    async def get_session(self, *, user_id, session_id=None, formatted=False, **_):
        return list(self.qa.get((user_id, session_id), []))

    async def get_session_context_entries(self, *, user_id, session_id=None):
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
def user():
    fake_user = SimpleNamespace(id=uuid.uuid4())
    token = session_user.set(fake_user)
    yield fake_user
    session_user.reset(token)


@pytest.fixture
def manager(user, monkeypatch):
    # Resolve the real modules via sys.modules: the package re-exports the
    # task functions under the same names, so dotted-path monkeypatching
    # would target the functions instead of the modules.
    extract_module = sys.modules["cognee.tasks.memify.extract_user_sessions"]
    cognify_module = sys.modules["cognee.tasks.memify.cognify_session"]

    fake = FakeSessionManager()
    monkeypatch.setattr(extract_module, "get_session_manager", lambda: fake)
    monkeypatch.setattr(cognify_module, "get_session_manager", lambda: fake)
    return fake


async def _extract_windows(session_ids) -> list[SessionPersistWindow]:
    return [window async for window in extract_user_sessions([{}], session_ids=session_ids)]


@pytest.mark.asyncio
async def test_fresh_session_extracts_all_entries(user, manager):
    user_id = str(user.id)
    manager.add_entry(user_id, SESSION, "q1", "a1")
    manager.add_entry(user_id, SESSION, "q2", "a2")

    windows = await _extract_windows([SESSION])

    assert len(windows) == 1
    assert "q1" in windows[0].text and "a1" in windows[0].text
    assert "q2" in windows[0].text and "a2" in windows[0].text
    assert windows[0].persisted_qa_count == 2
    assert windows[0].session_id == SESSION
    assert windows[0].user_id == user_id


@pytest.mark.asyncio
async def test_watermark_skips_already_persisted_entries(user, manager):
    user_id = str(user.id)
    for index in range(5):
        manager.add_entry(user_id, SESSION, f"q{index}", f"a{index}")
    await save_persisted_qa_count(manager, user_id, SESSION, 3)

    windows = await _extract_windows([SESSION])

    assert len(windows) == 1
    for old_index in range(3):
        assert f"q{old_index}" not in windows[0].text
    assert "q3" in windows[0].text and "q4" in windows[0].text
    assert windows[0].persisted_qa_count == 5


@pytest.mark.asyncio
async def test_fully_persisted_session_yields_nothing(user, manager):
    user_id = str(user.id)
    manager.add_entry(user_id, SESSION, "q1", "a1")
    await save_persisted_qa_count(manager, user_id, SESSION, 1)

    assert await _extract_windows([SESSION]) == []


@pytest.mark.asyncio
async def test_stale_watermark_resets_and_extracts_everything(user, manager):
    """A watermark above the entry count (cleared + rebuilt session) restarts from zero."""
    user_id = str(user.id)
    manager.add_entry(user_id, SESSION, "rebuilt-q", "rebuilt-a")
    await save_persisted_qa_count(manager, user_id, SESSION, 10)

    windows = await _extract_windows([SESSION])

    assert len(windows) == 1
    assert "rebuilt-q" in windows[0].text
    assert windows[0].persisted_qa_count == 1


@pytest.mark.asyncio
async def test_watermark_roundtrip_and_update(user, manager):
    user_id = str(user.id)
    assert await get_persisted_qa_count(manager, user_id, SESSION) == 0
    await save_persisted_qa_count(manager, user_id, SESSION, 4)
    assert await get_persisted_qa_count(manager, user_id, SESSION) == 4
    await save_persisted_qa_count(manager, user_id, SESSION, 7)
    assert await get_persisted_qa_count(manager, user_id, SESSION) == 7
    # One row per session, updated in place — not one row per save.
    assert len(manager.context[(user_id, SESSION)]) == 1


@pytest.mark.asyncio
async def test_cognify_session_advances_watermark_on_success(user, manager, monkeypatch):
    user_id = str(user.id)

    async def fake_add(*args, **kwargs):
        return None

    async def fake_cognify(*args, **kwargs):
        return None

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", fake_cognify)

    window = SessionPersistWindow(
        user_id=user_id,
        session_id=SESSION,
        text="Question: q\n\nAnswer: a\n\n",
        persisted_qa_count=3,
    )
    await cognify_session(window, dataset_id=uuid.uuid4(), user=user)

    assert await get_persisted_qa_count(manager, user_id, SESSION) == 3


@pytest.mark.asyncio
async def test_cognify_session_keeps_watermark_on_failure(user, manager, monkeypatch):
    user_id = str(user.id)

    async def fake_add(*args, **kwargs):
        return None

    async def failing_cognify(*args, **kwargs):
        raise RuntimeError("LLM exploded")

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", failing_cognify)

    window = SessionPersistWindow(
        user_id=user_id,
        session_id=SESSION,
        text="Question: q\n\nAnswer: a\n\n",
        persisted_qa_count=3,
    )
    with pytest.raises(CogneeSystemError):
        await cognify_session(window, dataset_id=uuid.uuid4(), user=user)

    assert await get_persisted_qa_count(manager, user_id, SESSION) == 0


@pytest.mark.asyncio
async def test_multi_run_completeness_without_reingestion(user, manager, monkeypatch):
    """The user-facing guarantee, end to end over three grow->persist cycles:

    every entry lands in exactly one window (all data ingested, nothing
    skipped, nothing ingested twice), and an unchanged session is a no-op.
    """
    user_id = str(user.id)
    ingested_texts: list[str] = []

    async def fake_add(text, *args, **kwargs):
        ingested_texts.append(text)

    async def fake_cognify(*args, **kwargs):
        return None

    monkeypatch.setattr(cognee, "add", fake_add)
    monkeypatch.setattr(cognee, "cognify", fake_cognify)

    async def run_improve_cycle():
        for window in await _extract_windows([SESSION]):
            await cognify_session(window, dataset_id=uuid.uuid4(), user=user)

    all_questions = []
    for batch in ([0, 1], [2], [3, 4, 5]):
        for index in batch:
            manager.add_entry(user_id, SESSION, f"unique-question-{index}", f"answer-{index}")
            all_questions.append(f"unique-question-{index}")
        await run_improve_cycle()

    combined = "".join(ingested_texts)
    for question in all_questions:
        assert combined.count(question) == 1, (
            f"{question} ingested {combined.count(question)} times — "
            "must be exactly once (no gaps, no re-ingestion)"
        )

    # Unchanged session: a fourth improve cycle ingests nothing.
    ingested_before = len(ingested_texts)
    await run_improve_cycle()
    assert len(ingested_texts) == ingested_before

    # Retry semantics: a failed cognify re-extracts the same entries next run.
    manager.add_entry(user_id, SESSION, "unique-question-6", "answer-6")

    async def failing_cognify(*args, **kwargs):
        raise RuntimeError("transient failure")

    monkeypatch.setattr(cognee, "cognify", failing_cognify)
    with pytest.raises(CogneeSystemError):
        await run_improve_cycle()

    monkeypatch.setattr(cognee, "cognify", fake_cognify)
    await run_improve_cycle()
    combined = "".join(ingested_texts)
    # The failed attempt added its text before cognify raised, so the entry
    # appears once for the failed try and once for the successful retry —
    # what matters is the successful window covered it and the watermark
    # now marks it done.
    assert "unique-question-6" in combined
    assert await get_persisted_qa_count(manager, user_id, SESSION) == 7
