"""The chunk task's temporal hint pass is opt-out: the indexing pipeline skips it."""

import asyncio
import importlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

task_module = importlib.import_module("cognee.tasks.documents.extract_chunks_from_documents")


class _Chunk:
    def __init__(self, text: str):
        self.text = text
        self.chunk_size = 3
        self.belongs_to_set = None
        self._temporal_hints = None
        self.chunker_id = None


class _Document:
    id = "doc-1"
    belongs_to_set = ["set-a"]

    async def read(self, chunker_cls, max_chunk_size):
        yield _Chunk("Pierre left Moscow in 1812.")
        yield _Chunk("He returned the following spring.")


class _Chunker:
    def read(self):  # pragma: no cover - only its presence is checked
        pass


@pytest.mark.asyncio
async def test_hints_are_attached_by_default(monkeypatch):
    monkeypatch.setattr(task_module, "update_document_token_count", AsyncMock())
    monkeypatch.setattr(task_module, "hint_lines", lambda text, base: (["hint for " + text], base))

    chunks = [
        chunk
        async for chunk in task_module.extract_chunks_from_documents(
            [_Document()], max_chunk_size=10, chunker=_Chunker
        )
    ]

    assert [chunk._temporal_hints for chunk in chunks] == [
        ["hint for Pierre left Moscow in 1812."],
        ["hint for He returned the following spring."],
    ]


@pytest.mark.asyncio
async def test_indexing_run_skips_the_hint_pass(monkeypatch):
    """The indexing pipeline only stores and embeds chunks; the hints are never
    persisted and extraction recomputes them, so the dateparser pass is skipped."""
    monkeypatch.setattr(task_module, "update_document_token_count", AsyncMock())

    def forbidden(text, base):
        raise AssertionError("hint_lines must not run when temporal_hints is off")

    monkeypatch.setattr(task_module, "hint_lines", forbidden)

    chunks = [
        chunk
        async for chunk in task_module.extract_chunks_from_documents(
            [_Document()], max_chunk_size=10, chunker=_Chunker, temporal_hints=False
        )
    ]

    assert len(chunks) == 2
    assert all(chunk._temporal_hints is None for chunk in chunks)
    assert all(chunk.belongs_to_set == ["set-a"] for chunk in chunks)


@pytest.mark.asyncio
async def test_indexing_task_list_turns_the_hints_off():
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    tasks = await cognify_module.get_indexing_tasks(chunk_size=512)

    chunk_task = tasks[1]
    assert chunk_task.executable is task_module.extract_chunks_from_documents
    assert chunk_task.default_params["kwargs"]["temporal_hints"] is False


class _LongDocument(_Document):
    async def read(self, chunker_cls, max_chunk_size):
        for index in range(5):
            yield _Chunk(f"chunk {index}")


async def _drain_while_ticking(**kwargs) -> int:
    """Drain the task while another coroutine ticks; return how often it ticked."""
    ticks = 0
    done = False

    async def ticker():
        nonlocal ticks
        while not done:
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    ticks = 0
    async for _ in task_module.extract_chunks_from_documents(
        [_LongDocument()], max_chunk_size=10, chunker=_Chunker, **kwargs
    ):
        pass
    done = True
    await ticker_task
    return ticks


@pytest.mark.asyncio
async def test_hint_pass_runs_off_the_event_loop(monkeypatch):
    """A slow hint pass must not block other coroutines (DB handshakes, probes)."""
    monkeypatch.setattr(task_module, "update_document_token_count", AsyncMock())
    loop_thread = threading.get_ident()
    hint_threads = []

    def slow_hints(text, base):
        hint_threads.append(threading.get_ident())
        time.sleep(0.05)
        return [], base

    monkeypatch.setattr(task_module, "hint_lines", slow_hints)

    ticks = await _drain_while_ticking()

    assert loop_thread not in hint_threads
    assert ticks >= 5


@pytest.mark.asyncio
async def test_chunks_without_hints_still_yield_to_the_loop(monkeypatch):
    monkeypatch.setattr(task_module, "update_document_token_count", AsyncMock())

    ticks = await _drain_while_ticking(temporal_hints=False)

    assert ticks >= 5
