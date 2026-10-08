import asyncio
import importlib
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.tasks.documents.extract_chunks_from_documents import extract_chunks_from_documents

# the package re-exports the function under the module's name
module = importlib.import_module("cognee.tasks.documents.extract_chunks_from_documents")


class _Chunker:
    read = None


def _document(texts):
    async def read(max_chunk_size, chunker_cls):
        for text in texts:
            yield SimpleNamespace(text=text, chunk_size=1, belongs_to_set=None)

    return SimpleNamespace(id=uuid4(), belongs_to_set=None, read=read)


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    async def noop(document_id, token_count):
        return None

    monkeypatch.setattr(module, "update_document_token_count", noop)


async def _run(documents):
    return [c async for c in extract_chunks_from_documents(documents, 10, chunker=_Chunker)]


@pytest.mark.asyncio
async def test_hint_lines_runs_off_the_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    seen = []

    def fake_hint_lines(text, base):
        seen.append(threading.get_ident())
        return [], base

    monkeypatch.setattr(module, "hint_lines", fake_hint_lines)
    await _run([_document(["a", "b"])])
    assert seen and all(t != loop_thread for t in seen)


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_hinting(monkeypatch):
    import time

    def slow_hint_lines(text, base):
        time.sleep(0.2)
        return [], base

    monkeypatch.setattr(module, "hint_lines", slow_hint_lines)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await _run([_document(["a"])])
    task.cancel()
    assert ticks >= 10


@pytest.mark.asyncio
async def test_rolling_base_is_ordered_and_isolated_per_document():
    docs = [
        _document(["It began on 5 March 1951.", "The following 27 April it ended."]),
        _document(["The following 27 April it ended."]),
    ]
    chunks = await _run(docs)
    assert chunks[0]._temporal_hints == []
    assert chunks[1]._temporal_hints and "1951-04-27" in chunks[1]._temporal_hints[0]
    # a new document starts with no base, so nothing is inferred for it
    assert chunks[2]._temporal_hints == []


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch):
    import time

    def slow_hint_lines(text, base):
        time.sleep(0.3)
        return [], base

    monkeypatch.setattr(module, "hint_lines", slow_hint_lines)
    task = asyncio.create_task(_run([_document(["a", "b", "c"])]))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
