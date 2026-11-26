"""The fastembed engine runs its model off the event loop, one batch at a time.

fastembed runs in this process and its ``embed`` call blocks until the batch is
done: about a second for 32 chunks on a small model, more than ten on a large one.
Called inside the coroutine, that stopped every other request for as long.
"""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
    FastembedEmbeddingEngine,
)


class _Model:
    """A fastembed model whose batch finishes only when the test lets it."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.threads: list[threading.Thread] = []
        self.running = 0
        self.most_running = 0
        self._counting = threading.Lock()

    def embed(self, texts, batch_size, parallel):
        with self._counting:
            self.running += 1
            self.most_running = max(self.most_running, self.running)
            self.threads.append(threading.current_thread())
        self.started.set()
        assert self.release.wait(timeout=5), "nothing let the batch finish"
        with self._counting:
            self.running -= 1
        return [np.array([float(len(text)), 0.0, 0.0]) for text in texts]


def _engine(model: _Model) -> FastembedEmbeddingEngine:
    """An engine around ``model``, without loading a real one."""
    engine = FastembedEmbeddingEngine.__new__(FastembedEmbeddingEngine)
    engine.model = "a-model"
    engine.dimensions = 3
    engine.mock = False
    engine.embedding_model = model
    engine.tokenizer = SimpleNamespace(count_tokens=lambda text: 1)
    return engine


@pytest.mark.asyncio
async def test_the_event_loop_runs_while_the_model_embeds():
    model = _Model()
    call = asyncio.create_task(_engine(model).embed_text(["a chunk"]))

    assert await asyncio.to_thread(model.started.wait, 5), "the model was never called"
    # The batch is running and cannot finish by itself. With the model on the event
    # loop this coroutine would not run again until it had, so nothing could let it.
    assert not call.done()
    model.release.set()
    result = await asyncio.wait_for(call, timeout=5)

    assert list(result) == [[7.0, 0.0, 0.0]]
    assert model.threads[0] is not threading.current_thread()


@pytest.mark.asyncio
async def test_batches_take_turns():
    """The model uses every core for one batch; a second one at once only adds memory."""
    model = _Model()
    engine = _engine(model)
    calls = asyncio.gather(engine.embed_text(["one"]), engine.embed_text(["two", "three"]))

    assert await asyncio.to_thread(model.started.wait, 5)
    await asyncio.sleep(0.05)  # room for the other batch to start, if it could
    assert model.most_running == 1
    model.release.set()
    first, second = await asyncio.wait_for(calls, timeout=5)

    assert model.most_running == 1
    assert len(first) == 1 and len(second) == 2
    assert len(set(model.threads)) == 1


@pytest.mark.asyncio
async def test_a_batch_runs_with_the_callers_context_variables():
    """Each call may carry its own settings; the worker thread must see the caller's."""
    from contextvars import ContextVar

    request = ContextVar("request", default=None)
    seen = []

    class _Recording(_Model):
        def embed(self, texts, batch_size, parallel):
            seen.append(request.get())
            return super().embed(texts, batch_size, parallel)

    model = _Recording()
    model.release.set()
    request.set("this request")

    await _engine(model).embed_text(["a chunk"])

    assert seen == ["this request"]
