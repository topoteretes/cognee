"""The graph engine handle must not keep an evicted engine open.

A handle can outlive its use inside a garbage reference cycle, e.g. when a
caught exception's traceback keeps the caller frames that hold it alive. A
handle that pinned its leased proxy strongly kept an evicted engine open until
the cyclic GC happened to run; for Ladybug that held the file lock, so the next
open of the same database failed with "Could not set lock on file".
"""

import contextlib
import gc
import importlib

import pytest

from cognee.infrastructure.databases.utils.closing_lru_cache import closing_lru_cache

# The package ``__init__`` re-exports a function with the same name as this
# submodule, so an attribute-based import resolves to the function.
graph_module = importlib.import_module("cognee.infrastructure.databases.graph.get_graph_engine")
Handle = graph_module._GraphEngineHandle


class _Closeable:
    """Engine stub that records close() and exposes async iteration/sessions."""

    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True

    async def rows(self):
        for row in ("x", "y"):
            yield row

    @contextlib.asynccontextmanager
    async def session(self):
        yield self.name

    def sessionmaker(self):
        self.made_session = _Session()
        return self.made_session


class _Session:
    """``async with``-capable resource, like SQLAlchemy's ``AsyncSession``."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    def execute(self):
        return "executed"


def _cached_factory(monkeypatch):
    @closing_lru_cache(maxsize=1)
    def factory(name):
        return _Closeable(name)

    monkeypatch.setattr(
        graph_module, "create_graph_engine", lambda **config: factory(config["name"])
    )
    return factory


def test_handle_in_garbage_cycle_does_not_defer_close(monkeypatch):
    factory = _cached_factory(monkeypatch)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        handle = Handle({"name": "a"})
        engine = handle._engine().__wrapped__
        assert handle.name == "a"

        # Leave the handle reachable only from a reference cycle the cyclic GC
        # has not collected, like a traceback that keeps the caller frames alive.
        cycle = [handle]
        cycle.append(cycle)
        del handle, cycle

        factory("b")  # capacity-evicts "a"

        assert engine.closed
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()


def test_live_handle_reresolves_after_eviction(monkeypatch):
    factory = _cached_factory(monkeypatch)

    handle = Handle({"name": "a"})
    first = handle._engine().__wrapped__
    factory("b")  # evicts "a" while the handle is idle

    assert first.closed
    second = handle._engine().__wrapped__
    assert second is not first
    assert not second.closed


@pytest.mark.asyncio
async def test_async_iteration_holds_engine_open_across_eviction(monkeypatch):
    factory = _cached_factory(monkeypatch)

    handle = Handle({"name": "a"})
    engine = handle._engine().__wrapped__
    rows = handle.rows()
    assert await rows.__anext__() == "x"

    factory("b")  # evicts "a" mid-iteration

    assert not engine.closed
    assert [row async for row in rows] == ["y"]
    assert engine.closed


@pytest.mark.asyncio
async def test_async_context_holds_engine_open_across_eviction(monkeypatch):
    factory = _cached_factory(monkeypatch)

    handle = Handle({"name": "a"})
    engine = handle._engine().__wrapped__
    async with handle.session() as name:
        factory("b")  # evicts "a" inside the context

        assert name == "a"
        assert not engine.closed

    assert engine.closed


def test_async_context_resource_is_returned_unwrapped(monkeypatch):
    _cached_factory(monkeypatch)

    handle = Handle({"name": "a"})
    session = handle.sessionmaker()

    assert session is handle._engine().__wrapped__.made_session
    assert session.execute() == "executed"
