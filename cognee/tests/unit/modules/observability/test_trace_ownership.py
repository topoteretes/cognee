"""Trace ownership in the in-memory span buffer (``bind_trace_owner``).

Every test binds owners inside its own ``contextvars`` context, the way each
request runs in its own task, so no owner leaks from one test into the next.
"""

import contextvars
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.observability.tracing import (
    CogneeSpanExporter,
    bind_trace_owner,
    current_trace_owner,
)

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


@pytest.fixture
def buffered():
    exporter = CogneeSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider.get_tracer("test")
    provider.shutdown()


def _user():
    return SimpleNamespace(id=uuid4())


def _run_trace(tracer, owner, *span_names):
    """One new trace (root plus children) finished in a context bound to ``owner``."""

    def run():
        if owner is not None:
            bind_trace_owner(owner)
        token = otel_context.attach(otel_context.Context())
        try:
            with tracer.start_as_current_span(span_names[0]) as root:
                for name in span_names[1:]:
                    with tracer.start_as_current_span(name):
                        pass
            return format(root.get_span_context().trace_id, "032x")
        finally:
            otel_context.detach(token)

    return contextvars.copy_context().run(run)


def test_bind_trace_owner_sets_the_owner_for_the_current_context_only():
    user = _user()

    def run():
        bind_trace_owner(user)
        return current_trace_owner()

    assert contextvars.copy_context().run(run) == str(user.id)
    assert current_trace_owner() is None


def test_a_trace_is_returned_to_its_owner_only(buffered):
    exporter, tracer = buffered
    alice, bob = _user(), _user()
    alice_trace = _run_trace(tracer, alice, "cognee.api.recall", "llm.call")
    bob_trace = _run_trace(tracer, bob, "cognee.api.search")

    assert set(exporter.get_traces_owned_by({str(alice.id)})) == {alice_trace}
    assert set(exporter.get_traces_owned_by({str(bob.id)})) == {bob_trace}
    assert set(exporter.get_traces_owned_by({str(alice.id), str(bob.id)})) == {
        alice_trace,
        bob_trace,
    }


def test_a_trace_with_no_owner_is_returned_to_nobody(buffered):
    exporter, tracer = buffered
    alice = _user()
    _run_trace(tracer, None, "startup.recovery")

    assert exporter.get_traces_owned_by({str(alice.id)}) == {}
    # Still in the buffer for the SDK's own readers.
    assert len(exporter.get_all_traces()) == 1


def test_a_trace_with_another_owner_among_its_owners_is_hidden(buffered):
    exporter, tracer = buffered
    alice, bob = _user(), _user()
    trace_id = _run_trace(tracer, alice, "root")
    # A second span of the same trace, finished under Bob.
    exporter._owners[trace_id].add(str(bob.id))

    assert exporter.get_traces_owned_by({str(alice.id)}) == {}
    assert set(exporter.get_traces_owned_by({str(alice.id), str(bob.id)})) == {trace_id}


def test_owners_are_evicted_and_cleared_with_their_traces(buffered, monkeypatch):
    from cognee.modules.observability import tracing as tracing_module

    monkeypatch.setattr(tracing_module, "_MAX_TRACES", 2)
    exporter, tracer = buffered
    alice = _user()
    first = _run_trace(tracer, alice, "one")
    _run_trace(tracer, alice, "two")
    _run_trace(tracer, alice, "three")

    assert first not in exporter._owners
    assert len(exporter._owners) == 2

    exporter.clear()
    assert exporter._owners == {}
