"""Token sink: the side-channel that carries answer deltas out of the LLM call.

Streaming is never allowed to be load-bearing — the adapter still returns the
complete string, every existing signature is unchanged, and a request with no
sink installed behaves exactly as it did before. These tests pin that, plus the
properties that are easy to regress: only the answer call may stream, only one
lane's tokens reach the consumer, a retry must not leave two copies of the
answer, and nothing the request does may hang a consumer or leak provider detail
to it.
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.streaming.token_sink import (
    TokenSink,
    get_active_token_sink,
    requested_token_sink,
    answer_scope,
)


async def _drain(sink: TokenSink) -> list:
    return [event async for event in sink]


def _flag(enabled: bool):
    """Toggle LLM_ANSWER_STREAMING. It defaults to False, so every test that
    expects promotion has to turn it on explicitly — which is the point."""
    return patch(
        "cognee.infrastructure.llm.config.get_llm_context_config",
        return_value=SimpleNamespace(llm_answer_streaming=enabled),
    )


@contextmanager
def _requested(sink: TokenSink):
    """Install the sink the way a request entry point would."""
    token = requested_token_sink.set(sink)
    try:
        yield sink
    finally:
        requested_token_sink.reset(token)


# --------------------------- the sink itself ---------------------------


@pytest.mark.asyncio
async def test_deltas_reach_the_consumer_in_order():
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with answer_scope():
            sink.put_delta("Hello")
            sink.put_delta(" world")

    events = await _drain(sink)
    assert [e.type for e in events] == ["delta", "delta", "answer_done"]
    assert "".join(e.text for e in events if e.type == "delta") == "Hello world"


@pytest.mark.asyncio
async def test_retry_emits_reset_so_the_answer_is_not_duplicated():
    """Tenacity retries the whole call, which re-streams from the start."""
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with answer_scope():
            sink.begin_attempt()  # first attempt: nothing emitted yet, no reset
            sink.put_delta("par")
            sink.put_delta("tial")
            sink.begin_attempt()  # retry: must tell the consumer to discard
            sink.put_delta("complete")

    types = [e.type for e in await _drain(sink)]
    assert types == ["delta", "delta", "reset", "delta", "answer_done"]


@pytest.mark.asyncio
async def test_only_the_owning_lane_may_reset_the_consumer():
    """Every promoted lane calls begin_attempt on its way into the adapter. If a
    non-owner could reset, a second dataset's answer call would tell the client
    to discard the first one's answer — and its own deltas are then dropped, so
    nothing replaces what was wiped."""
    sink = TokenSink()

    async def _owner():
        async with answer_scope():
            for _ in range(3):
                sink.put_delta("A")
                await asyncio.sleep(0)

    async def _intruder():
        async with answer_scope():
            await asyncio.sleep(0)
            sink.begin_attempt()

    with _flag(True), _requested(sink):
        await asyncio.gather(_owner(), _intruder())

    events = await _drain(sink)
    assert "reset" not in [e.type for e in events]
    assert "".join(e.text for e in events if e.type == "delta") == "AAA"


@pytest.mark.asyncio
async def test_detach_stops_buffering_without_killing_the_producer():
    """A client that disconnects must not cost the turn: the LLM call keeps
    running and the session write still lands, we just stop buffering."""
    sink = TokenSink()
    sink.put_delta("seen")
    sink.detach()
    sink.put_delta("unseen")
    sink.close()

    assert [e.text for e in await _drain(sink) if e.type == "delta"] == ["seen"]


@pytest.mark.asyncio
async def test_a_detached_producer_still_counts_as_having_streamed():
    """`_emitted` records what was produced, not what was delivered. If a
    disconnect reset it, ownership would pass to another lane mid-answer."""
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with answer_scope():
            sink.detach()
            sink.put_delta("produced but not delivered")

    assert sink.emitted_any is True


@pytest.mark.asyncio
async def test_the_delta_buffer_is_bounded_when_nobody_drains():
    """A sink whose consumer never arrives must not retain one object per token
    for the life of the request."""
    sink = TokenSink(max_buffered_events=8)
    with _flag(True), _requested(sink):
        async with answer_scope():
            for index in range(100):
                sink.put_delta(f"token{index}")

    events = await _drain(sink)
    assert len([e for e in events if e.type == "delta"]) <= 8


@pytest.mark.asyncio
async def test_control_events_survive_a_full_buffer():
    """Dropping one of these is worse than the memory it saves: a lost `error`
    ends the stream cleanly on a failed request, so the client renders a
    truncated partial answer as if it were complete."""
    sink = TokenSink(max_buffered_events=3)
    for index in range(20):
        sink.put_delta(f"token{index}")
    sink.fail("upstream exploded")

    types = [e.type for e in await _drain(sink)]
    assert "error" in types, types


@pytest.mark.asyncio
async def test_a_second_consumer_is_refused_rather_than_splitting_the_stream():
    """Two iterators over one queue would each get an arbitrary half of the
    deltas, and only one would ever see the sentinel."""
    sink = TokenSink()
    sink.close()
    sink.__aiter__()
    with pytest.raises(RuntimeError, match="single consumer"):
        sink.__aiter__()


# --------------------------- promotion scoping ---------------------------


@pytest.mark.asyncio
async def test_no_sink_active_by_default():
    """Every LLM call in the codebase sees None unless deliberately promoted."""
    assert get_active_token_sink() is None


@pytest.mark.asyncio
async def test_answer_scope_is_a_noop_without_a_requested_sink():
    """The flag being off, or no caller listening, must change nothing."""
    async with answer_scope():
        assert get_active_token_sink() is None


@pytest.mark.asyncio
async def test_promotion_is_scoped_to_the_block():
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with answer_scope(stage="generating"):
            assert get_active_token_sink() is sink
        assert get_active_token_sink() is None


@pytest.mark.asyncio
async def test_only_the_answer_task_can_stream():
    """THE isolation invariant.

    The live cloud path runs asyncio.gather(analyze_turn, _retrieve_and_answer):
    two concurrent LLM calls. gather snapshots context per task, so a sink
    promoted inside the answer task is structurally invisible to the analysis
    task. If this regresses, SessionTurnAnalysis output interleaves into the
    user's visible answer.
    """
    sink = TokenSink()
    seen = {}

    async def _answer_lane():
        async with answer_scope():
            await asyncio.sleep(0)  # force interleaving with the sibling task
            seen["answer"] = get_active_token_sink()

    async def _analysis_lane():
        await asyncio.sleep(0)
        seen["analysis"] = get_active_token_sink()

    with _flag(True), _requested(sink):
        await asyncio.gather(_answer_lane(), _analysis_lane())

    assert seen["answer"] is sink
    assert seen["analysis"] is None, "analysis lane must never see the sink"


@pytest.mark.asyncio
async def test_one_lane_owns_the_stream_and_the_others_are_dropped():
    """Multi-dataset recall runs one answer call per dataset. Their deltas must
    not interleave into one incoherent answer."""
    sink = TokenSink()

    async def _lane(word: str):
        async with answer_scope():
            await asyncio.sleep(0)
            for _ in range(3):
                sink.put_delta(word)
                await asyncio.sleep(0)

    with _flag(True), _requested(sink):
        await asyncio.gather(_lane("A"), _lane("B"), _lane("C"))
    sink.close()

    streamed = "".join(e.text for e in await _drain(sink) if e.type == "delta")
    assert streamed in ("AAA", "BBB", "CCC"), streamed


@pytest.mark.asyncio
async def test_a_lane_that_cannot_stream_does_not_starve_one_that_can():
    """Whether a call *can* stream is unknown until it reaches the adapter — a
    structured response_model never streams, and not every adapter has a
    plain-text path. Claiming on entry would let such a lane lock out a
    streamable one and the request would emit nothing."""
    sink = TokenSink()

    async def _cannot_stream():
        async with answer_scope():
            await asyncio.sleep(0.01)  # promoted first, never emits

    async def _can_stream():
        async with answer_scope():
            await asyncio.sleep(0)
            sink.put_delta("Neon won.")

    with _flag(True), _requested(sink):
        await asyncio.gather(_cannot_stream(), _can_stream())
    sink.close()

    assert "".join(e.text for e in await _drain(sink) if e.type == "delta") == "Neon won."


@pytest.mark.asyncio
async def test_stage_marks_the_boundary_once_even_with_several_lanes():
    """The stage is how a consumer times retrieval→generation; seeing it twice
    for one answer measures the wrong interval."""
    sink = TokenSink()

    async def _lane():
        async with answer_scope(stage="generating"):
            await asyncio.sleep(0)

    with _flag(True), _requested(sink):
        await asyncio.gather(_lane(), _lane(), _lane())
    sink.close()

    events = await _drain(sink)
    assert [e.type for e in events] == ["stage"]
    assert events[0].stage == "generating"


# --------------- lifecycle: termination and failure ---------------


@pytest.mark.asyncio
async def test_successful_stream_terminates_with_answer_done():
    """The consumer sees the answer end without waiting for persistence."""
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with answer_scope():
            sink.put_delta("hi")

    # Terminates on its own — draining must not block.
    assert [e.type for e in await _drain(sink)] == ["delta", "answer_done"]


@pytest.mark.asyncio
async def test_a_failure_is_reported_without_leaking_provider_detail():
    """Provider errors embed the rendered prompt — i.e. the whole retrieved
    graph context — plus endpoints and request bodies. None of that may reach
    an HTTP client, so the event carries a type, and the detail goes to the log.
    """
    sink = TokenSink()
    secret = "ContentPolicyViolation: <entire graph context and api_base>"
    with _flag(True), _requested(sink):
        with pytest.raises(RuntimeError):
            async with answer_scope():
                sink.put_delta("half")
                raise RuntimeError(secret)

    events = await _drain(sink)
    assert [e.type for e in events] == ["delta", "error"]
    assert secret not in events[-1].error
    assert "RuntimeError" in events[-1].error


@pytest.mark.asyncio
async def test_cancellation_is_not_reported_as_a_failure():
    """str(CancelledError()) is "", so reporting it renders a blank error — and
    it defeats detach(), whose whole purpose is that a vanished consumer does
    not become a visible failure."""
    sink = TokenSink()
    with _flag(True), _requested(sink):
        with pytest.raises(asyncio.CancelledError):
            async with answer_scope():
                sink.put_delta("half")
                raise asyncio.CancelledError()

    sink.close()
    assert [e.type for e in await _drain(sink)] == ["delta"]


@pytest.mark.asyncio
async def test_a_call_that_streamed_nothing_leaves_the_sink_for_the_creator():
    """Not every call can stream, and a later one in the same request may. The
    creator owns terminating the sink — the transport does this in a finally."""
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with answer_scope():
            pass
        assert sink.emitted_any is False

    sink.close()  # what api/v1/recall/recall_stream.py does
    assert await _drain(sink) == []


@pytest.mark.asyncio
async def test_flag_off_emits_nothing_at_all():
    """Checked at the promotion site, not in the adapter: the two can disagree,
    and a promotion the adapter then refuses to honour emits a stage and no
    tokens."""
    sink = TokenSink()
    with _flag(False), _requested(sink):
        async with answer_scope(stage="generating"):
            assert get_active_token_sink() is None

    sink.close()
    assert await _drain(sink) == []
