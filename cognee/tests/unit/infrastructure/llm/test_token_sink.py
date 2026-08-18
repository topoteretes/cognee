"""Token sink: the side-channel that carries answer deltas out of the LLM call.

Streaming is never allowed to be load-bearing — the adapter still returns the
complete string, every existing signature is unchanged, and a request with no
sink installed behaves exactly as it did before. These tests pin that, plus the
three properties that are easy to regress: only the answer call may stream, only
one producer may claim a sink, and a retry must not leave the consumer holding
two copies of the answer.
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.streaming.token_sink import (
    TokenSink,
    active_token_sink,
    get_active_token_sink,
    requested_token_sink,
    stream_answer_tokens,
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
    sink.put_delta("Hello")
    sink.put_delta(" world")
    sink.answer_done()
    sink.close()

    events = await _drain(sink)
    assert [e.type for e in events] == ["delta", "delta", "answer_done"]
    assert "".join(e.text for e in events if e.type == "delta") == "Hello world"


@pytest.mark.asyncio
async def test_claim_admits_exactly_one_producer():
    """search_in_datasets_context runs one answer call per dataset. Without a
    single claim they would interleave into one incoherent stream."""
    sink = TokenSink()
    assert sink.try_claim() is True
    assert sink.try_claim() is False


@pytest.mark.asyncio
async def test_retry_emits_reset_so_the_answer_is_not_duplicated():
    """Tenacity retries the whole call, which re-streams from the start."""
    sink = TokenSink()
    sink.begin_attempt()  # first attempt: nothing emitted yet, no reset
    sink.put_delta("par")
    sink.put_delta("tial")
    sink.begin_attempt()  # retry: must tell the consumer to discard
    sink.put_delta("complete")
    sink.close()

    types = [e.type for e in await _drain(sink)]
    assert types == ["delta", "delta", "reset", "delta"]


@pytest.mark.asyncio
async def test_detach_stops_buffering_without_killing_the_producer():
    """A client that disconnects must not cost the turn: the LLM call keeps
    running and the session write still lands, we just stop buffering."""
    sink = TokenSink()
    sink.put_delta("seen")
    sink.detach()
    sink.put_delta("unseen")
    sink.close()

    texts = [e.text for e in await _drain(sink) if e.type == "delta"]
    assert texts == ["seen"]


@pytest.mark.asyncio
async def test_failure_is_reported_then_the_stream_closes():
    sink = TokenSink()
    sink.put_delta("half an ans")
    sink.fail(RuntimeError("upstream exploded"))

    events = await _drain(sink)
    assert [e.type for e in events] == ["delta", "error"]
    assert "upstream exploded" in events[-1].error


# --------------------------- promotion scoping ---------------------------


@pytest.mark.asyncio
async def test_no_sink_active_by_default():
    """Every LLM call in the codebase sees None unless deliberately promoted."""
    assert get_active_token_sink() is None


@pytest.mark.asyncio
async def test_stream_answer_tokens_is_a_noop_without_a_requested_sink():
    """The flag being off, or no caller listening, must change nothing."""
    async with stream_answer_tokens():
        assert get_active_token_sink() is None


@pytest.mark.asyncio
async def test_promotion_is_scoped_to_the_block():
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with stream_answer_tokens(stage="generating"):
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
        async with stream_answer_tokens():
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
async def test_second_concurrent_answer_call_does_not_stream():
    """Multi-dataset recall: one dataset streams, the rest run normally and
    still contribute to the final payload."""
    sink = TokenSink()
    promoted = []

    async def _lane():
        async with stream_answer_tokens():
            await asyncio.sleep(0)
            promoted.append(get_active_token_sink() is sink)

    with _flag(True), _requested(sink):
        await asyncio.gather(_lane(), _lane(), _lane())

    assert promoted.count(True) == 1, promoted


@pytest.mark.asyncio
async def test_stage_marks_the_retrieval_to_generation_boundary():
    """Entering promotion emits the stage, which is what makes it possible to
    measure how much of the wait is retrieval that streaming cannot help."""
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with stream_answer_tokens(stage="generating"):
            pass
    sink.close()

    events = await _drain(sink)
    assert events[0].type == "stage"
    assert events[0].stage == "generating"


# --------------- lifecycle: the sink must always terminate ---------------
#
# The consumer's `async for` only ends on a sentinel. Anything that leaves the
# answer call without emitting one hangs it until the HTTP client times out.


@pytest.mark.asyncio
async def test_successful_stream_terminates_with_answer_done():
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with stream_answer_tokens():
            sink.put_delta("hi")

    # Terminates on its own — draining must not block.
    types = [e.type for e in await _drain(sink)]
    assert types == ["delta", "answer_done"]


@pytest.mark.asyncio
async def test_failed_answer_call_reports_an_error_and_closes():
    """Without this the consumer hangs forever after an LLM failure."""
    sink = TokenSink()
    with _flag(True), _requested(sink):
        with pytest.raises(RuntimeError, match="llm exploded"):
            async with stream_answer_tokens():
                sink.put_delta("half")
                raise RuntimeError("llm exploded")

    events = await _drain(sink)
    assert [e.type for e in events] == ["delta", "error"]
    assert "llm exploded" in events[-1].error


@pytest.mark.asyncio
async def test_call_that_could_not_stream_gives_the_claim_back():
    """Most adapters have no plain-text streaming path, and a structured
    response_model never streams. Holding the claim would silently disable
    streaming for the rest of the request."""
    sink = TokenSink()
    with _flag(True), _requested(sink):
        async with stream_answer_tokens():
            pass  # nothing streamed
        assert sink.emitted_any is False

        async with stream_answer_tokens():  # a later call can still claim it
            sink.put_delta("streamed this time")

    assert [e.type for e in await _drain(sink)] == ["delta", "answer_done"]


@pytest.mark.asyncio
async def test_flag_off_emits_nothing_at_all():
    """Checked at the promotion site, not only in the adapter: otherwise a
    flag-off request still claimed the sink and emitted a stage event."""
    sink = TokenSink()
    with _flag(False), _requested(sink):
        async with stream_answer_tokens(stage="generating"):
            assert get_active_token_sink() is None

    sink.close()
    assert await _drain(sink) == []


@pytest.mark.asyncio
async def test_retry_keeps_the_streamed_signal_so_the_claim_is_not_returned():
    """begin_attempt() resets what the consumer shows, not whether we streamed."""
    sink = TokenSink()
    assert sink.try_claim() is True
    sink.put_delta("attempt one")
    sink.begin_attempt()  # retry: resets the view, not the fact that we streamed

    assert sink.emitted_any is True
    sink.release_claim()
    assert sink.try_claim() is False, "a sink that streamed must stay claimed"
