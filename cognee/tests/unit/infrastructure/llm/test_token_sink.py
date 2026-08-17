"""Token sink: the side-channel that carries answer deltas out of the LLM call.

Streaming is never allowed to be load-bearing — the adapter still returns the
complete string, every existing signature is unchanged, and a request with no
sink installed behaves exactly as it did before. These tests pin that, plus the
three properties that are easy to regress: only the answer call may stream, only
one producer may claim a sink, and a retry must not leave the consumer holding
two copies of the answer.
"""

import asyncio

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
    token = requested_token_sink.set(sink)
    try:
        async with stream_answer_tokens(stage="generating"):
            assert get_active_token_sink() is sink
        assert get_active_token_sink() is None
    finally:
        requested_token_sink.reset(token)


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

    token = requested_token_sink.set(sink)
    try:
        await asyncio.gather(_answer_lane(), _analysis_lane())
    finally:
        requested_token_sink.reset(token)

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

    token = requested_token_sink.set(sink)
    try:
        await asyncio.gather(_lane(), _lane(), _lane())
    finally:
        requested_token_sink.reset(token)

    assert promoted.count(True) == 1, promoted


@pytest.mark.asyncio
async def test_stage_marks_the_retrieval_to_generation_boundary():
    """Entering promotion emits the stage, which is what makes it possible to
    measure how much of the wait is retrieval that streaming cannot help."""
    sink = TokenSink()
    token = requested_token_sink.set(sink)
    try:
        async with stream_answer_tokens(stage="generating"):
            pass
    finally:
        requested_token_sink.reset(token)
    sink.close()

    events = await _drain(sink)
    assert events[0].type == "stage"
    assert events[0].stage == "generating"


@pytest.fixture(autouse=True)
def _reset_context():
    """ContextVars leak across tests in the same task otherwise."""
    yield
    active_token_sink.set(None)
    requested_token_sink.set(None)
