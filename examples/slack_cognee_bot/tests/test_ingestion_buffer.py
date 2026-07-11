"""Unit tests for the per-channel ingestion buffer (issue #3609, commit 3).

The adapter (ingest / flush / answer) is mocked — no cognee, no keys. These
tests exercise only the buffering + trigger policy.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.ingestion_buffer import IngestionBuffer
from src.memory_adapter import Answer, ConversationRef

REF_A = ConversationRef(team_id="T1", channel_id="A")
REF_B = ConversationRef(team_id="T1", channel_id="B")


def _fake_memory():
    memory = AsyncMock()
    memory.answer.return_value = Answer(text="answer", citations=[])
    return memory


def _add(buffer, ref, n, start=0):
    async def _run():
        for i in range(start, start + n):
            await buffer.add_message(
                ref, ts=f"{i}.0", text=f"msg {i}", permalink=f"https://x/{i}", author="alice"
            )

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# accumulation                                                                #
# --------------------------------------------------------------------------- #


def test_messages_accumulate_without_flushing_below_threshold():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=5)

    _add(buffer, REF_A, 3)

    assert memory.ingest.await_count == 3
    assert buffer.pending_count("A") == 3
    memory.flush.assert_not_awaited()


# --------------------------------------------------------------------------- #
# size-threshold trigger                                                       #
# --------------------------------------------------------------------------- #


def test_size_threshold_fires_flush_at_exact_count():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=3)

    # First two must not flush.
    _add(buffer, REF_A, 2)
    memory.flush.assert_not_awaited()
    assert buffer.pending_count("A") == 2

    # Third hits the threshold -> exactly one flush, counter resets.
    _add(buffer, REF_A, 1, start=2)
    memory.flush.assert_awaited_once()
    assert memory.flush.await_args.args[0] is REF_A
    assert buffer.pending_count("A") == 0


def test_second_batch_triggers_a_second_flush():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=2)

    _add(buffer, REF_A, 4)  # two full batches

    assert memory.ingest.await_count == 4
    assert memory.flush.await_count == 2
    assert buffer.pending_count("A") == 0


# --------------------------------------------------------------------------- #
# on-demand flush before answer                                               #
# --------------------------------------------------------------------------- #


def test_answer_flushes_pending_first():
    memory = _fake_memory()
    call_order = []
    memory.flush.side_effect = lambda ref: call_order.append(("flush", ref.channel_id))

    async def _answer_side_effect(ref, *, query):
        call_order.append(("answer", ref.channel_id))
        return Answer(text="answer", citations=[])

    memory.answer.side_effect = _answer_side_effect

    buffer = IngestionBuffer(memory, batch_size=100)
    _add(buffer, REF_A, 2)  # pending, below threshold
    assert buffer.pending_count("A") == 2

    result = asyncio.run(buffer.answer(REF_A, query="what did we decide?"))

    assert isinstance(result, Answer)
    # flush happened before answer, and pending was cleared.
    assert call_order == [("flush", "A"), ("answer", "A")]
    assert buffer.pending_count("A") == 0


def test_answer_with_no_pending_is_still_answered_without_flush():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=5)

    result = asyncio.run(buffer.answer(REF_A, query="q"))

    assert isinstance(result, Answer)
    memory.flush.assert_not_awaited()  # empty buffer -> no cognify
    memory.answer.assert_awaited_once()


# --------------------------------------------------------------------------- #
# per-channel isolation                                                        #
# --------------------------------------------------------------------------- #


def test_buffers_are_isolated_per_channel():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=3)

    _add(buffer, REF_A, 2)  # A: 2 pending
    _add(buffer, REF_B, 1)  # B: 1 pending

    assert buffer.pending_count("A") == 2
    assert buffer.pending_count("B") == 1
    memory.flush.assert_not_awaited()

    # Filling A to threshold flushes A only; B untouched.
    _add(buffer, REF_A, 1, start=2)
    assert memory.flush.await_count == 1
    assert memory.flush.await_args.args[0] is REF_A
    assert buffer.pending_count("A") == 0
    assert buffer.pending_count("B") == 1


def test_flush_one_channel_does_not_touch_another():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=100)

    _add(buffer, REF_A, 2)
    _add(buffer, REF_B, 2)

    asyncio.run(buffer.flush(REF_A))

    assert memory.flush.await_count == 1
    assert buffer.pending_count("A") == 0
    assert buffer.pending_count("B") == 2  # B still pending


# --------------------------------------------------------------------------- #
# empty-buffer flush is a clean no-op                                          #
# --------------------------------------------------------------------------- #


def test_empty_buffer_flush_is_noop():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=5)

    asyncio.run(buffer.flush(REF_A))

    memory.flush.assert_not_awaited()
    assert buffer.pending_count("A") == 0


def test_forget_clears_pending_and_delegates_to_adapter():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=100)

    _add(buffer, REF_A, 2)  # A has pending
    _add(buffer, REF_B, 1)
    assert buffer.pending_count("A") == 2

    asyncio.run(buffer.forget(REF_A))

    memory.forget.assert_awaited_once()
    assert memory.forget.await_args.args[0] is REF_A
    assert buffer.pending_count("A") == 0  # A's buffered state dropped
    assert buffer.pending_count("B") == 1  # B untouched


def test_flush_is_noop_again_immediately_after_flushing():
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=2)

    _add(buffer, REF_A, 2)  # triggers one flush, resets to 0
    assert memory.flush.await_count == 1

    asyncio.run(buffer.flush(REF_A))  # nothing pending now
    assert memory.flush.await_count == 1  # unchanged


def test_batch_size_floor_is_one():
    # A non-positive batch size would never flush on size; it is clamped to 1.
    memory = _fake_memory()
    buffer = IngestionBuffer(memory, batch_size=0)

    _add(buffer, REF_A, 1)

    memory.flush.assert_awaited_once()
    assert buffer.pending_count("A") == 0
