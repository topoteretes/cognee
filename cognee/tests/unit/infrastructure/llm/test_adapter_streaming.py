"""The adapter's streaming branch of ``acreate_str_output``.

The contract that matters: the method returns the same complete string whether
or not it streamed. Streaming only decides whether the tokens are *also* pushed
to a listening sink on the way past, so nothing upstream — the retriever, the
session write, usage accounting — can tell the difference.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.streaming.token_sink import TokenSink, active_token_sink
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)

MODULE = "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter"


def _adapter() -> GenericAPIAdapter:
    return GenericAPIAdapter(
        api_key="k",
        model="litellm_proxy/litellm",
        max_completion_tokens=1024,
        name="custom",
        endpoint="http://litellm.test",
    )


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _usage_chunk():
    """The final chunk when stream_options={"include_usage": True}.

    It carries usage and an EMPTY choices list. Indexing choices[0] here is the
    single most common way to break a streaming integration.
    """
    return SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=42))


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _streaming_completion(chunks):
    async def _acompletion(*_args, **kwargs):
        assert kwargs.get("stream") is True
        return _FakeStream(chunks)

    return _acompletion


@contextmanager
def _active(sink: TokenSink):
    """Set/reset the ContextVar explicitly.

    An autouse fixture cannot do this: pytest-asyncio runs each test in its own
    Task, which copies the context, so a `set()` made inside the test never
    reaches the fixture's context — and calling `set(None)` there would instead
    permanently shadow the module default for the collector.
    """
    token = active_token_sink.set(sink)
    try:
        yield sink
    finally:
        active_token_sink.reset(token)


async def _drain(sink):
    return [event async for event in sink]


@pytest.mark.asyncio
async def test_streams_deltas_and_returns_the_complete_string():
    sink = TokenSink()
    chunks = [_chunk("Neon "), _chunk("was "), _chunk("chosen."), _usage_chunk()]

    with (
        _active(sink),
        patch(f"{MODULE}.get_llm_context_config") as cfg,
        patch(f"{MODULE}.litellm.acompletion", new=_streaming_completion(chunks)),
    ):
        cfg.return_value = SimpleNamespace(llm_answer_streaming=True)
        result = await _adapter().acreate_str_output("q", "sys")

    sink.close()
    streamed = "".join(e.text for e in await _drain(sink) if e.type == "delta")

    assert result == "Neon was chosen."
    assert streamed == result, "what the client saw must equal what callers get"


@pytest.mark.asyncio
async def test_usage_chunk_with_empty_choices_does_not_raise():
    """Regression guard for chunk.choices[0] on the include_usage chunk."""
    sink = TokenSink()

    with (
        _active(sink),
        patch(f"{MODULE}.get_llm_context_config") as cfg,
        patch(
            f"{MODULE}.litellm.acompletion",
            new=_streaming_completion([_chunk("ok"), _usage_chunk()]),
        ),
    ):
        cfg.return_value = SimpleNamespace(llm_answer_streaming=True)
        assert await _adapter().acreate_str_output("q", "sys") == "ok"


@pytest.mark.asyncio
async def test_role_only_and_empty_chunks_are_skipped():
    """Providers send a leading role-only chunk and trailing finish chunks whose
    delta.content is None."""
    sink = TokenSink()
    chunks = [_chunk(None), _chunk("real"), _chunk(""), _chunk(None)]

    with (
        _active(sink),
        patch(f"{MODULE}.get_llm_context_config") as cfg,
        patch(f"{MODULE}.litellm.acompletion", new=_streaming_completion(chunks)),
    ):
        cfg.return_value = SimpleNamespace(llm_answer_streaming=True)
        assert await _adapter().acreate_str_output("q", "sys") == "real"

    sink.close()
    assert [e.text for e in await _drain(sink) if e.type == "delta"] == ["real"]


@pytest.mark.asyncio
async def test_no_sink_takes_the_original_non_streaming_path():
    """Flag on, but nobody listening — must not request a stream at all."""
    seen = {}

    async def _acompletion(*_args, **kwargs):
        seen["stream"] = kwargs.get("stream")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="blocking"))]
        )

    with (
        patch(f"{MODULE}.get_llm_context_config") as cfg,
        patch(f"{MODULE}.litellm.acompletion", new=_acompletion),
    ):
        cfg.return_value = SimpleNamespace(llm_answer_streaming=True)
        assert await _adapter().acreate_str_output("q", "sys") == "blocking"

    assert seen["stream"] is None


@pytest.mark.asyncio
async def test_flag_off_ignores_an_active_sink():
    """The whole PR is inert until the flag is turned on."""
    sink = TokenSink()
    seen = {}

    async def _acompletion(*_args, **kwargs):
        seen["stream"] = kwargs.get("stream")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="blocking"))]
        )

    with (
        _active(sink),
        patch(f"{MODULE}.get_llm_context_config") as cfg,
        patch(f"{MODULE}.litellm.acompletion", new=_acompletion),
    ):
        cfg.return_value = SimpleNamespace(llm_answer_streaming=False)
        assert await _adapter().acreate_str_output("q", "sys") == "blocking"

    assert seen["stream"] is None
    sink.close()
    assert await _drain(sink) == []


@pytest.mark.asyncio
async def test_mid_stream_failure_propagates_so_retry_still_fires():
    """Swallowing this would defeat the tenacity retry on the caller."""
    sink = TokenSink()

    class _Exploding:
        def __aiter__(self):
            async def _gen():
                yield _chunk("partial")
                raise RuntimeError("connection reset")

            return _gen()

    async def _acompletion(*_args, **_kwargs):
        return _Exploding()

    with (
        _active(sink),
        patch(f"{MODULE}.get_llm_context_config") as cfg,
        patch(f"{MODULE}.litellm.acompletion", new=_acompletion),
    ):
        cfg.return_value = SimpleNamespace(llm_answer_streaming=True)
        with pytest.raises(RuntimeError, match="connection reset"):
            await _adapter().acreate_str_output("q", "sys")
