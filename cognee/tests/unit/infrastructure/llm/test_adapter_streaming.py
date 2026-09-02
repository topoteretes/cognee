"""The shared streaming completion, and the adapters that route into it.

The contract that matters: the call returns the same complete string whether or
not it streamed. Streaming only decides whether the tokens are *also* pushed to a
listening sink on the way past, so nothing upstream — the retriever, the session
write, usage accounting — can tell the difference.

Split in two on purpose. The stream mechanics are tested against
``stream_text_completion`` directly, because the adapter entry points are wrapped
in a tenacity retry with exponential backoff and the failure cases would spend
minutes sleeping. The adapters are then tested only for what is theirs: routing
to the streaming path when a sink is active and to the blocking path when not —
for *both* of them, because the gateway picks one from STRUCTURED_OUTPUT_FRAMEWORK
and LLM_PROVIDER, and streaming implemented in only one is dead code on every
other deployment, including the defaults.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.streaming.stream_completion import stream_text_completion
from cognee.infrastructure.llm.streaming.token_sink import (
    TokenSink,
    active_token_sink,
    requested_token_sink,
    answer_scope,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter import (
    GenericAPIAdapter,
)
from cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter import (
    NativeLiteLLMAdapter,
)

STREAM_MODULE = "cognee.infrastructure.llm.streaming.stream_completion"
GENERIC_MODULE = "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.generic_llm_api.adapter"
NATIVE_MODULE = (
    "cognee.infrastructure.llm.structured_output_framework.litellm_native.native_adapter"
)


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _usage_chunk():
    """A trailing usage chunk: carries an EMPTY choices list. Indexing choices[0]
    here is the single most common way to break a streaming integration."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=42))


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()

    async def aclose(self):
        self.closed = True


def _streaming_completion(chunks, seen=None, streams=None):
    async def _acompletion(*_args, **kwargs):
        if seen is not None:
            seen.update(kwargs)
        assert kwargs.get("stream") is True
        stream = _FakeStream(chunks)
        if streams is not None:
            streams.append(stream)
        return stream

    return _acompletion


@contextmanager
def _flag(enabled: bool, adapter_streams: bool = True):
    """Set both preconditions for promotion.

    The capability is pinned for the same reason test_token_sink pins it: the
    probe reads the ambient LLM configuration, and whether promotion happens
    must not depend on what the host environment configures. Left unpinned, a
    host where the probe resolves False skips promotion, the sink is never
    terminated, and the drain below waits forever — a 300s pytest-timeout in
    CI's container job. patch.object, not a string target: the LLMGateway class
    shadows its module, and Python 3.10's mock resolves strings onto the class.
    """
    with (
        patch(
            "cognee.infrastructure.llm.config.get_llm_context_config",
            return_value=SimpleNamespace(llm_answer_streaming=enabled),
        ),
        patch.object(
            LLMGateway,
            "supports_answer_streaming",
            return_value=adapter_streams,
        ),
    ):
        yield


@contextmanager
def _requested(sink: TokenSink):
    token = requested_token_sink.set(sink)
    try:
        yield sink
    finally:
        requested_token_sink.reset(token)


@contextmanager
def _active(sink: TokenSink):
    """Set/reset the ContextVar explicitly.

    An autouse fixture cannot do this: pytest-asyncio runs each test in its own
    Task, which copies the context, so a `set()` made inside the test never
    reaches the fixture's context.
    """
    token = active_token_sink.set(sink)
    try:
        yield sink
    finally:
        active_token_sink.reset(token)


async def _drain(sink):
    return [event async for event in sink]


async def _stream(sink, chunks, **kwargs):
    return await stream_text_completion(
        sink=sink, model="m", system_prompt="sys", text_input="q", **kwargs
    )


# --------------------------- stream mechanics ---------------------------


@pytest.mark.asyncio
async def test_streams_deltas_and_returns_the_complete_string():
    sink = TokenSink()
    chunks = [_chunk("Neon "), _chunk("was "), _chunk("chosen."), _usage_chunk()]

    with patch(f"{STREAM_MODULE}.litellm.acompletion", new=_streaming_completion(chunks)):
        result = await _stream(sink, chunks)

    sink.close()
    streamed = "".join(e.text for e in await _drain(sink) if e.type == "delta")

    assert result == "Neon was chosen."
    assert streamed == result, "what the client saw must equal what callers get"


@pytest.mark.asyncio
async def test_usage_chunk_with_empty_choices_does_not_raise():
    sink = TokenSink()
    chunks = [_chunk("ok"), _usage_chunk()]
    with patch(f"{STREAM_MODULE}.litellm.acompletion", new=_streaming_completion(chunks)):
        assert await _stream(sink, chunks) == "ok"


@pytest.mark.asyncio
async def test_role_only_and_empty_chunks_are_skipped():
    """Providers send a leading role-only chunk and trailing finish chunks whose
    delta.content is None."""
    sink = TokenSink()
    chunks = [_chunk(None), _chunk("real"), _chunk(""), _chunk(None)]

    with patch(f"{STREAM_MODULE}.litellm.acompletion", new=_streaming_completion(chunks)):
        assert await _stream(sink, chunks) == "real"

    sink.close()
    assert [e.text for e in await _drain(sink) if e.type == "delta"] == ["real"]


@pytest.mark.asyncio
async def test_an_empty_stream_returns_the_same_empty_string_as_the_blocking_call():
    """A content-filter refusal, a reasoning-only reply, or a tool-call-only
    reply. The blocking path is `content or ""` — it raises only when the
    *choices list* is empty — so raising here would break parity in the one
    direction that costs real time: ValueError is retryable, and the stop
    condition is `stop_after_attempt(2) & stop_after_delay(240)` (an `and`), so
    the same query would burn four minutes with the flag on and answer instantly
    with it off."""
    sink = TokenSink()
    chunks = [_chunk(None), _usage_chunk()]
    with patch(f"{STREAM_MODULE}.litellm.acompletion", new=_streaming_completion(chunks)):
        assert await _stream(sink, chunks) == ""


@pytest.mark.asyncio
async def test_include_usage_survives_caller_supplied_stream_options():
    """include_usage must be forced on, not merely defaulted.

    A caller-supplied stream_options dict is truthy, so replacing instead of
    merging would silently drop include_usage. The provider then sends no usage
    chunk, LiteLLM records no spend, and a streamed answer bills nothing.
    """
    sink = TokenSink()
    seen: dict = {}
    with (
        _flag(True),
        _active(sink),
        patch(
            f"{STREAM_MODULE}.litellm.acompletion",
            new=_streaming_completion([_chunk("hi")], seen=seen),
        ),
    ):
        await _stream(sink, [_chunk("hi")], stream_options={"other_key": "value"})

    assert seen["stream_options"]["include_usage"] is True
    assert seen["stream_options"]["other_key"] == "value"


@pytest.mark.asyncio
async def test_sampling_parameters_are_not_dropped():
    """drop_params is global to the call, so enabling it to tolerate one
    unsupported key would silently discard temperature/seed as well — the same
    query answering differently depending on whether anyone was listening."""
    sink = TokenSink()
    seen: dict = {}
    chunks = [_chunk("ok")]

    with patch(
        f"{STREAM_MODULE}.litellm.acompletion",
        new=_streaming_completion(chunks, seen=seen),
    ):
        await _stream(sink, chunks, temperature=0.0, seed=7)

    assert seen.get("temperature") == 0.0 and seen.get("seed") == 7
    # drop_params is global to the call, so tolerating one unsupported key would
    # silently discard these two as well.
    assert seen.get("drop_params") is None
    # Usage is requested so a streamed answer is not invisible to spend
    # accounting — the cloud credit guard reads it.
    assert seen.get("stream_options") == {"include_usage": True}


@pytest.mark.asyncio
async def test_the_stream_is_closed_even_when_iteration_fails():
    """litellm returns a CustomStreamWrapper, not a native async generator, so
    abandoning iteration runs no cleanup — every failed attempt would leak its
    underlying HTTP response, and the caller retries the whole call."""
    sink = TokenSink()
    closed = {}

    class _Exploding:
        def __aiter__(self):
            async def _gen():
                yield _chunk("partial")
                raise RuntimeError("connection reset")

            return _gen()

        async def aclose(self):
            closed["yes"] = True

    async def _acompletion(*_args, **_kwargs):
        return _Exploding()

    with patch(f"{STREAM_MODULE}.litellm.acompletion", new=_acompletion):
        with pytest.raises(RuntimeError, match="connection reset"):
            await _stream(sink, [])

    assert closed.get("yes") is True


@pytest.mark.asyncio
async def test_a_retry_tells_the_consumer_to_discard_the_partial_answer():
    """Driven through a promotion, because only the owning lane may reset."""
    sink = TokenSink()
    chunks = [_chunk("complete")]

    with (
        _flag(True),
        _requested(sink),
        patch(f"{STREAM_MODULE}.litellm.acompletion", new=_streaming_completion(chunks)),
    ):
        async with answer_scope():
            sink.put_delta("partial from the first attempt")
            await _stream(sink, chunks)

    assert [e.type for e in await _drain(sink)] == [
        "delta",
        "reset",
        "delta",
        "answer_done",
    ]


# --------------------------- adapter routing ---------------------------


async def _call_generic(**kwargs):
    adapter = GenericAPIAdapter(
        api_key="k",
        model="litellm_proxy/litellm",
        max_completion_tokens=1024,
        name="custom",
        endpoint="http://litellm.test",
    )
    return await adapter.acreate_str_output("q", "sys", **kwargs)


async def _call_native(**kwargs):
    # The native adapter has no acreate_str_output; a plain-text answer reaches
    # it as acreate_structured_output(response_model=str), which is how
    # generate_completion calls the gateway.
    adapter = NativeLiteLLMAdapter(
        api_key="k", model="openai/gpt-5-mini", max_completion_tokens=1024, endpoint=None
    )
    return await adapter.acreate_structured_output("q", "sys", str, **kwargs)


ADAPTERS = [
    pytest.param(_call_generic, GENERIC_MODULE, id="generic"),
    # STRUCTURED_OUTPUT_FRAMEWORK defaults to litellm_native, so this is the
    # adapter an out-of-the-box install answers through. Streaming that skips it
    # reaches nobody — which is exactly what this parameter now prevents.
    pytest.param(_call_native, NATIVE_MODULE, id="litellm_native"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("call,_module", ADAPTERS)
async def test_an_active_sink_routes_to_the_streaming_path(call, _module):
    sink = TokenSink()
    chunks = [_chunk("Neon "), _chunk("won.")]

    with (
        _active(sink),
        patch(f"{STREAM_MODULE}.litellm.acompletion", new=_streaming_completion(chunks)),
    ):
        assert await call() == "Neon won."

    sink.close()
    assert "".join(e.text for e in await _drain(sink) if e.type == "delta") == "Neon won."


@pytest.mark.asyncio
@pytest.mark.parametrize("call,module", ADAPTERS)
async def test_no_sink_takes_the_original_non_streaming_path(call, module):
    """Nobody listening — must not request a stream at all."""
    seen = {}

    async def _acompletion(*_args, **kwargs):
        seen["stream"] = kwargs.get("stream")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="blocking"))]
        )

    with patch(f"{module}.litellm.acompletion", new=_acompletion):
        assert await call() == "blocking"

    assert seen["stream"] is None
