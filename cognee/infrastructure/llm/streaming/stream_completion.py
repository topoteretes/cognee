"""The one streaming plain-text completion, shared by every adapter that has one.

Adapters differ in how they build a request, not in how a stream is consumed, so
the consumption lives here. Keeping it in a single place is what stops streaming
from being a property of whichever adapter happened to get the feature: the
gateway picks an adapter from ``STRUCTURED_OUTPUT_FRAMEWORK`` *and*
``LLM_PROVIDER``, and a hook added to only one of those combinations is dead code
for every other deployment — including the defaults.

The contract is the same for all of them: **return exactly what the blocking call
would have returned**, and push the tokens to the sink on the way past. Anything
that would make the two differ — a dropped parameter, an empty answer where the
blocking path raises — is a bug here, not a quirk of streaming.
"""

from __future__ import annotations

from typing import Any, Optional

import litellm

from cognee.shared.rate_limiting import llm_rate_limiter_context_manager
from cognee.infrastructure.llm.streaming.token_sink import TokenSink
from cognee.shared.logging_utils import get_logger

logger = get_logger("stream_completion")


async def stream_text_completion(
    *,
    sink: TokenSink,
    model: str,
    system_prompt: str,
    text_input: str,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    adapter_name: str = "LLM",
    **merged_kwargs: Any,
) -> str:
    """Stream a completion into ``sink`` and return the complete text.

    ``stream_options={"include_usage": True}`` is requested so a streamed call
    reports the same usage a blocking one does. Without it an OpenAI-compatible
    stream returns no usage block at all, and any accounting that reads it — the
    cloud credit guard among them — sees streamed answers as free. It is not
    paired with ``drop_params``: that flag is global to the call, so tolerating
    one unsupported key would silently discard *any* other, and the same query
    would answer differently depending on whether a consumer was listening. A
    server that rejects ``stream_options`` therefore fails loudly rather than
    quietly changing the sampling parameters.
    """
    # A caller-supplied value would collide with the keyword below and raise
    # TypeError on every completion.
    merged_kwargs.pop("stream", None)
    # Merge rather than replace: a caller-supplied dict is truthy, so `or` would
    # drop include_usage entirely. Without it the provider sends no usage chunk,
    # LiteLLM records no spend for the call, and the cloud credit guard bills
    # nothing for a streamed answer — the failure the spend check exists to catch.
    caller_options = merged_kwargs.pop("stream_options", None) or {}
    stream_options = {**caller_options, "include_usage": True}

    parts: list[str] = []
    sink.begin_attempt()

    # The iteration stays inside the rate limiter so a mid-stream failure is
    # still seen by the overload policy. Note this does NOT bound how many
    # generations run at once: AsyncLimiter caps request *rate* and releases
    # nothing on exit, so concurrency here is unbounded either way.
    async with llm_rate_limiter_context_manager():
        stream = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input},
            ],
            api_key=api_key,
            api_base=endpoint,
            api_version=api_version,
            stream=True,
            stream_options=stream_options,
            **merged_kwargs,
        )
        try:
            async for chunk in stream:
                # Some providers still send a trailing usage chunk with
                # `choices == []`; indexing it is the single most common way to
                # break a streaming integration.
                if not chunk.choices:
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                if delta is None:
                    # Some proxies emit a final chunk with a populated choices
                    # list and no delta at all.
                    continue
                piece = getattr(delta, "content", None)
                if not piece:  # role-only and finish chunks carry None
                    continue
                if not isinstance(piece, str):
                    # Multimodal/content-block deltas arrive as a list; joining
                    # those at the end would raise TypeError instead.
                    logger.debug("Skipping non-text stream delta of type %s", type(piece))
                    continue
                parts.append(piece)
                sink.put_delta(piece)
        finally:
            # litellm hands back a CustomStreamWrapper, not a native async
            # generator, so abandoning iteration runs no cleanup at all — every
            # failed attempt would leak its underlying HTTP response, and
            # tenacity retries the whole call.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 - cleanup must not mask the real error
                    logger.debug("Failed to close LLM stream", exc_info=True)

    # An empty result is returned, never raised. The blocking path is
    # `response.choices[0].message.content or ""`, which raises *only* when the
    # choices list itself is empty — a content-filter refusal, a reasoning-only
    # reply and a tool-call-only reply all come back as "" there. Raising here
    # instead would not just break parity: ValueError is retryable, and the stop
    # condition is `stop_after_attempt(2) & stop_after_delay(240)` — an `and`, so
    # it keeps retrying until 240s have elapsed. The same query would answer
    # instantly with the flag off and burn four minutes with it on.
    #
    # Exceptions from the stream itself still propagate: the tenacity retry must
    # fire for those, and begin_attempt() emits a `reset` on re-entry so the
    # consumer discards the partial answer.
    return "".join(parts)
