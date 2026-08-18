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

    ``stream_options`` is deliberately not requested. It is the parameter
    OpenAI-compatible servers most often reject (llama.cpp, LM Studio, older
    vLLM), and the usual workaround — ``drop_params=True`` — is global to the
    call, so it would silently discard *any* unsupported parameter on the
    streaming path only: same query, same config, but a `temperature` or `seed`
    quietly dropped and a different answer, depending on whether a consumer
    happened to be listening. Token usage is accounted from the returned string
    upstream, so nothing here needs the usage chunk.
    """
    # A caller-supplied value would collide with the keyword below and raise
    # TypeError on every completion.
    merged_kwargs.pop("stream", None)
    merged_kwargs.pop("stream_options", None)

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
            **merged_kwargs,
        )
        try:
            async for chunk in stream:
                # Some providers still send a trailing usage chunk with
                # `choices == []`; indexing it is the single most common way to
                # break a streaming integration.
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if not piece:  # role-only and finish chunks carry None
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

    if not parts:
        # The blocking path raises for the same provider responses — a
        # content-filter refusal, a tool-call-only reply, an error envelope
        # streamed as an immediate [DONE]. Returning "" here instead would hand
        # back an empty answer that `commit_turn` then persists, and would skip
        # the tenacity retry the blocking path gets.
        raise ValueError(f"{adapter_name} streamed no content for a plain-text completion")

    # Exceptions deliberately propagate: the tenacity retry on the caller must
    # still fire, and begin_attempt() emits a `reset` on re-entry so the consumer
    # discards the partial answer.
    return "".join(parts)
