"""OpenRouter with EMBEDDING_ENCODING_FORMAT=base64: whichever way it answers, cognee
handles it on the first call.

cognee used to force float on every OpenRouter route; the setting now wins. Nothing
has recorded what OpenRouter does with a base64 request, so this accepts each handled
outcome and fails on anything else:

- honoured: a real float vector of the configured size, never 0/1 values;
- ignored (float answered): EmbeddingEncodingError at once, naming the setting;
- rejected: a 400/422 whose message names the encoding, so the hint fits it.

One short embedding request, no LLM call. Config comes from the environment:
EMBEDDING_PROVIDER=custom, an openrouter/ model, OPENROUTER key as
EMBEDDING_API_KEY, EMBEDDING_DIMENSIONS, and EMBEDDING_ENCODING_FORMAT=base64.
"""

import asyncio
import math
import time

import litellm

from cognee.infrastructure.databases.exceptions import EmbeddingEncodingError
from cognee.infrastructure.databases.vector.embeddings.encoding import rejects_encoding_format
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)

# Count requests: a terminal error answers on the first one, and a retried error shows
# as more than one, however slow the runner or the provider is. The engine looks up
# litellm.aembedding on every call, so wrapping it here is enough.
attempts = 0
_aembedding = litellm.aembedding


async def _counted_aembedding(*args, **kwargs):
    global attempts
    attempts += 1
    return await _aembedding(*args, **kwargs)


litellm.aembedding = _counted_aembedding


async def main() -> None:
    engine = get_embedding_engine()
    assert engine.encoding_format == "base64", f"engine picked {engine.encoding_format}"

    started = time.perf_counter()
    try:
        vectors = await engine.embed_text(["hello"])
    except EmbeddingEncodingError as error:
        elapsed = time.perf_counter() - started
        assert "EMBEDDING_ENCODING_FORMAT" in str(error), str(error)
        assert attempts == 1, f"terminal error after {attempts} requests: it was retried"
        print(f"OK (ignored): EmbeddingEncodingError after {elapsed:.1f}s: {error}")
        return
    except (
        litellm.exceptions.BadRequestError,
        litellm.exceptions.UnprocessableEntityError,
    ) as error:
        elapsed = time.perf_counter() - started
        assert rejects_encoding_format(error), f"rejection does not name the encoding: {error}"
        assert attempts == 1, f"terminal error after {attempts} requests: it was retried"
        print(f"OK (rejected): {type(error).__name__} after {elapsed:.1f}s: {error}")
        return

    (vector,) = vectors
    assert len(vector) == engine.dimensions, f"{len(vector)} values, expected {engine.dimensions}"
    assert all(isinstance(v, float) and math.isfinite(v) for v in vector), "not finite floats"
    assert not set(vector) <= {0.0, 1.0}, "a 0/1 vector: a binary format was stored as floats"
    print(f"OK (honoured): {len(vector)} floats, first values {vector[:3]}")


if __name__ == "__main__":
    asyncio.run(main())
