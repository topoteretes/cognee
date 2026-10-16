"""Bedrock with EMBEDDING_ENCODING_FORMAT=base64 fails on the first call, for good.

litellm maps base64 to Titan's 1-bit "binary" embedding type, so Bedrock answers
with a list instead of a base64 string. The engine must raise
EmbeddingEncodingError at once, name the setting, and not walk the retry ladder.

One short embedding request against the real service, no LLM call. Config comes
from the environment: EMBEDDING_PROVIDER=bedrock, Bedrock credentials, and
EMBEDDING_ENCODING_FORMAT=base64.
"""

import asyncio
import time

from cognee.infrastructure.databases.exceptions import EmbeddingEncodingError
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)

# A terminal error is raised on the first reply. Anything near the retry ladder's
# minutes means it was retried.
MAX_SECONDS = 30


async def main() -> None:
    engine = get_embedding_engine()
    assert engine.encoding_format == "base64", f"engine picked {engine.encoding_format}"

    started = time.perf_counter()
    try:
        vectors = await engine.embed_text(["hello"])
    except EmbeddingEncodingError as error:
        elapsed = time.perf_counter() - started
        assert "EMBEDDING_ENCODING_FORMAT" in str(error), str(error)
        assert elapsed < MAX_SECONDS, f"terminal error took {elapsed:.0f}s: it was retried"
        print(f"OK: EmbeddingEncodingError after {elapsed:.1f}s: {error}")
        return

    first = vectors[0]
    raise AssertionError(
        f"Bedrock answered a base64 request with a {type(first).__name__} of {len(first)} "
        "values and nothing was raised; expected EmbeddingEncodingError"
    )


if __name__ == "__main__":
    asyncio.run(main())
