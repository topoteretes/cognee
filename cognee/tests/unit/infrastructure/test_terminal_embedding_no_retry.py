"""Verify that the terminal split path (third == 0) raises TerminalEmbeddingException
immediately without hitting the @retry decorator.

Before the fix, the terminal split raised a generic EmbeddingException which the
retry decorator caught and retried with exponential backoff (up to 128 seconds per
attempt).  After the fix the terminal case propagates immediately.

See: https://github.com/topoteretes/cognee/issues/3319
"""

import time
from unittest.mock import Mock, patch

import pytest

from cognee.infrastructure.databases.exceptions import (
    EmbeddingException,
    TerminalEmbeddingException,
)


# ---------------------------------------------------------------------------
# TerminalEmbeddingException IS-A EmbeddingException
# ---------------------------------------------------------------------------


def test_terminal_embedding_exception_is_subclass_of_embedding_exception():
    """TerminalEmbeddingException must remain a subclass so existing except
    blocks that catch EmbeddingException still work."""
    assert issubclass(TerminalEmbeddingException, EmbeddingException)


def test_terminal_embedding_exception_carries_message():
    exc = TerminalEmbeddingException("custom message")
    assert "custom message" in str(exc)
    assert exc.name == "TerminalEmbeddingException"


# ---------------------------------------------------------------------------
# FastembedEmbeddingEngine — terminal split does NOT retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fastembed_terminal_split_does_not_retry():
    """A two-character input that triggers the context-window handler should
    raise TerminalEmbeddingException immediately (no retry backoff)."""
    pytest.importorskip("fastembed")

    with (
        patch(
            "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine."
            "FastembedEmbeddingEngine.get_tokenizer",
            return_value=Mock(),
        ),
        patch(
            "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine."
            "TextEmbedding",
        ) as mock_text_embedding,
    ):
        embedding_model = Mock()
        embedding_model.embed.side_effect = RuntimeError("context window exceeded")
        mock_text_embedding.return_value = embedding_model

        from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
            FastembedEmbeddingEngine,
        )

        engine = FastembedEmbeddingEngine(model="test-model", dimensions=2)

        start = time.monotonic()
        with pytest.raises(TerminalEmbeddingException, match="too short to split further"):
            await engine.embed_text(["ab"])
        elapsed = time.monotonic() - start

        # If retries were happening this would take >=8 seconds (first backoff).
        # A clean fast-fail should complete in well under 2 seconds.
        assert elapsed < 2.0, (
            f"Terminal split took {elapsed:.1f}s — retry decorator is probably still catching it"
        )
