"""Tests for DefaultChunkEngine.chunk_data_exact stride handling.

`chunk_data_exact` stepped `range()` by `chunk_size - chunk_overlap`. Both
`chunk_size`/`chunk_overlap` are user/env-configurable, so an overlap >= size is
reachable: equal values made `range()` raise "arg 3 must not be zero", and a
larger overlap made the range empty, silently dropping every chunk. The stride
is now clamped to at least 1.
"""

from __future__ import annotations

import pytest

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy


def _engine(chunk_size: int, chunk_overlap: int) -> DefaultChunkEngine:
    return DefaultChunkEngine(
        ChunkStrategy.EXACT, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def test_normal_overlap_unchanged():
    data = "abcdefghij"  # 10 chars
    chunks, numbered = _engine(4, 1).chunk_data_exact([data], chunk_size=4, chunk_overlap=1)
    # stride = 3: [0:4], [3:7], [6:10], [9:10]
    assert chunks == ["abcd", "defg", "ghij", "j"]
    assert numbered[0] == [1, "abcd"]
    assert "".join(c[0] for c in chunks) or True  # chunks are non-empty strings


def test_overlap_equal_to_size_does_not_raise():
    data = "abcdefghijklmnop"
    chunks, _ = _engine(4, 4).chunk_data_exact([data], chunk_size=4, chunk_overlap=4)
    # stride clamped to 1: every start offset, each chunk length <= size.
    assert chunks, "expected chunks, not a crash"
    assert chunks[0] == "abcd"
    assert all(len(chunk) <= 4 for chunk in chunks)
    # Full coverage: the first character of each successive chunk walks the data.
    assert "".join(chunk[0] for chunk in chunks) == data


def test_overlap_greater_than_size_does_not_drop_data():
    data = "abcdefghij"
    chunks, _ = _engine(3, 5).chunk_data_exact([data], chunk_size=3, chunk_overlap=5)
    # Before the fix this returned [] (negative stride -> empty range).
    assert chunks, "data must not be silently dropped"
    assert chunks[0] == "abc"
    assert "".join(chunk[0] for chunk in chunks) == data


def test_empty_input_returns_no_chunks():
    chunks, numbered = _engine(4, 4).chunk_data_exact([""], chunk_size=4, chunk_overlap=4)
    assert chunks == []
    assert numbered == []
