import pytest

from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy


@pytest.mark.parametrize("chunk_overlap", [5, 6])
def test_chunk_data_exact_rejects_overlap_at_or_above_chunk_size(chunk_overlap):
    engine = DefaultChunkEngine(ChunkStrategy.EXACT, chunk_size=5, chunk_overlap=chunk_overlap)

    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        engine.chunk_data_exact(["abcdef"], chunk_size=5, chunk_overlap=chunk_overlap)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "chunk_size must be greater than 0"),
        (5, -1, "chunk_overlap must be greater than or equal to 0"),
    ],
)
def test_chunk_data_exact_rejects_invalid_chunk_options(chunk_size, chunk_overlap, message):
    engine = DefaultChunkEngine(
        ChunkStrategy.EXACT, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    with pytest.raises(ValueError, match=message):
        engine.chunk_data_exact(["abcdef"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
