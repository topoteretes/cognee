from cognee.infrastructure.data.chunking.DefaultChunkEngine import DefaultChunkEngine
from cognee.shared.data_models import ChunkStrategy


def test_sentence_chunking_splits_long_sentences_into_strings():
    engine = DefaultChunkEngine(ChunkStrategy.SENTENCE, chunk_size=10, chunk_overlap=0)

    chunks, numbered_chunks = engine.chunk_by_sentence(
        ["abcdefghijklmnop."], chunk_size=10, chunk_overlap=0
    )

    assert chunks == ["abcdefghij", "klmnop."]
    assert numbered_chunks == [[1, "abcdefghij"], [2, "klmnop."]]
