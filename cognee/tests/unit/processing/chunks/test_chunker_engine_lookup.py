"""The chunkers resolve the embedding engine once per call, not once per word or pair.

The per-word lookup rebuilt the embedding config for every word; with the keyless
fastembed default that meant a full model-registry scan per word, which held the
event loop for about 15 s per MB of text.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

# The module, not the function the chunks package re-exports under the same name.
chunk_by_sentence_module = importlib.import_module("cognee.tasks.chunks.chunk_by_sentence")


class _WordTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_engine_is_resolved_once_per_call():
    engine = SimpleNamespace(tokenizer=_WordTokenizer())
    text = "One sentence here. Another one follows.\n\nA new paragraph with more words. " * 50

    with patch.object(
        chunk_by_sentence_module, "get_embedding_engine", return_value=engine
    ) as lookup:
        sentences = list(chunk_by_sentence_module.chunk_by_sentence(text, maximum_size=64))

    assert lookup.call_count == 1
    assert "".join(sentence for _, sentence, _, _ in sentences) == text


def test_words_count_as_one_token_without_a_tokenizer():
    engine = SimpleNamespace(tokenizer=None)
    with patch.object(chunk_by_sentence_module, "get_embedding_engine", return_value=engine):
        sizes = [
            size for _, _, size, _ in chunk_by_sentence_module.chunk_by_sentence("a b c. d e.")
        ]

    assert sum(sizes) == len(list(chunk_by_sentence_module.chunk_by_word("a b c. d e.")))


def test_row_chunker_resolves_the_engine_once_per_call():
    chunk_by_row_module = importlib.import_module("cognee.tasks.chunks.chunk_by_row")
    engine = SimpleNamespace(tokenizer=_WordTokenizer())
    rows = "\n\n".join(f"id: {i}, name: row {i}, value: {i * 3}" for i in range(200))

    with patch.object(chunk_by_row_module, "get_embedding_engine", return_value=engine) as lookup:
        chunks = list(chunk_by_row_module.chunk_by_row(rows, max_chunk_size=32))

    assert lookup.call_count == 1
    assert len(chunks) > 1
