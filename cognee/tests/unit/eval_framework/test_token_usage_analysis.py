import importlib
from pathlib import Path

import pytest
import tiktoken

from cognee.eval_framework.token_usage_analysis import corpus


@pytest.mark.parametrize("model", ["gpt-4", "gpt-4o"])
def test_measure_counts_literal_special_tokens(monkeypatch, model):
    # measure.py does `from cost_model import ...` (script-style sibling import), so its
    # directory must be on sys.path; the import leaves `cost_model` in sys.modules.
    monkeypatch.syspath_prepend(str(Path(corpus.__file__).parent))
    measure = importlib.import_module("cognee.eval_framework.token_usage_analysis.measure")
    text = "Document containing <|endoftext|> and <|fim_prefix|>."

    assert measure.count_tokens(text, model) == len(
        tiktoken.encoding_for_model(model).encode_ordinary(text)
    )


def test_corpus_chunking_preserves_literal_special_tokens(monkeypatch):
    chunk_by_sentence = importlib.import_module("cognee.tasks.chunks.chunk_by_sentence")
    # chunk_text() permanently overwrites chunk_by_sentence.get_embedding_engine. This
    # self-assignment only registers the original with monkeypatch so teardown restores it.
    monkeypatch.setattr(
        chunk_by_sentence, "get_embedding_engine", chunk_by_sentence.get_embedding_engine
    )
    text = "Document containing <|endoftext|>.\nAnother literal <|fim_prefix|>."

    chunks = corpus.chunk_text(text)

    assert "".join(chunks) == text
