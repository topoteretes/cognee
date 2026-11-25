"""Embedding input tokens: the provider's own count when it reports one, else the
engine tokenizer's count; counting never fails an embedding."""

import logging
from types import SimpleNamespace

from cognee.infrastructure.databases.vector.embeddings.utils import count_input_tokens


class _UnusedTokenizer:
    def count_tokens(self, text):
        raise AssertionError("the provider reported a count; the tokenizer must not be used")


def test_reported_usage_object_is_used():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=42, total_tokens=50))

    assert count_input_tokens(response, ["a", "b"], _UnusedTokenizer()) == 42


def test_reported_usage_dict_falls_back_to_total_tokens():
    response = {"data": [], "usage": {"total_tokens": 7}}

    assert count_input_tokens(response, ["a"], _UnusedTokenizer()) == 7


def test_ollama_prompt_eval_count_is_used():
    response = {"embeddings": [[0.1]], "prompt_eval_count": 12}

    assert count_input_tokens(response, ["a"], _UnusedTokenizer()) == 12


class _WordTokenizer:
    def count_tokens(self, text):
        return len(text.split())


def test_missing_usage_falls_back_to_the_engine_tokenizer():
    texts = ["two words", "and three words"]

    assert count_input_tokens(SimpleNamespace(usage=None), texts, _WordTokenizer()) == 5
    assert count_input_tokens({"embedding": [0.1]}, texts, _WordTokenizer()) == 5
    assert count_input_tokens(None, texts, _WordTokenizer()) == 5


class _BrokenTokenizer:
    def count_tokens(self, text):
        raise RuntimeError("tokenizer broke")


def test_a_counting_failure_counts_zero_and_is_logged(caplog):
    with caplog.at_level(logging.ERROR):
        assert count_input_tokens(None, ["a"], _BrokenTokenizer()) == 0

    assert "Could not count embedding input tokens" in caplog.text
