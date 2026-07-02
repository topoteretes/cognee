"""Tests for embedding model to tokenizer resolution (issue #3646).

Resolution must pick the tokenizer that matches the configured embedding model
and warn (advisory, never raising) when it has to fall back. These tests patch
the three tokenizer classes in the resolver with sentinels so they assert the
selection logic and warnings only, with no optional deps and no network.
"""

import logging
from unittest.mock import patch

import pytest

from cognee.infrastructure.llm.tokenizer import resolver
from cognee.infrastructure.llm.tokenizer.resolver import resolve_embedding_tokenizer

_MODULE = "cognee.infrastructure.llm.tokenizer.resolver"


class _FakeTokenizer:
    """Stand-in that records the class name and constructor args."""

    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.kwargs = kwargs


def _fakes():
    """Patch TikToken / HuggingFace / Mistral in the resolver with sentinels."""
    return (
        patch(
            f"{_MODULE}.TikTokenTokenizer",
            side_effect=lambda **kw: _FakeTokenizer("tiktoken", **kw),
        ),
        patch(
            f"{_MODULE}.HuggingFaceTokenizer",
            side_effect=lambda **kw: _FakeTokenizer("huggingface", **kw),
        ),
        patch(
            f"{_MODULE}.MistralTokenizer",
            side_effect=lambda **kw: _FakeTokenizer("mistral", **kw),
        ),
    )


def _resolve(**kwargs):
    tik, hf, mis = _fakes()
    with tik, hf, mis:
        return resolve_embedding_tokenizer(**kwargs)


def test_openai_uses_tiktoken_with_bare_model():
    tok = _resolve(provider="openai", model="openai/text-embedding-3-large")
    assert tok.kind == "tiktoken"
    assert tok.kwargs["model"] == "text-embedding-3-large"


def test_gemini_uses_default_tiktoken():
    tok = _resolve(provider="gemini", model="gemini/text-embedding-004")
    assert tok.kind == "tiktoken"
    assert tok.kwargs["model"] is None


def test_mistral_uses_mistral_tokenizer():
    tok = _resolve(provider="mistral", model="mistral/mistral-embed")
    assert tok.kind == "mistral"
    assert tok.kwargs["model"] == "mistral-embed"


def test_fastembed_known_model_resolves_its_hf_tokenizer():
    # BGE is a wordpiece model; it must NOT be counted with the OpenAI BPE tokenizer.
    tok = _resolve(provider="fastembed", model="BAAI/bge-small-en-v1.5")
    assert tok.kind == "huggingface"
    assert tok.kwargs["model"] == "BAAI/bge-small-en-v1.5"


def test_fastembed_unknown_model_falls_back_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        tok = _resolve(provider="fastembed", model="totally-unknown-model")
    assert tok.kind == "tiktoken"
    assert any("not in the known model map" in r.message for r in caplog.records)


def test_openai_compatible_uses_model_as_hf_repo():
    tok = _resolve(provider="openai_compatible", model="BAAI/bge-m3")
    assert tok.kind == "huggingface"
    assert tok.kwargs["model"] == "BAAI/bge-m3"


def test_ollama_prefers_huggingface_override():
    tok = _resolve(
        provider="ollama",
        model="avr/sfr-embedding-mistral:latest",
        huggingface_tokenizer="Salesforce/SFR-Embedding-Mistral",
    )
    assert tok.kind == "huggingface"
    assert tok.kwargs["model"] == "Salesforce/SFR-Embedding-Mistral"


def test_override_differing_from_model_logs_advisory(caplog):
    with caplog.at_level(logging.INFO):
        _resolve(
            provider="ollama",
            model="avr/sfr-embedding-mistral:latest",
            huggingface_tokenizer="Salesforce/SFR-Embedding-Mistral",
        )
    assert any("HUGGINGFACE_TOKENIZER" in r.message for r in caplog.records)


def test_huggingface_load_failure_falls_back_with_warning(caplog):
    tik = patch(
        f"{_MODULE}.TikTokenTokenizer",
        side_effect=lambda **kw: _FakeTokenizer("tiktoken", **kw),
    )
    hf = patch(f"{_MODULE}.HuggingFaceTokenizer", side_effect=RuntimeError("offline"))
    mis = patch(f"{_MODULE}.MistralTokenizer")
    with caplog.at_level(logging.WARNING), tik, hf, mis:
        tok = resolve_embedding_tokenizer(provider="openai_compatible", model="BAAI/bge-m3")
    assert tok.kind == "tiktoken"
    assert any("Falling back" in r.message for r in caplog.records)


def test_no_resolvable_target_falls_back_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        tok = _resolve(provider="custom", model=None)
    assert tok.kind == "tiktoken"
    assert any("No tokenizer could be resolved" in r.message for r in caplog.records)


def test_never_raises_and_returns_tokenizer():
    # Even a fully unknown provider must yield a usable tokenizer, never an error.
    tok = _resolve(provider="something-new", model="some/repo")
    assert tok is not None


def test_bare_model_strips_one_provider_tag():
    assert resolver._bare_model("openai/text-embedding-3-large") == "text-embedding-3-large"
    # Splits once, so a multi-segment repo after the provider tag survives.
    assert resolver._bare_model("hosted_vllm/BAAI/bge-m3") == "BAAI/bge-m3"
    assert resolver._bare_model(None) is None
