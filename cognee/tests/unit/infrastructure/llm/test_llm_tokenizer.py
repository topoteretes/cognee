"""The tokenizer the LLM-side token counters use."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "model,encoding", [("openai/gpt-4o", "o200k_base"), ("custom/unknown-model", "cl100k_base")]
)
def test_llm_tokenizer_uses_model_encoding_or_default(monkeypatch, model, encoding):
    from cognee.infrastructure.llm import config
    from cognee.infrastructure.llm.utils import get_llm_tokenizer

    monkeypatch.setattr(config, "get_llm_config", lambda: SimpleNamespace(llm_model=model))
    assert get_llm_tokenizer().tokenizer.name == encoding
